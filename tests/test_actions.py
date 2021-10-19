import unittest
import os
import math
from arrow import Arrow
from typing import Callable, Any
from intraday.providers.moex import MoexArchiveProvider
from intraday.processor import IntervalProcessor, ImbalanceProcessor, RunProcessor
from intraday.features import Copy, KAMA, EMA
from intraday.actions import PingPongAction
from intraday.rewards import ConstantReward
from intraday.env import SingleAgentEnv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'data')
FILE_MASK = 'SPBFUT.RI??.*.trades.gz'
EPISODE_MAX_DURATION = int(4 * 60 * 60)


class TestActions(unittest.TestCase):
    @staticmethod
    def _get_env(method='time', interval=200,
                 period: tuple = (2, 10, 5),
                 initial_balance=300000):
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        if method in ('time', 'tick', 'volume', 'money'):
            processor = IntervalProcessor(method=method, interval=interval)
        elif method in ('ti', 'vi', 'mi'):
            processor = ImbalanceProcessor(method=method, initial_threshold=interval)
        elif method in ('tr', 'vr', 'mr'):
            processor = RunProcessor(method=method, initial_threshold=interval)
        else:
            raise ValueError()

        frames_ema_period = 200

        features_pipeline = [
            KAMA(period=period, source='low', write_to='both'),
            KAMA(period=period, source='high', write_to='both'),
            EMA(
                period=frames_ema_period,
                source=('true_range',),
                write_to='both'
            ),
            Copy(source=('time_start', 'time_end', 'open', 'high', 'low', 'close', 'vwap', 'volume')),
        ]
        action_scheme = PingPongAction()
        reward_scheme = ConstantReward(value=0.0)

        # Initialize environment
        env = SingleAgentEnv(
            provider=provider,
            processor=processor,
            features_pipeline=features_pipeline,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            initial_balance=initial_balance,
        )

        # Return environment
        return env

    @staticmethod
    def _run_episode(env: SingleAgentEnv, model: [None, Callable],
                     episode_start_datetime: [Arrow, None] = None, on_update: [Callable[..., Any], None] = None):
        # Start new episode
        state = env.reset(episode_start_datetime=episode_start_datetime, episode_max_duration=EPISODE_MAX_DURATION)
        if state is None:
            raise RuntimeError('Failed to start new episode')
        episode_start_datetime = env.episode_start_datetime
        states = [state]

        # Subscribe to account updates
        if callable(on_update):
            env.account.subscribe('me', on_update)
        
        # Iterate until episode end
        while True:
            if callable(model):
                action = model(state)
            else:
                action = env.action_scheme.get_default_action()
            state, reward, done, frame = env.step(action)
            states.append(state)
            if done:
                break
        return episode_start_datetime, states
    
    def test_PingPongAction(self):
        for method, interval in [('time', 60)]:
            period = (2, 10, 5)
            env = self._get_env(method=method, interval=interval, period=period)

            # Initialize strategy
            names = [
                f'kama_{"_".join(str(x) for x in period)}_low',
                f'kama_{"_".join(str(x) for x in period)}_high',
                'ema_200_true_range'
            ]

            def price_round(price, step):
                if -math.inf < price < math.inf:
                    return math.floor(0.5 + price / step) * step
                return price

            def ping_pong_strategy(state):
                low, high, atr = state[names[0]], state[names[1]], state[names[2]]
                lower = price_round(low, 10)
                upper = price_round(high, 10)
                trail = atr * 0.5
                stop = atr
                return lower, upper, trail, stop

            # Define account update callback
            def on_update(exchange, account, time):
                a = env.action_scheme
                orders = [
                    a.buy_order_id[account],
                    a.sell_order_id[account],
                    a.buy_stop_order_id[account],
                    a.sell_stop_order_id[account]
                ]
                orders = '\n'.join([str(exchange.get_order(id)) for id in orders])
                print('update', account.position, time, '{\n', orders, '\n}')
            
            # Run two episodes from the same starting point and obtain list of states for each run
            episode_start_datetime = Arrow(2020, 3, 23, 10, 5, 0, tzinfo='Europe/Moscow')
            _, states = self._run_episode(
                env=env,
                model=ping_pong_strategy,
                episode_start_datetime=episode_start_datetime,
                on_update=on_update
            )

    def test_BuySellCloseAction(self):
        # TODO Implement test_BuySellCloseAction
        pass


if __name__ == '__main__':
    unittest.main()
