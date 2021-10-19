import os
import unittest
from datetime import date, timedelta

import numpy as np

from intraday.providers import *
from intraday.processor import IntervalProcessor
from intraday.features import *
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from intraday.env import SingleAgentEnv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'data')
FILE_MASK = 'SPBFUT.RI??.*.trades.gz'
EPISODE_MIN_DURATION = timedelta(hours=100)
EPISODE_MAX_DURATION = timedelta(hours=100)


class TestEnvRender(unittest.TestCase):
    @staticmethod
    def _get_env(initial_balance=300000):
        # provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        provider = [
            # SineProvider(period=timedelta(minutes=35)),
            # MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK),
            BinanceKlines(data_dir=DATA_DIR, symbol='ETHUSDT', spread=0.0007,
                          date_from=date(2018, 5, 1), date_to=date(2018, 8, 31)),
            # BinanceArchiveProvider(data_dir=DATA_DIR, symbol='ONTBNB',
            #                        date_from=date(2018, 5, 1), date_to=date(2018, 8, 31)),
        ]
        processor = IntervalProcessor(method='time', interval=60*60)
        period = 1000
        atr_name = f'ema_{period}_true_range'
        features_pipeline = [
            EMA(period=period, source='true_range', write_to='frame'),
        ]
        action_scheme = BuySellCloseAction()
        reward_scheme = BalanceReward(norm_factor=atr_name)
        env = SingleAgentEnv(
            provider=provider,
            processor=processor,
            features_pipeline=features_pipeline,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            initial_balance=initial_balance,
            delay_per_second=0.1,
            warm_up_time=timedelta(hours=10)
        )
        print(repr(env))
        return env
    
    @staticmethod
    def _run_episode(env: SingleAgentEnv, episode_start_datetime=None, render=True):
        state = env.reset(
            episode_start_datetime=episode_start_datetime,
            episode_min_duration=EPISODE_MIN_DURATION,
            episode_max_duration=EPISODE_MAX_DURATION
        )
        while True:
            if render:
                env.render('human')
            action = env.action_space.sample()
            state, reward, done, frame = env.step(action)
            print(state)
            if done:
                break
    
    def test_render(self):
        env = self._get_env()
        for _ in range(1):
            self._run_episode(env)
        env.close()
    
    def test_example(self):
        from datetime import date, timedelta
        from intraday.providers import BinanceArchiveProvider
        from intraday.processor import IntervalProcessor
        from intraday.features import EMA, Copy, PriceEncoder
        from intraday.actions import BuySellCloseAction
        from intraday.rewards import BalanceReward
        from intraday.env import SingleAgentEnv
    
        provider = BinanceArchiveProvider(data_dir='.', symbol='ETHUSDT',
                                          date_from=date(2018, 5, 1), date_to=date(2018, 5, 31))
        processor = IntervalProcessor(method='time', interval=5 * 60)
        period = 1000
        atr_name = f'ema_{period}_true_range'
        features_pipeline = [
            PriceEncoder(source='close', write_to='both'),
            EMA(period=period, source='true_range', write_to='frame'),
            Copy(source=['volume'])
        ]
        action_scheme = BuySellCloseAction()
        reward_scheme = BalanceReward(norm_factor=atr_name)
        env = SingleAgentEnv(
            provider=provider,
            processor=processor,
            features_pipeline=features_pipeline,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            initial_balance=1000000,
            warm_up_time=timedelta(hours=1)
        )
    
        state = env.reset(episode_max_duration=timedelta(hours=10))
        while True:
            env.render('human')
            action = action_scheme.get_random_action()
            state, reward, done, frame = env.step(action)
            if done:
                break
    
        env.close()


if __name__ == '__main__':
    unittest.main()
