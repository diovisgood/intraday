import os
import unittest
from numbers import Real
from collections import OrderedDict
from arrow import Arrow
from datetime import date, timedelta

from intraday.frame import Frame
from intraday.providers.moex import MoexArchiveProvider
from intraday.providers.binance import BinanceArchiveProvider
from intraday.processor import IntervalProcessor, ImbalanceProcessor, RunProcessor
from intraday.features import Copy
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from intraday.env import SingleAgentEnv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'data')
FILE_MASK = 'SPBFUT.RI??.*.trades.gz'
EPISODE_MIN_DURATION = timedelta(hours=1)
EPISODE_MAX_DURATION = int(4 * 60 * 60)


class TestEnv(unittest.TestCase):
    @staticmethod
    def _get_env(method='time', interval=200,
                 ema_period_frames=200, ema_period_trades=2000,
                 initial_balance=300000,
                 multiple_providers=False):
        if multiple_providers:
            provider = [
                MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK),
                BinanceArchiveProvider(data_dir=DATA_DIR, symbol='ONTBNB',
                                       date_from=date(2018, 5, 1), date_to=date(2018, 8, 31)),
            ]
        else:
            provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)

        if method in ('time', 'tick', 'volume', 'money'):
            processor = IntervalProcessor(method=method, interval=interval)
        elif method in ('ti', 'vi', 'mi'):
            processor = ImbalanceProcessor(method=method, initial_threshold=interval)
        elif method in ('tr', 'vr', 'mr'):
            processor = RunProcessor(method=method, initial_threshold=interval)
        else:
            raise ValueError()
        # features_pipeline = default_feature_set(trades_ema_period=ema_period_trades, frames_ema_period=ema_period_frames)
        features_pipeline = [Copy()]
        action_scheme = BuySellCloseAction()
        reward_scheme = BalanceReward()
        env = SingleAgentEnv(
            provider=provider,
            processor=processor,
            features_pipeline=features_pipeline,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            initial_balance=initial_balance,
        )
        return env
    
    @staticmethod
    def _run_episode(env: SingleAgentEnv, episode_start_datetime=None):
        states = []
        frames = []
        rewards = []
        state = env.reset(
            episode_start_datetime=episode_start_datetime,
            episode_min_duration=EPISODE_MIN_DURATION,
            episode_max_duration=EPISODE_MAX_DURATION
        )
        if state is None:
            raise RuntimeError('Failed to start new episode')
        episode_start_datetime = env.episode_start_datetime
        states.append(state)
        while True:
            action = env.action_scheme.get_default_action()
            state, reward, done, frame = env.step(action)
            states.append(state)
            frames.append(frame)
            rewards.append(reward)
            if done:
                break
        return episode_start_datetime, states, frames, rewards
    
    def test_episode_repeat(self):
        for method, interval in (('time', 60), ('tick', 100), ('volume', 300), ('tr', 300), ('vr', 300)):
            env = self._get_env(method=method, interval=interval)
            # Run two episodes from the same starting point and obtain list of frames for each run
            # Note: here we compare frames, not states
            # as states can differ on each run because of random initial action
            episode_start_datetime1, _, frames1, *_ = self._run_episode(env, None)
            episode_start_datetime2, _, frames2, *_ = self._run_episode(env, episode_start_datetime1)
            print('method=', method, 'interval=', interval)
            self.maxDiff = None
            self.assertEqual(episode_start_datetime1, episode_start_datetime2)
            self.assertEqual(len(frames1), len(frames2), 'Lengths of frames are not equal')
            for frame1, frame2 in zip(frames1, frames2):
                self.assertEqual(frame1, frame2, 'Frames are not equal')

    def test_seed_repeat(self):
        for method, interval in (('time', 60), ('tick', 100), ('volume', 300), ('tr', 300), ('vr', 300)):
            env = self._get_env(method=method, interval=interval)
            # Run two episodes from the same starting point and obtain list of frames for each run
            # Note: here we compare frames, not states
            # as states can differ on each run because of random initial action
            seed1 = env.seed()
            episode_start_datetime1, _, frames1, *_ = self._run_episode(env)
            seed2 = env.seed(seed=seed1)
            episode_start_datetime2, _, frames2, *_ = self._run_episode(env)
            print('method=', method, 'interval=', interval)
            self.maxDiff = None
            self.assertEqual(episode_start_datetime1, episode_start_datetime2)
            self.assertEqual(len(frames1), len(frames2), 'Lengths of frames are not equal')
            for frame1, frame2 in zip(frames1, frames2):
                self.assertEqual(frame1, frame2, 'Frames are not equal')

    def test_episode_length(self):
        for method, interval in (('time', 60), ('tick', 100), ('volume', 300), ('tr', 300), ('vr', 300)):
            env = self._get_env(method=method, interval=interval)
            print(f'Testing {method}@{interval}')
            _, states, frames, *_ = self._run_episode(env)
            time_start = states[0]['time_start']
            time_end = states[-1]['time_end']
            print(
                f' got {len(states)} with {time_end} - {time_start}',
                f'= {time_end - time_start}'
            )
            self.assertGreaterEqual((time_end - time_start), EPISODE_MIN_DURATION)

    def test_report(self):
        env = self._get_env(method='tr', interval=200)
        self._run_episode(env)
        report = env.account.report
        print(report)
        self.assertTrue(hasattr(report, 'sharpe_ratio'))
        self.assertTrue(hasattr(report, 'sortino_ratio'))
        self.assertTrue(hasattr(report, 'profit_factor'))
        self.assertTrue(hasattr(report, 'rorann'))
        self.assertTrue(hasattr(report, 'net_profit'))
        self.assertTrue(hasattr(report, 'total_commission'))
        
    def test_episode(self):
        env = self._get_env(method='tr', interval=200)
        state = env.reset()
        if state is None:
            raise RuntimeError('Failed to start new episode')
        while True:
            print(state)
            state, reward, done, frame = env.step(0)
            if done:
                break
            self.assertTrue(isinstance(state, OrderedDict))
            self.assertTrue(isinstance(reward, Real))
            self.assertTrue(isinstance(done, bool))
            self.assertTrue(isinstance(frame, Frame))
            
    def test_next_episode(self):
        env = self._get_env(method='time', interval=300)
        state = env.reset(seek='first')
        if state is None:
            raise RuntimeError('Failed to start new episode')
        desired_start = Arrow(2020, 3, 20, 10, 0, 0, tzinfo='Europe/Moscow')
        desired_end = Arrow(2020, 3, 20, 23, 50, 0, tzinfo='Europe/Moscow')
        self.assertAlmostEqual(desired_start.timestamp(), env.provider.episode_start_datetime.timestamp(), 0)
        last_frame = None
        while True:
            state, reward, done, frame = env.step(0)
            if frame is not None:
                last_frame = frame
            if done:
                break
            self.assertTrue(isinstance(state, OrderedDict))
            self.assertTrue(isinstance(reward, Real))
            self.assertTrue(isinstance(done, bool))
            self.assertTrue(isinstance(frame, Frame))
        self.assertAlmostEqual(desired_end.timestamp(), last_frame.time_end.timestamp(), 0)

        state = env.reset(seek='next', keep_state=True)
        if state is None:
            raise RuntimeError('Failed to start new episode')
        desired_start = Arrow(2020, 3, 23, 10, 0, 0, tzinfo='Europe/Moscow')
        desired_end = Arrow(2020, 3, 23, 23, 50, 0, tzinfo='Europe/Moscow')
        self.assertAlmostEqual(desired_start.timestamp(), env.provider.episode_start_datetime.timestamp(), 0)
        while True:
            state, reward, done, frame = env.step(0)
            if frame is not None:
                last_frame = frame
            if done:
                break
            self.assertTrue(isinstance(state, OrderedDict))
            self.assertTrue(isinstance(reward, Real))
            self.assertTrue(isinstance(done, bool))
            self.assertTrue(isinstance(frame, Frame))
        self.assertAlmostEqual(desired_end.timestamp(), last_frame.time_end.timestamp(), 0)

    def test_multiple_providers(self):
        for method, interval in (('time', 60), ('tick', 100), ('volume', 300), ('tr', 300), ('vr', 300)):
            env = self._get_env(method=method, interval=interval, multiple_providers=True)
            # Run two episodes from the same starting point and obtain list of frames for each run
            # Note: here we compare frames, not states
            # as states can differ on each run because of random initial action
            seed1 = env.seed()
            episode_start_datetime1, _, frames1, *_ = self._run_episode(env)
            seed2 = env.seed(seed=seed1)
            episode_start_datetime2, _, frames2, *_ = self._run_episode(env)
            print('method=', method, 'interval=', interval)
            self.maxDiff = None
            self.assertEqual(episode_start_datetime1, episode_start_datetime2)
            self.assertEqual(len(frames1), len(frames2), 'Lengths of frames are not equal')
            for frame1, frame2 in zip(frames1, frames2):
                self.assertEqual(frame1, frame2, 'Frames are not equal')


if __name__ == '__main__':
    unittest.main()
