# import os
import copy
import unittest
from arrow import Arrow
from intraday.frame import Frame
from intraday.processor import Trade, IntervalProcessor, ImbalanceProcessor, RunProcessor

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'data')
# FILE_MASK = 'SPBFUT.RI??.*.trades.gz'


class TestProcessor(unittest.TestCase):
    RegularTrades = [
        Trade(Arrow(2020, 1, 1, 10, 0, 0), 'B', 1, 100),
        Trade(Arrow(2020, 1, 1, 10, 0, 1), 'S', 2, 110),
        Trade(Arrow(2020, 1, 1, 10, 0, 2), 'B', 1, 110),
        Trade(Arrow(2020, 1, 1, 10, 0, 2), 'B', 1, 120),
        Trade(Arrow(2020, 1, 1, 10, 0, 3), 'S', 5, 110),
        Trade(Arrow(2020, 1, 1, 10, 0, 4), 'S', 2, 100),
        Trade(Arrow(2020, 1, 1, 10, 0, 4), 'B', 1, 120),
        Trade(Arrow(2020, 1, 1, 10, 0, 5), 'S', 1, 110),
        Trade(Arrow(2020, 1, 1, 10, 0, 6), 'S', 3, 100),
        Trade(Arrow(2020, 1, 1, 10, 0, 6), 'B', 2, 110),
        Trade(Arrow(2020, 1, 1, 10, 0, 7), 'B', 1, 120),
    ]

    def test_interval_time(self):
        trades = copy.deepcopy(self.RegularTrades)
        processor = IntervalProcessor(method='time', interval=2)
        # Process sequence of trades and collect frames from processor
        actual_frames = []
        for i in range(1, len(trades) + 1):
            frame = processor.process(trades=trades[0:i])
            if frame is not None:
                actual_frames.append(frame)
        # Compare actual frames with expected ones
        expected_frames = [
            Frame(open=100, high=110, low=100, close=110, volume=3, ticks=2, buy_ticks=1, sell_ticks=1),
            Frame(open=110, high=120, low=110, close=110, volume=7, ticks=3, buy_ticks=2, sell_ticks=1),
            Frame(open=100, high=120, low=100, close=110, volume=4, ticks=3, buy_ticks=1, sell_ticks=2),
        ]
        self.assertEqual(len(expected_frames), len(actual_frames))
        for expected, actual in zip(expected_frames, actual_frames):
            for name, expected_value in expected.__dict__.items():
                if name in {'open', 'high', 'low', 'close', 'volume', 'ticks', 'buy_ticks', 'sell_ticks'}:
                    self.assertTrue(hasattr(actual, name))
                    self.assertEqual(expected_value, getattr(actual, name), msg=f'Comparing {name}')

    def test_interval_tick(self):
        trades = copy.deepcopy(self.RegularTrades)
        processor = IntervalProcessor(method='tick', interval=2)
        # Process sequence of trades and collect frames from processor
        actual_frames = []
        for i in range(1, len(trades) + 1):
            frame = processor.process(trades=trades[0:i])
            if frame is not None:
                actual_frames.append(frame)
        # Compare actual frames with expected ones
        expected_frames = [
            Frame(open=100, high=110, low=100, close=110, volume=3, ticks=2, buy_ticks=1, sell_ticks=1),
            Frame(open=110, high=120, low=110, close=110, volume=7, ticks=3, buy_ticks=2, sell_ticks=1),
            Frame(open=100, high=120, low=100, close=110, volume=4, ticks=3, buy_ticks=1, sell_ticks=2),
            Frame(open=100, high=120, low=100, close=120, volume=6, ticks=3, buy_ticks=2, sell_ticks=1),
        ]
        self.assertEqual(len(expected_frames), len(actual_frames))
        for expected, actual in zip(expected_frames, actual_frames):
            for name, expected_value in expected.__dict__.items():
                if name in {'open', 'high', 'low', 'close', 'volume', 'ticks', 'buy_ticks', 'sell_ticks'}:
                    self.assertTrue(hasattr(actual, name))
                    self.assertEqual(expected_value, getattr(actual, name), msg=f'Comparing {name}')

    def test_interval_volume(self):
        trades = copy.deepcopy(self.RegularTrades)
        processor = IntervalProcessor(method='volume', interval=2)
        # Process sequence of trades and collect frames from processor
        actual_frames = []
        for i in range(1, len(trades) + 1):
            frame = processor.process(trades=trades[0:i])
            if frame is not None:
                actual_frames.append(frame)
        # Compare actual frames with expected ones
        expected_frames = [
            Frame(open=100, high=110, low=100, close=110, volume=3, ticks=2, buy_ticks=1, sell_ticks=1),
            Frame(open=110, high=120, low=110, close=110, volume=7, ticks=3, buy_ticks=2, sell_ticks=1),
            Frame(open=100, high=120, low=100, close=110, volume=4, ticks=3, buy_ticks=1, sell_ticks=2),
            Frame(open=100, high=120, low=100, close=120, volume=6, ticks=3, buy_ticks=2, sell_ticks=1),
        ]
        self.assertEqual(len(expected_frames), len(actual_frames))
        for expected, actual in zip(expected_frames, actual_frames):
            for name, expected_value in expected.__dict__.items():
                if name in {'open', 'high', 'low', 'close', 'volume', 'ticks', 'buy_ticks', 'sell_ticks'}:
                    self.assertTrue(hasattr(actual, name))
                    self.assertEqual(expected_value, getattr(actual, name), msg=f'Comparing {name}')

    def test_imbalance(self):
        # TODO Implement test_imbalance on ImbalanceProcessor
        pass

    def test_run(self):
        # TODO Implement test_imbalance on RunProcessor
        pass


if __name__ == '__main__':
    unittest.main()
    