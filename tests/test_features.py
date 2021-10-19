import unittest
import math
import copy
import numpy as np
from arrow import Arrow
from collections import OrderedDict
from intraday.frame import Frame
from intraday.provider import Trade, TradeOI
from intraday.features import *

_EPS = 1e-8


class TestFeatures(unittest.TestCase):
    RegularTrades = [
        Trade(Arrow(2020, 1, 1, 10, 0, 0), 'B', 1, 100, 1),
        Trade(Arrow(2020, 1, 1, 10, 0, 1), 'S', 2, 100, 2),
        Trade(Arrow(2020, 1, 1, 10, 0, 2), 'B', 1, 100, 3),
        Trade(Arrow(2020, 1, 1, 10, 0, 2), 'B', 1, 100, 4),
        Trade(Arrow(2020, 1, 1, 10, 0, 3), 'S', 5, 100, 5),
        Trade(Arrow(2020, 1, 1, 10, 0, 4), 'S', 2, 100, 6),
        Trade(Arrow(2020, 1, 1, 10, 0, 4), 'B', 1, 100, 7),
        Trade(Arrow(2020, 1, 1, 10, 0, 5), 'S', 1, 100, 8),
        Trade(Arrow(2020, 1, 1, 10, 0, 6), 'S', 3, 100, 9),
        Trade(Arrow(2020, 1, 1, 10, 0, 6), 'B', 2, 100, 10),
        Trade(Arrow(2020, 1, 1, 10, 0, 7), 'B', 1, 100, 11),
    ]

    OnlyBuyTrades = [trade for trade in RegularTrades if trade.operation == 'B']
    
    OnlySellTrades = [trade for trade in RegularTrades if trade.operation == 'S']
    
    RegularFrames = [
        Frame(open=110, high=120, low=105, close=106, volume=10, true_range=15),
        Frame(open=107, high=122, low=110, close=115, volume=12, true_range=16),
        Frame(open=115, high=125, low=115, close=125, volume=11, true_range=10),
        Frame(open=126, high=130, low=126, close=130, volume=9, true_range=5),
        Frame(open=129, high=130, low=120, close=121, volume=5, true_range=10),
        Frame(open=121, high=125, low=115, close=115, volume=7, true_range=10),
        Frame(open=115, high=130, low=112, close=120, volume=9, true_range=18),
        Frame(open=120, high=130, low=120, close=128, volume=10, true_range=10),
        Frame(open=128, high=135, low=125, close=134, volume=11, true_range=10),
        Frame(open=134, high=135, low=130, close=132, volume=10, true_range=4),
    ]

    BadFrames = [
        Frame(open=100, high=100, low=100, close=100, volume=0, true_range=0),
        Frame(open=100, high=100, low=100, close=100, volume=0, true_range=0),
        Frame(open=110, high=110, low=110, close=110, volume=1, true_range=10),
        Frame(open=105, high=105, low=105, close=105, volume=1, true_range=5),
        Frame(open=115, high=115, low=115, close=115, volume=1, true_range=10),
        Frame(open=130, high=130, low=100, close=100, volume=2, true_range=30),
    ]

    def test_Snapshot(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature for period less than number of frames
        feature = Snapshot(period=4, write_to='state')
        state = OrderedDict()
        feature.process(frames[:1], state)
        self.assertTrue(np.all(state['snapshot_4_price'] == np.zeros(4, dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_proxy'] == np.ones(4, dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_iou'] == np.ones(4, dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_volume'] == np.ones(4, dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_tr'] == np.ones(4, dtype=np.float32)))

        feature.process(frames[:2], state)
        iou = (120 - 110) / (122 - 105)
        self.assertTrue(np.all(state['snapshot_4_price'] == np.array([0, -1, -1, -1], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_proxy'] == np.array([1, 0, 0, 0], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_iou'] == np.array([1, iou, iou, iou], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_volume'] == np.array([1, 0, 0, 0], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_tr'] == np.array([1, 0, 0, 0], dtype=np.float32)))

        feature.process(frames[:2], state)
        iou1 = (120 - 110) / (122 - 105)
        iou2 = (120 - 110) / (122 - 105)
        self.assertTrue(np.all(state['snapshot_4_price'] == np.array([0, -1, -1, -1], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_proxy'] == np.array([1, 0, 0, 0], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_iou'] == np.array([1, iou, iou, iou], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_volume'] == np.array([1, 0, 0, 0], dtype=np.float32)))
        self.assertTrue(np.all(state['snapshot_4_tr'] == np.array([1, 0, 0, 0], dtype=np.float32)))

    def test_EfficiencyRatio(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature for period less than number of frames
        feature = EfficiencyRatio(period=5, source='close', write_to='state')
        # Calculate kama_value manually
        volatility = (
            abs(frames[-5].close - frames[-4].close) +
            abs(frames[-4].close - frames[-3].close) +
            abs(frames[-3].close - frames[-2].close) +
            abs(frames[-2].close - frames[-1].close)
        )
        change = frames[-1].close - frames[-5].close
        value = (change / volatility) if (volatility != 0) else 0.0
        # Test process_trade for last 5 frames
        state = OrderedDict()
        feature.process(frames, state)
        self.assertEqual(state['efficiency_ratio_5_close'], value)

        # Test for period which exceeds the number of frames
        feature = EfficiencyRatio(period=11, source='open', write_to='frame')
        # Calculate kama_value for all available frames
        volatility = (
            abs(frames[-10].open - frames[-9].open) +
            abs(frames[-9].open - frames[-8].open) +
            abs(frames[-8].open - frames[-7].open) +
            abs(frames[-7].open - frames[-6].open) +
            abs(frames[-6].open - frames[-5].open) +
            abs(frames[-5].open - frames[-4].open) +
            abs(frames[-4].open - frames[-3].open) +
            abs(frames[-3].open - frames[-2].open) +
            abs(frames[-2].open - frames[-1].open)
        )
        change = frames[-1].open - frames[-10].open
        value = (change / volatility) if (volatility != 0) else 0.0
        # Test process_trade for all available frames
        state = OrderedDict()
        feature.process(frames, state)
        self.assertEqual(frames[-1].efficiency_ratio_11_open, value)

    def test_MarketDimension(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature for period less than number of frames
        feature = MarketDimension(period=5, write_to='state')
        # Calculate kama_value manually
        lowest = min(frames[-5].low, frames[-4].low, frames[-3].low, frames[-2].low, frames[-1].low)
        highest = max(frames[-5].high, frames[-4].high, frames[-3].high, frames[-2].high, frames[-1].high)
        S1 = (highest - lowest)
        S2 = S1 * 5
        SN = (
            (frames[-5].high - frames[-5].low) +
            (frames[-4].high - frames[-4].low) +
            (frames[-3].high - frames[-3].low) +
            (frames[-2].high - frames[-2].low) +
            (frames[-1].high - frames[-1].low)
        )
        value = ((SN - S1) / (S2 - S1)) if (S1 > 0) else 0.5
        # Test process_trade for last 5 frames
        state = OrderedDict()
        feature.process(frames, state)
        self.assertEqual(state['market_dimension_5_low_high'], value)

        # Test for period which exceeds the number of frames
        feature = MarketDimension(period=11, write_to='frame')
        # Calculate kama_value for all available frames
        lowest = min(frames[-10].low, frames[-9].low, frames[-8].low, frames[-7].low, frames[-6].low,
                     frames[-5].low, frames[-4].low, frames[-3].low, frames[-2].low, frames[-1].low)
        highest = max(frames[-10].high, frames[-9].high, frames[-8].high, frames[-7].high, frames[-6].high,
                      frames[-5].high, frames[-4].high, frames[-3].high, frames[-2].high, frames[-1].high)
        S1 = (highest - lowest)
        S2 = S1 * 10
        SN = (
            (frames[-10].high - frames[-10].low) +
            (frames[-9].high - frames[-9].low) +
            (frames[-8].high - frames[-8].low) +
            (frames[-7].high - frames[-7].low) +
            (frames[-6].high - frames[-6].low) +
            (frames[-5].high - frames[-5].low) +
            (frames[-4].high - frames[-4].low) +
            (frames[-3].high - frames[-3].low) +
            (frames[-2].high - frames[-2].low) +
            (frames[-1].high - frames[-1].low)
        )
        value = ((SN - S1) / (S2 - S1)) if (S1 > 0) else 0.5
        # Test process_trade for all available frames
        state = OrderedDict()
        feature.process(frames, state)
        self.assertEqual(frames[-1].market_dimension_11_low_high, value)

    def test_FractalDimension(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature for period less than number of frames
        feature = FractalDimension(period=5, write_to='state')
        # Calculate kama_value manually
        L1 = min(frames[-5].low, frames[-4].low)
        H1 = max(frames[-5].high, frames[-4].high)
        L2 = min(frames[-3].low, frames[-2].low, frames[-1].low)
        H2 = max(frames[-3].high, frames[-2].high, frames[-1].high)
        N1 = (H1 - L1) / 2
        N2 = (H2 - L2) / (5 - 2)
        N3 = (max(H1, H2) - min(L1, L2)) / 5
        value = (math.log(N1 + N2 + _EPS) - math.log(N3 + _EPS)) / math.log(2)
        # Test process_trade for last 5 frames
        state = OrderedDict()
        feature.process(frames, state)
        self.assertEqual(state['fractal_dimension_5_low_high'], value)

        # Test for period which exceeds the number of frames
        feature = FractalDimension(period=11, write_to='frame')
        # Calculate kama_value for all available frames
        L1 = min(frames[-10].low, frames[-9].low, frames[-8].low, frames[-7].low, frames[-6].low)
        H1 = max(frames[-10].high, frames[-9].high, frames[-8].high, frames[-7].high, frames[-6].high)
        L2 = min(frames[-5].low, frames[-4].low, frames[-3].low, frames[-2].low, frames[-1].low)
        H2 = max(frames[-5].high, frames[-4].high, frames[-3].high, frames[-2].high, frames[-1].high)
        N1 = (H1 - L1) / 5
        N2 = (H2 - L2) / 5
        N3 = (max(H1, H2) - min(L1, L2)) / 10
        value = (math.log(N1 + N2 + _EPS) - math.log(N3 + _EPS)) / math.log(2)
        # Test process_trade for all available frames
        state = OrderedDict()
        feature.process(frames, state)
        self.assertEqual(frames[-1].fractal_dimension_11_low_high, value)

    def test_KAMA(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature
        feature = KAMA(period=(1, 5, 3), source='close', write_to='both')
        # Process last two frames manually
        value = frames[-2].close
        volatility = (
            abs(frames[-3].close - frames[-2].close) +
            abs(frames[-2].close - frames[-1].close)
        )
        change = frames[-1].close - frames[-3].close
        efficiency_ratio = (change / volatility) if (volatility != 0) else 0.0
        fast_ema_factor = 2.0 / (1.0 + 1.0)
        slow_ema_factor = 2.0 / (5.0 + 1.0)
        ema_factor = (slow_ema_factor + abs(efficiency_ratio) * (fast_ema_factor - slow_ema_factor)) ** 2.0
        value = value + ema_factor * (frames[-1].close - value)

        # Test process_trade for last 2 frames
        state = OrderedDict()
        feature.process(frames[:-1], OrderedDict())
        feature.process(frames, state)
        self.assertEqual(state['kama_1_5_3_close'], value)
        self.assertEqual(frames[-1].kama_1_5_3_close, value)

    def test_HeikenAshi(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature
        feature = HeikenAshi(write_to='both')
        # Calculate values manually
        last_frame = frames[-1]
        if len(frames) > 1:
            prev_frame = frames[-2]
            open = (prev_frame.open + prev_frame.close) / 2
        else:
            open = last_frame.open
        high = last_frame.high
        low = last_frame.low
        close = (last_frame.open + last_frame.high + last_frame.low + last_frame.close) / 4
        # Test process_trade for last frame
        state = OrderedDict()
        feature.process(frames, state)
        self.assertEqual(state['heiken_open'], open)
        self.assertEqual(state['heiken_high'], high)
        self.assertEqual(state['heiken_low'], low)
        self.assertEqual(state['heiken_close'], close)
        self.assertEqual(frames[-1].heiken_open, open)
        self.assertEqual(frames[-1].heiken_high, high)
        self.assertEqual(frames[-1].heiken_low, low)
        self.assertEqual(frames[-1].heiken_close, close)

    def test_ExponentialMovingAverage(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature
        feature = EMA(period=3, source=('close',), write_to='both')
        # Calculate ema value manually
        ema_factor = 2 / (3 + 1)
        value = frames[0].close
        value = (value + frames[1].close) / 2
        value = (value * 2 + frames[2].close) / 3
        value = value * (1 - ema_factor) + frames[3].close * ema_factor
        value = value * (1 - ema_factor) + frames[4].close * ema_factor
        value = value * (1 - ema_factor) + frames[5].close * ema_factor
        value = value * (1 - ema_factor) + frames[6].close * ema_factor
        value = value * (1 - ema_factor) + frames[7].close * ema_factor
        value = value * (1 - ema_factor) + frames[8].close * ema_factor
        value = value * (1 - ema_factor) + frames[9].close * ema_factor
        # Test process_trade for last frame
        state = OrderedDict()
        for i in range(len(frames)):
            feature.process(frames[:i + 1], state)
        self.assertEqual(state['ema_3_close'], value)
        self.assertEqual(frames[-1].ema_3_close, value)

    def test_Copy(self):
        # TODO Implement test_CopyAttributes
        pass

    def test_AverageTrade(self):
        # TODO Implement test_AverageTrade
        pass

    def test_PriceMoveEase(self):
        # TODO Implement test_PriceMoveEase
        pass

    def test_Delta(self):
        # TODO Implement test_Delta
        pass

    def test_Change(self):
        # TODO Implement test_Change
        pass

    def test_PriceEncoder(self):
        # TODO Implement test_PriceEncoder
        pass

    def test_TimeEncoder(self):
        # TODO Implement test_TimeEncoder
        pass

    def test_GaussianSmooth(self):
        # TODO Implement test_GaussianSmooth
        pass

    def test_FractalExtremums(self):
        # TODO Implement test_FractalExtremums
        pass

    def test_CumulativeSumFilter(self):
        # TODO Implement test_CumulativeSumFilter
        pass

    def test_Ehlers(self):
        frames = copy.deepcopy(self.RegularFrames)
        # Test feature
        feature = Ehlers(write_to='both')
        # Test process_trade for last frame
        state = OrderedDict()
        for i in range(len(frames)):
            feature.process(frames[:i + 1], state)


if __name__ == '__main__':
    unittest.main()
