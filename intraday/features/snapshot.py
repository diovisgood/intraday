from typing import Sequence, Literal
from collections import OrderedDict
import numpy as np

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class Snapshot(Feature):
    """
    Saves some price of n past frames in the relation to the latest price
    """
    
    def __init__(self,
                 period: int = 10,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        assert isinstance(period, int) and (period > 0)
        super().__init__(write_to=write_to, period=period)
        prefix = f'snapshot_{self.period}_'
        self.names = [
            prefix + 'price',
            prefix + 'proxy',
            prefix + 'iou',
            prefix + 'volume',
            prefix + 'tr',
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-1, 1, shape=(period,)),
                self.names[1]: gym.spaces.Box(0, 1, shape=(period,)),
                self.names[2]: gym.spaces.Box(-1, 1, shape=(period,)),
                self.names[3]: gym.spaces.Box(0, 1, shape=(period,)),
                self.names[4]: gym.spaces.Box(0, 1, shape=(period,)),
            })
        else:
            self.spaces = OrderedDict()
        self.values = np.zeros((period, 5), dtype=np.float32)
        self.empty = True
    
    def reset(self):
        self.values[...] = 0
        self.empty = True
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        close, high, low, volume, tr = frame.close, frame.high, frame.low, frame.volume, frame.true_range
        if self.empty:
            self.values[:, 0] = close
            self.values[:, 1] = high
            self.values[:, 2] = low
            self.values[:, 3] = volume
            self.values[:, 4] = tr
            self.empty = False
        else:
            self.values[1:, :] = self.values[:-1, :]
            self.values[0, 0] = close
            self.values[0, 1] = high
            self.values[0, 2] = low
            self.values[0, 3] = volume
            self.values[0, 4] = tr
        
        p = self.values[:, 0]
        delta = p - close
        m = max(abs(delta.max()), abs(delta.min()))
        price = (delta / m) if (m > 1e-8) else delta
        
        abs_delta = np.abs(delta)
        d = abs_delta.max() - abs_delta.min()
        proxy = (1.0 - (abs_delta / d)) if (d > 1e-8) else (1.0 - abs_delta)
        
        highs, lows = self.values[:, 1], self.values[:, 2]
        intersection = np.minimum(highs, high) - np.maximum(lows, low)
        union = np.maximum(highs, high) - np.minimum(lows, low)
        iou = np.clip(intersection / union, -1.0, 1.0)
        
        v = self.values[:, 3]
        m = v.max() - v.min()
        volume = ((v - v.min()) / m) if (m > 1e-8) else np.ones(self.period, dtype=np.float32)
        
        trs = self.values[:, 4]
        m = trs.max() - trs.min()
        tr = ((trs - trs.min()) / m) if (m > 1e-8) else np.ones(self.period, dtype=np.float32)
        
        if self.write_to_frame:
            setattr(frame, self.names[0], price)
            setattr(frame, self.names[1], proxy)
            setattr(frame, self.names[2], iou)
            setattr(frame, self.names[3], volume)
            setattr(frame, self.names[4], tr)
        if self.write_to_state:
            state[self.names[0]] = price
            state[self.names[1]] = proxy
            state[self.names[2]] = iou
            state[self.names[3]] = volume
            state[self.names[4]] = tr
        return price, proxy, iou, volume, tr
    
    def __repr__(self):
        return f'{self.__class__.__name__}(period={self.period}, write_to={self.write_to})'
