from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class FractalDimension(Feature):
    _eps = 1e-8
    
    def __init__(self,
                 period: int = 10,
                 source: Tuple[str, str] = ('low', 'high'),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        assert isinstance(period, int) and (period > 0)
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, Tuple) and (len(source) == 2)
        self.source = source
        name = f'fractal_dimension_{self.period}_{self.source[0]}_{self.source[1]}'
        self.names = [name]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        result = FractalDimension.calculate(frames[-self.period:], self.source)
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result
    
    @staticmethod
    def calculate(frames: Sequence[Frame], source: Tuple[str, str]):
        period = len(frames)
        if period < 2:
            return 0.0
        half_period = period // 2
        H1, L1 = None, None
        for frame in frames[:half_period]:
            low = getattr(frame, source[0])
            high = getattr(frame, source[1])
            if (H1 is None) or (H1 < high):
                H1 = high
            if (L1 is None) or (L1 > low):
                L1 = low
        H2, L2 = None, None
        for frame in frames[half_period:]:
            low = getattr(frame, source[0])
            high = getattr(frame, source[1])
            if (H2 is None) or (H2 < high):
                H2 = high
            if (L2 is None) or (L2 > low):
                L2 = low
        N1 = (H1 - L1) / half_period
        N2 = (H2 - L2) / (period - half_period)
        N3 = (max(H1, H2) - min(L1, L2)) / period
        return (math.log(N1 + N2 + FractalDimension._eps) - math.log(N3 + FractalDimension._eps)) / math.log(2)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})'
