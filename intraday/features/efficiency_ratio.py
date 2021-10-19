from typing import Sequence, Literal
from collections import OrderedDict

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class EfficiencyRatio(Feature):
    """
    Computes the Efficiency Ratio - how effective price moved over some period in time

    Notes
    -----
    Imagine two cases of market price movement:
    Case 1, price chart:
    ● start
    |  /\
    | | |
    \/  ●  end

    Case 2, price chart:
    ● start
     \
       \
        ●  end

    In both cases price moved from starting level down to ending level over five time periods (5 candles).
    But in case 2 price moved strictly, while in case 1 price fluctuated a lot (note the zigzag).

    We consider that in case 1 the efficiency ratio was very low, apprx. 0.33
    While in case 2 the efficiency ratio was the highest possible: 1.

    Values produced by this implementation are in the range [0...1]
    """
    
    def __init__(self,
                 period: int = 10,
                 source: str = 'close',
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        assert isinstance(period, int) and (period > 0)
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, str)
        self.source = source
        name = f'efficiency_ratio_{self.period}_{self.source}'
        self.names = [name]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(0, 1, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        result = EfficiencyRatio.calculate(frames[-self.period:], self.source)
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result
    
    @staticmethod
    def calculate(frames: Sequence[Frame], source: str):
        prev_value = getattr(frames[0], source)
        value = getattr(frames[-1], source)
        change = value - prev_value
        volatility = 0.0
        for frame in frames[1:]:
            value = getattr(frame, source)
            volatility += abs(value - prev_value)
            prev_value = value
        return (change / volatility) if (volatility != 0) else 0.0
    
    def __repr__(self):
        return f'{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})'
