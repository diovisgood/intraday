from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature
from intraday.features.efficiency_ratio import EfficiencyRatio


class KAMA(Feature):
    """Kaufman's Adaptive Moving Average"""
    
    def __init__(self,
                 period: Tuple[int, int, int] = (2, 10, 8),
                 source: str = 'close',
                 write_to: Literal['frame', 'state', 'both'] = 'both'):
        """
        Initialization of `KAMA` feature processor

        Parameters
        ----------
        period
            A tuple of three periods (short_ema_period, long_ema_period, kama_period)
            short_ema_period - the shortest period to use at times of highest volatility.
            long_ema_period - the longest period to use at times of lowest volatility.
            kama_period - the period to calculate volatility. Typically you may set it equal to long_ema_period or
             use kama_value in between [short_ema_period, long_ema_period].
        source : str
            Name of attribute which will be used to get price kama_value from each frame.
        write_to : str
            String of where to put a result {'state', 'frame', 'both'}
        """
        assert isinstance(period, Tuple) and (len(period) == 3)
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, str)
        self.source = source
        self.names = [
            f'efficiency_ratio_{period[2]}_{self.source}',
            f'kama_{"_".join(str(x) for x in self.period)}_{self.source}'
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        self.kama_value = None
    
    def reset(self):
        self.kama_value = None
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        # Read latest kama_value
        last_frame = frames[-1]
        value = getattr(last_frame, self.source)
        
        if self.kama_value is None:
            efficiency_ratio = 0.0
            self.kama_value = value
        else:
            fast_period, slow_period, kama_period = self.period
            
            # Calculate efficiency ratio
            efficiency_ratio = EfficiencyRatio.calculate(frames[-kama_period:], self.source)
            
            # Calculate EMA factor
            fast_ema_factor = 2.0 / (fast_period + 1.0)
            slow_ema_factor = 2.0 / (slow_period + 1.0)
            ema_factor = (slow_ema_factor + abs(efficiency_ratio) * (fast_ema_factor - slow_ema_factor)) ** 2
            
            # Update KAMA kama_value
            self.kama_value = self.kama_value + ema_factor * (value - self.kama_value)
        
        if self.write_to_frame:
            setattr(last_frame, self.names[0], efficiency_ratio)
            setattr(last_frame, self.names[1], self.kama_value)
        if self.write_to_state:
            state[self.names[0]] = efficiency_ratio
            state[self.names[1]] = self.kama_value
        
        return self.kama_value
    
    def __repr__(self):
        return f'{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})'
