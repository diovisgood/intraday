from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class ParabolicSAR(Feature):
    """
    Parabolic SAR for both upward and downward trends

    https://www.daytrading.com/parabolic-sar
    """
    
    def __init__(self,
                 acceleration: float = 0.02,
                 max_velocity: float = 0.2,
                 source: Tuple[str, str] = ('low', 'high'),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initialization of `ParabolicSAR` feature
        Parameters
        ----------
        acceleration : float
            A value which is added to velocity on each step during following the trend.
            Default: 0.02
        max_velocity : float
            A maximum value for velocity. It won't increase more than this value.
            Default: 0.2
        source : Tuple[str, str]
            Tuple of two strings of attribute names which will be used to get price values from each frame.
            Default: (low', 'high')
        write_to : str
            String of where to put a result {'state', 'frame', 'both'}
        """
        super().__init__(write_to=write_to)
        assert isinstance(acceleration, float) and (0 < acceleration < 1)
        assert isinstance(max_velocity, float) and (0 < max_velocity < 1)
        assert acceleration < max_velocity
        self.acceleration = acceleration
        self.max_velocity = max_velocity
        assert isinstance(source, Tuple) and (len(source) == 2)
        self.source = source
        a = f'{acceleration}'[2:]  # .replace('.', '_')
        v = f'{max_velocity}'[2:]  # .replace('.', '_')
        self.names = [
            f'psar_u_value_{a}_{v}_{source[0]}_{source[1]}',
            f'psar_u_reset_{a}_{v}_{source[0]}_{source[1]}',
            f'psar_d_value_{a}_{v}_{source[0]}_{source[1]}',
            f'psar_d_reset_{a}_{v}_{source[0]}_{source[1]}',
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        # Initialize state variables
        self.u_velocity = 0
        self.d_velocity = 0
        self.lowest = None
        self.highest = None
        self.u_sar = None
        self.d_sar = None
    
    def reset(self):
        self.u_velocity = 0
        self.d_velocity = 0
        self.lowest = None
        self.highest = None
        self.u_sar = None
        self.d_sar = None
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        # Load last candle high and low
        last_frame = frames[-1]
        low = getattr(last_frame, self.source[0])
        high = getattr(last_frame, self.source[1])
        
        # For upward trend:
        # Check for new extreme point and update upward velocity
        if (self.u_velocity == 0) or (self.highest is None) or (self.highest < high):
            self.highest = high
            self.u_velocity = min(self.max_velocity, self.u_velocity + self.acceleration)
        
        # Calculate new uptrend SAR value
        prior = self.u_sar if (self.u_sar is not None) else low
        self.u_sar = prior + self.u_velocity * (self.highest - prior)
        
        # Check for uptrend reset
        u_reset = 0
        if self.u_sar >= low:
            self.u_velocity = 0
            self.u_sar = low
            self.highest = high
            u_reset = 1
        
        # Check if SAR is above two prior candle's low
        if len(frames) > 1:
            low1 = getattr(frames[-2], self.source[0])
            if self.u_sar > low1:
                self.u_sar = low1
        if len(frames) > 2:
            low2 = getattr(frames[-3], self.source[0])
            if self.u_sar > low2:
                self.u_sar = low2
        
        # For downwards trend:
        # Check for new extreme point and update downward velocity
        if (self.d_velocity == 0) or (self.lowest is None) or (self.lowest > low):
            self.lowest = low
            self.d_velocity = min(self.max_velocity, self.d_velocity + self.acceleration)
        
        # Calculate new down trend SAR value
        prior = self.d_sar if (self.d_sar is not None) else high
        self.d_sar = prior - self.d_velocity * (prior - self.lowest)
        
        # Check for down trend reset
        d_reset = 0
        if self.d_sar <= high:
            self.d_velocity = 0
            self.d_sar = high
            self.lowest = low
            d_reset = 1
        
        # Check if SAR is below two prior candle's high
        if len(frames) > 1:
            high1 = getattr(frames[-2], self.source[1])
            if self.d_sar < high1:
                self.d_sar = high1
        if len(frames) > 2:
            high2 = getattr(frames[-3], self.source[1])
            if self.d_sar < high2:
                self.d_sar = high2
        
        # Write out values
        if self.write_to_frame:
            setattr(last_frame, self.names[0], self.u_sar)
            setattr(last_frame, self.names[1], u_reset)
            setattr(last_frame, self.names[2], self.d_sar)
            setattr(last_frame, self.names[3], d_reset)
        if self.write_to_state:
            state[self.names[0]] = self.u_sar
            state[self.names[1]] = u_reset
            state[self.names[2]] = self.d_sar
            state[self.names[3]] = d_reset
