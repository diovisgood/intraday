from typing import Sequence, Union, Literal
from collections import OrderedDict
from numbers import Real
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class Change(Feature):
    """Computes change of some values over last N periods: change_3_close = (close[-1] - close[-3])"""
    
    def __init__(self, period: int, source: Union[str, Sequence[str]] = ('close', 'vwap'),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `Change` feature processor

        Parameters
        ----------
        period : int
            Number of frames to compute change over.
        source : str or Sequence[str]
            Names of Frame's attributes for which a change over N frames will be computed.
        write_to : str {'frame','state','both'}
            Destination of where to put computed values.
        """
        assert isinstance(period, int) and (period > 0)
        super().__init__(write_to=write_to, period=period)
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for name in self.source:
            assert isinstance(name, str)
            self.names.append(f'change_{period}_{name}')
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        prev_frame = frames[-min(self.period, len(frames))]
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            # Read values
            prev_value = getattr(prev_frame, name)
            last_value = getattr(last_frame, name)
            # Compute difference
            if (not isinstance(prev_value, Real)) or (not isinstance(last_value, Real)):
                result = 0.0
            else:
                result = (last_value - prev_value)
            # Write out result
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result
    
    def __repr__(self):
        return f'{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})'
