from typing import Sequence, Union, Literal
from collections import OrderedDict
from numbers import Real
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class Log(Feature):
    """Computes signed log of value: r = sign(v) * log(|v|)"""
    
    def __init__(self,
                 source: Union[str, Sequence[str]],
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `Log` feature processor

        Parameters
        ----------
        source : Union[str, Sequence[str]]
            Sequence of strings of attribute names which will be used to compute signed log.
        write_to : str {'state', 'frame', 'both'}
            string of where to put a result
        """
        super().__init__(write_to=write_to)
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for name in self.source:
            assert isinstance(name, str)
            self.names.append(f'log_{name}')
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            # Read values
            value = getattr(last_frame, name)
            # Compute result
            if (not isinstance(value, Real)) or (abs(value) <= 1):
                result = 0.0
            else:
                sign = 1 if (value > 0) else -1 if (value < 0) else 0
                result = sign * math.log(abs(value))
            # Write out result
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result
    
    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
