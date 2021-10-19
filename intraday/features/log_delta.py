from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
from numbers import Real
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class LogDelta(Feature):
    """Computes signed log of difference of two values: r = sign(a - b) * log(|a - b|)"""
    
    def __init__(self,
                 source: Union[Tuple[str, str], Sequence[Tuple[str, str]]] = (('close', 'open'),),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `LogDelta` feature processor

        Parameters
        ----------
        source : Sequence[Tuple[str, str]]
            Sequence of tuples of two strings of attribute names which will be used to get compute deltas.
            Default: (('close', 'open'),)
        write_to : str {'state', 'frame', 'both'}
            string of where to put a result
        """
        super().__init__(write_to=write_to)
        if isinstance(source, Tuple) and (len(source) == 2) and all(isinstance(v, str) for v in source):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for (name1, name2) in self.source:
            assert isinstance(name1, str) and isinstance(name2, str)
            self.names.append(f'logdelta_{name1}_{name2}')
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, (name1, name2) in enumerate(self.source):
            # Read values
            value1 = getattr(last_frame, name1)
            value2 = getattr(last_frame, name2)
            # Compute result
            if (not isinstance(value1, Real)) or (not isinstance(value2, Real)):
                result = 0.0
            else:
                delta = _delta(value1, value2)
                sign = 1 if (delta > 0) else -1 if (delta < 0) else 0
                if abs(delta) < 1:
                    # warnings.warn(f'{self.__class__.__name__}: Invalid value: {name1}={value1} - {name2}={value2}')
                    result = 0.0
                else:
                    result = sign * math.log(abs(delta))
            # Write out result
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result
    
    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'


def _delta(a, b):
    return 0.0 if (a is None) or (b is None) else (a - b)
