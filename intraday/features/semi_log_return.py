from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
from numbers import Real
import math
import warnings

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class SemiLogReturn(Feature):
    """
    This is both the Return and LogReturn:

    Notes
    -----
    Given two values a and b, returns:
        1. k*(a/b - s)^2 +b  | if (a/b) < 1
        2. ln(a/b)           | if (a/b) >= 1

    When value of a/b is close to zero, result goes to -2
    When value of a/b is very big, result is comparably small.

    It is important that both a and b should be positive numbers.
    """
    X = 1.0
    Y = -2.0
    s = X * (X + 2 * Y) / (2 * (X + Y))
    b = (s - X) / 2
    k = -1 / (4 * b)
    
    def __init__(self,
                 source: Union[Tuple[str, str], Sequence[Tuple[str, str]]] = (('close', 'open'),),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `SemiLogReturn` feature processor

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
            self.names.append(f'semilogreturn_{name1}_{name2}')
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
            if (not isinstance(value1, Real)) or (value1 < 0) or (not isinstance(value2, Real)) or (value2 <= 0):
                warnings.warn(f'{self.__class__.__name__}: Invalid values: {name1}={value1} / {name2}={value2}')
                result = 0.0
            elif value1 / value2 < 1:
                result = self.k * (value1 / value2 - self.s) ** 2 + self.b
            else:
                result = math.log(value1 / value2)
            # Write result
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result
    
    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
