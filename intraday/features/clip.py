from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
from numbers import Real
import math
import warnings

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class Clip(Feature):
    """Clips values into [min. max] diapason"""

    def __init__(self,
                 source: Union[Tuple[str, Real, Real], Sequence[Tuple[str, Real, Real]]],
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `Clip` feature processor

        Parameters
        ----------
        source : Union[Tuple[str, Real, Real], Sequence[Tuple[str, Real, Real]]]
            Sequence of tuples of three values:
            First string is the name of the attribute.
            Second is the Real min value.
            Third is the Real max value.
        write_to : str {'state', 'frame', 'both'}
            string of where to put a result
        """
        super().__init__(write_to=write_to)
        if isinstance(source, Tuple) and (len(source) == 3):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for (name, minval, maxval) in self.source:
            assert isinstance(name, str)
            assert isinstance(minval, Real) and isinstance(maxval, Real) and (minval < maxval)
            self.names.append(f'clip_{name}')
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, (name, minval, maxval) in enumerate(self.source):
            # Read values
            value = getattr(last_frame, name)
            # Compute result
            if not isinstance(value, Real):
                warnings.warn(f'{self.__class__.__name__}: Invalid value: {name}={value}')
                result = 0.0
            else:
                result = minval if (value <= minval) else maxval if (value >= maxval) else value
            # Write out result
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result

    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
