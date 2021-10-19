from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
from numbers import Real
import math
import warnings

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class DivDelta(Feature):
    """Computes difference of two values, divided by a third value: r = (a - b) / c"""

    def __init__(self,
                 source: Union[Tuple[str, str, Union[str, Real]], Sequence[Tuple[str, str, Union[str, Real]]]],
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `DivDelta` feature processor

        Parameters
        ----------
        source : Union[Tuple[str, str, Union[str, Real]], Sequence[Tuple[str, str, Union[str, Real]]]]
            Sequence of tuples of three strings of attribute names which will be used to get compute deltas.
            First string is the name of value1 attribute.
            Second string is the name of value2 attribute.
            Third element can be either a string with the attribute name, or the actual real value3.
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
        for (name1, name2, name3) in self.source:
            assert isinstance(name1, str) and isinstance(name2, str) and isinstance(name3, (str, Real))
            self.names.append(f'divdelta_{name1}_{name2}_{str(name3).replace(".", "_")}')
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, (name1, name2, name3) in enumerate(self.source):
            # Read values
            value1 = getattr(last_frame, name1)
            value2 = getattr(last_frame, name2)
            value3 = getattr(last_frame, name3) if isinstance(name3, str) else name3
            # Compute result
            if ((not isinstance(value1, Real)) or (not isinstance(value2, Real))
               or (not isinstance(value3, Real)) or (value3 == 0)):
                warnings.warn(
                    f'{self.__class__.__name__}: Invalid value: {name1}={value1}-{name2}={value2}/{name3}={value3}'
                )
                result = 0.0
            else:
                result = (value1 - value2) / value3
            # Write out result
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result

    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
