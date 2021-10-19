from typing import Sequence, Union, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class Abs(Feature):
    """
    Gets absolute values of source attributes.

    Parameters
    ----------
    source : str or Sequence[str]
        Names of Frame's attributes which absolute values will be the output.
    write_to : {'frame', 'state', 'both'}
        destination of where to put computed values
    """
    def __init__(self,
                 source: Union[str, Sequence[str]],
                 write_to: Literal['state', 'frame', 'both'] = 'state'):
        super().__init__(write_to=write_to)
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for name in self.source:
            field = f'abs_{name}'
            self.names.append(field)
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(0, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            value = abs(getattr(last_frame, name))
            if self.write_to_frame:
                setattr(last_frame, self.names[i], value)
            if self.write_to_state:
                state[self.names[i]] = value

    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
