from typing import Sequence, Union
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class Copy(Feature):
    """
    Copies some or all attributes from frame to state.

    Parameters
    ----------
    source : str or Sequence[str] or None
        Names of Frame's attributes to be copied into state object.
        If None - all attributes are copied into state.
    """
    
    def __init__(self, source: Union[None, str, Sequence[str]] = None):
        super().__init__(write_to='state')
        if source is None:
            frame = Frame()
            self.source = list(frame.__dict__.keys())
        elif isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        self.spaces = {}
        for name in self.source:
            self.names.append(name)
        self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if self.write_to_state:
            last_frame = frames[-1]
            for name in self.source:
                state[name] = getattr(last_frame, name)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
