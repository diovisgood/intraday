from typing import Sequence, Union, Literal
from collections import OrderedDict

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class TimeEncoder(Feature):
    """
    Encodes timestamp into format a model can understand

    Notes
    -----

    Machine Learning models can't efficiently utilize raw timestamp values.

    Given a timestamp they can hardly say:

    - is it summer or winter?
    - is it monday or friday?
    - is it morning or noon?

    We can convert timestamp into some floating numbers to make it easier for models to use it:

    - day of year offset: [0...1], where 0 - January 1st and 1- December 31
    - day of week offset: [0...1], where 0 - Monday, 1 - Sunday
    - time of day offset: [0...1], where 0 - 00:00:00, 1 - 23:59:59

    Parameters
    ----------
    source : str or Sequence[str]
        Names of Frame's attributes for which a change over N frames will be the output.
    yday : bool
        If False: sets day of year offset to zero. Default: True
    wday : bool
        If False: sets day of week offset to zero. Default: True
    write_to : str {'frame','state','both'}
        Destination of where to put computed values.
    """
    
    def __init__(self,
                 source: Union[str, Sequence[str]] = 'time_start',
                 yday=True,
                 wday=True,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        super().__init__(write_to=write_to)
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        self.yday = yday
        self.wday = wday
        for name in self.source:
            assert isinstance(name, str)
            self.names.append(f'yday_{name}')
            self.names.append(f'wday_{name}')
            self.names.append(f'time_{name}')
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(0, 1, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            datetime = getattr(last_frame, name)
            # Calculate day of year offset
            # year_start = datetime.floor('year') # .replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            # year_end = datetime.ceil('year') # .replace(month=12, day=31, hour=23, minute=59, second=59,
            # microsecond=999999)
            # day_start = datetime.floor('day') # .replace(hour=0, minute=0, second=0, microsecond=0)
            year_start = datetime.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            year_end = datetime.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
            day_start = datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            if self.yday:
                yday = (day_start - year_start) / (year_end - year_start)
            else:
                yday = 0.0
            # Calculate day of week offset
            if self.wday:
                wday = datetime.isoweekday() / 7
            else:
                wday = 0.0
            # Calculate time of day offset
            time = (datetime.timestamp() - day_start.timestamp()) / float(24 * 60 * 60)
            # Write out values
            if self.write_to_frame:
                setattr(last_frame, self.names[i * 3 + 0], yday)
                setattr(last_frame, self.names[i * 3 + 1], wday)
                setattr(last_frame, self.names[i * 3 + 2], time)
            if self.write_to_state:
                state[self.names[i * 3 + 0]] = yday
                state[self.names[i * 3 + 1]] = wday
                state[self.names[i * 3 + 2]] = time
    
    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
