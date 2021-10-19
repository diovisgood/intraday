from typing import Sequence, Union, Literal
from collections import OrderedDict, namedtuple
from numbers import Real
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


Event = namedtuple('Event', 'frame sign level', defaults=(None, 0, None))


class CumulativeSum(Feature):
    """
    Cumulative Sum Filter detects positive and negative breakouts in price chart

    https://en.wikipedia.org/wiki/CUSUM
    """
    
    def __init__(self,
                 threshold: Union[Real, str],
                 threshold_factor: Real = 1,
                 source: str = 'close',
                 write_to: Literal['frame', 'state', 'both'] = 'state',
                 limit: int = 1000):
        super().__init__(write_to=write_to)
        
        assert isinstance(source, str)
        self.source = source
        
        assert isinstance(threshold, str) or (isinstance(threshold, Real) and (threshold > 0))
        self.threshold = threshold
        
        assert isinstance(threshold_factor, Real) and (threshold_factor > 0)
        self.threshold_factor = threshold_factor
        
        assert isinstance(limit, int) and (limit > 0)
        self.limit = limit
        
        self.names.append(
            f'cusum_{source}_threshold_'
            f'{("%.3g" % threshold_factor).replace(".", "_")}_'
            f'{("%.3g" % threshold).replace(".", "_") if isinstance(threshold, Real) else threshold}'
        )
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        self.pos = 0.0
        self.neg = 0.0
        self.events = []
    
    def reset(self):
        self.pos = 0.0
        self.neg = 0.0
        self.events.clear()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        prev_frame = frames[-2] if (len(frames) > 1) else None
        # Get expected value from previous frame
        expected = getattr(prev_frame, self.source) if (prev_frame is not None) else None
        # Read source value and prediction for next value
        value = getattr(last_frame, self.source)
        # Update cumulative counters
        self.pos = max(0, self.pos + value - expected) if (expected is not None) else 0
        self.neg = min(0, self.neg + value - expected) if (expected is not None) else 0
        # Check for threshold breakthrough
        sign = 0
        if isinstance(self.threshold, Real):
            threshold = self.threshold * self.threshold_factor
        else:
            threshold = getattr(last_frame, self.threshold) * self.threshold_factor
        if self.pos >= threshold:
            self.events.append(Event(frame=last_frame, sign=1, level=value))
            self.pos = 0
            sign = 1
        if self.neg <= -threshold:
            self.events.append(Event(frame=last_frame, sign=-1, level=value))
            self.neg = 0
            sign = -1
        # Drop oldest event if list is too long
        if len(self.events) > self.limit:
            del self.events[0]
        # Write result
        if self.write_to_frame:
            setattr(last_frame, self.names[0], sign)
        if self.write_to_state:
            state[self.names[0]] = sign
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold}, source={self.source})'
