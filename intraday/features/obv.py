from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class OBV(Feature):
    """
    On-Balance Volume (OBV)

    Notes
    -----
    As described here: https://www.investopedia.com/terms/o/onbalancevolume.asp

    Values of OBV may take amplitude similar to volume values, but they can be both positive and negative.
    The absolute value of OBV is not important.
    One should care about trend in OBV and also divergence of OBV line and the price chart.
    """
    
    def __init__(self,
                 source: Tuple[str, str] = ('close', 'volume'),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `OnBalanceVolume` feature processor

        Parameters
        ----------
        source : Tuple[str, str]
            Names for `close` and `volume` of frame's attributes.
        write_to : str {'frame','state','both'}
            Destination of where to put computed values.
        """
        super().__init__(write_to=write_to, period=2)
        assert isinstance(source, Tuple) and (len(source) == 2)
        self.source = source
        self.names = [f'obv_{self.source[0]}_{self.source[1]}']
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        self.value = 0.0
    
    def reset(self):
        self.value = 0.0
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        close, volume = getattr(frame, self.source[0]), getattr(frame, self.source[1])
        prev_close = getattr(frames[-2], self.source[0]) if (len(frames) > 1) else close
        if close > prev_close:
            self.value += volume
        elif close < prev_close:
            self.value -= volume
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], self.value)
        if self.write_to_state:
            state[self.names[0]] = self.value
        return self.value
