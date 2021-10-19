from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class VI(Feature):
    """
    Positive and Negative Volume Indexes (PVI, NVI)

    Notes
    -----
    As described here:
    PVI - https://www.investopedia.com/terms/p/pvi.asp
    NVI - https://www.investopedia.com/terms/n/nvi.asp

    Values of volume index can be around 100 and may also be negative.
    Generally, traders will watch both PVI and NVI indicators
    to get a sense of the market’s trend in terms of volume.
    PVI will be more volatile when volume is rising
    and NVI will be more volatile when volume is decreasing.

    Traders watch for relative position of PVI/NVI and their
    relative moving average.
    When PVI crosses above EMA(PVI) - it is a signal to buy.
    When NVI crosses below EMA(NVI) - is is a signall to sell.
    """
    
    def __init__(self,
                 source: Tuple[str, str] = ('close', 'volume'),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `VolumeIndex` feature processor

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
        self.names = [
            f'pvi_{self.source[0]}_{self.source[1]}',
            f'nvi_{self.source[0]}_{self.source[1]}',
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        self.pvi = 100.0
        self.nvi = 100.0
    
    def reset(self):
        self.pvi = 100.0
        self.nvi = 100.0
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        close, volume = getattr(frame, self.source[0]), getattr(frame, self.source[1])
        prev_close = getattr(frames[-2], self.source[0]) if (len(frames) > 1) else close
        prev_volume = getattr(frames[-2], self.source[1]) if (len(frames) > 1) else volume
        change = (close - prev_close) / prev_close
        if volume > prev_volume:
            self.pvi *= (1 + change)
        elif volume < prev_volume:
            self.nvi *= (1 + change)
        if self.write_to_frame:
            setattr(frame, self.names[0], self.pvi)
            setattr(frame, self.names[1], self.nvi)
        if self.write_to_state:
            state[self.names[0]] = self.pvi
            state[self.names[1]] = self.nvi
        return self.pvi, self.nvi
