from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class ADL(Feature):
    """
    Accumulation Distribution Line (ADL)

    Notes
    -----
    As described here: https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line

    The values of ADL can be big in both positive and negative.
    The absolute value of ADL is not important.
    One should care about trend in divergence of ADL line and the price chart.
    """
    
    def __init__(self,
                 source: Tuple[str, str, str, str] = ('low', 'high', 'close', 'volume'),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `AccumulationDistributionLine` feature processor

        Parameters
        ----------
        source : Tuple[str, str, str, str]
            Names for `low`, `high`, `close` and `volume` of frame's attributes.
        write_to : str {'frame','state','both'}
            Destination of where to put computed values.
        """
        super().__init__(write_to=write_to, period=1)
        assert isinstance(source, Tuple) and (len(source) == 4)
        self.source = source
        self.names = [f'adl_{self.source[0]}_{self.source[1]}_{self.source[2]}_{self.source[3]}']
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        self.value = 0.0
    
    def reset(self):
        self.value = 0.0
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        low, high = getattr(frame, self.source[0]), getattr(frame, self.source[1])
        close, volume = getattr(frame, self.source[2]), getattr(frame, self.source[3])
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low) if (high != low) else 0.0
        self.value = self.value + volume * money_flow_multiplier
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], self.value)
        if self.write_to_state:
            state[self.names[0]] = self.value
        return self.value
