from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
from numbers import Real
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class EOM(Feature):
    """
    Ease of Movement (EOM)

    Notes
    -----
    As described here: https://www.daytrading.com/ease-of-movement

    Values of EOM can be larger than volume, but also positive and negative.
    Typically you may use a volume_factor to reduce them somehow.
    But absolute values of EOM do not matter.
    Some traders use a break above or below the zero line in order to enter into a trade.
    Above zero readings are bullish and below zero readings are bearish.
    """
    
    def __init__(self,
                 source: Tuple[str, str, str] = ('low', 'high', 'volume'),
                 price_factor: Union[str, Real, None] = None,
                 volume_factor: Union[str, Real, None] = None,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `EaseOfMovement` feature processor

        Parameters
        ----------
        source : Tuple[str, str, str]
            Names for `low`, `high` and `volume` of frame's attributes.
        price_factor : Union[str, Real, None]
            A value to divide price range (`high` - `low`) by, to normalize it.
            Or a string name of frame's attribute to get this value from.
            Or specify None to avoid price range normalization.
        volume_factor : Union[str, Real, None]
            A value to divide `volume` by, to normalize it.
            Or a string name of frame's attribute to get this value from.
            Or specify None to avoid `volume` normalization.
        write_to : str {'frame','state','both'}
            Destination of where to put computed values.
        """
        super().__init__(write_to=write_to, period=2)
        assert isinstance(source, Tuple) and (len(source) == 3)
        self.source = source
        assert (volume_factor is None) or isinstance(volume_factor, (str, Real))
        self.price_factor = price_factor
        assert (volume_factor is None) or isinstance(volume_factor, (str, Real))
        self.volume_factor = volume_factor
        self.names = [f'eom_{self.source[0]}_{self.source[1]}_{self.source[2]}']
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if len(frames) > 1:
            frame, prev_frame = frames[-1], frames[-2]
            low, high, = getattr(frame, self.source[0]), getattr(frame, self.source[1])
            prev_low, prev_high = getattr(prev_frame, self.source[0]), getattr(prev_frame, self.source[1])
            # Get price factor
            if isinstance(self.price_factor, Real):
                price_factor = self.price_factor
            elif isinstance(self.price_factor, str):
                price_factor = getattr(frame, self.price_factor)
            else:
                price_factor = 1
            # Get volume and volume factor
            volume = getattr(frame, self.source[2])
            if isinstance(self.volume_factor, Real):
                volume_factor = self.volume_factor
            elif isinstance(self.volume_factor, str):
                volume_factor = getattr(frame, self.volume_factor)
            else:
                volume_factor = 1
            # Compute result
            distance = (high + low - prev_high - prev_low) / 2
            result = (
                (distance * (high - low) * volume_factor) / (volume * price_factor * price_factor)
                if (volume != 0) and (price_factor != 0)
                else 0.0
            )
        else:
            result = 0.0
        # Write out result
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result
