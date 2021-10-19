from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class ZEMA(Feature):
    """
    Zero lag EMA, based on the book "Rocket Science for Traders" by John F. Ehlers

    Parameters
    ----------
    period : int
        The period to calculate momentum: price[0] - price[period].
    alpha : float
        The value in range [0, 1] which has the same meaning as for EMA.
        The closer the alpha to 1.0 - the faster the ZEMA is.
        Default: 0.25
    K : float
        The value in range [0, 1] which defines the velocity coefficient
    source : str
        Name of attribute which will be used to get price zema_value from each frame.
    write_to : str
        String of where to put a result {'state', 'frame', 'both'}
    """
    
    def __init__(self,
                 period: int = 3,
                 alpha: float = 0.25,
                 K: float = 0.5,
                 source: str = 'hlc3',
                 write_to: Literal['frame', 'state', 'both'] = 'both'):
        assert isinstance(period, int) and (period > 0)
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, str)
        self.source = source
        self.alpha = alpha
        self.K = K
        self.names = [
            f'zema_{self.period}_{self.source}'
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        self.zema_value = None
    
    def reset(self):
        self.zema_value = None
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        # Read latest kama_value
        frame = frames[-1]
        frame1 = frames[max(0, len(frames) - self.period)]
        value = getattr(frame, self.source)
        value1 = getattr(frame1, self.source)

        if self.zema_value is None:
            self.zema_value = value
        else:
            new_value = value + self.K * (value - value1)
            self.zema_value = self.alpha * new_value + (1 - self.alpha) * self.zema_value
        
        if self.write_to_frame:
            setattr(frame, self.names[0], self.zema_value)
        if self.write_to_state:
            state[self.names[0]] = self.zema_value
        
        return self.zema_value
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'period={self.period}, alpha={self.alpha}, K={self.K}, '
            f'source={self.source}, write_to={self.write_to})'
        )
