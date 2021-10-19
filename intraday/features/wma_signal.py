from typing import Sequence, Literal, Optional
from collections import OrderedDict, deque
import numpy as np

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class WMASignal(Feature):
    """
    Computes predicted direction signal based on Weighted Moving Average
    
    Parameters
    ----------
    period : int
        Number of frames to take into account when updating weighted moving average. Default: 7
    source : str
        Name of Frame's price attribute. Default: 'hlc3'
    write_to : str {'frame','state','both'}
        Destination of where to put computed values
    """
    
    def __init__(self,
                 period: int = 7,
                 source: str = 'hlc3',
                 write_to: Literal['frame', 'state', 'both'] = 'both'):
        assert isinstance(period, int) and (period > 0)
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, str)
        self.source = source
        self.weights: Optional[np.ndarray] = None
        self.weights2: Optional[np.ndarray] = None
        self.prices = deque(maxlen=period)
        self.wma_prices = deque(maxlen=period)
        self.predicted_prices = deque(maxlen=4)
        self.names = [
            f'wma_signal_{source}'
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-np.inf, np.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def reset(self):
        self.weights = None
        self.prices.clear()
        self.wma_prices.clear()
        self.predicted_prices.clear()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        price = getattr(frame, self.source)
        self.prices.append(price)
        if len(self.prices) < 2:
            self.wma_prices.append(price)
            self.predicted_prices.append(price)
            wma_signal = 0.0
        else:
            if (self.weights is None) or (len(self.weights) != len(self.prices)):
                self.weights = np.arange(1, len(self.prices) + 1, dtype=np.float32)
            wma_price = np.sum(np.multiply(self.weights, self.prices)) / self.weights.sum()
            self.wma_prices.append(wma_price)
            wma_wma_price = np.sum(np.multiply(self.weights, self.wma_prices)) / self.weights.sum()
            predicted_price = 2 * wma_price - wma_wma_price
            self.predicted_prices.append(predicted_price)
            if (self.weights2 is None) or (len(self.weights2) != len(self.predicted_prices)):
                self.weights2 = np.arange(1, len(self.predicted_prices) + 1, dtype=np.float32)
            wma_predicted_price = np.sum(np.multiply(self.weights2, self.predicted_prices)) / self.weights2.sum()
            wma_signal = (predicted_price - wma_predicted_price)

        # Write value
        if self.write_to_frame:
            setattr(frame, self.names[0], wma_signal)
        if self.write_to_state:
            state[self.names[0]] = wma_signal
    
    def __repr__(self):
        return f'{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})'
