from typing import Sequence, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.processor import Trade
from intraday.feature import TradesFeature


class AbnormalTrades(TradesFeature):
    """
    Detects abnormal trades which differ from average values in a stream of trades.

    Notes
    -----
    In particular, following values are computed:

    - 'abnormal_{trades_period}_buy_trades'
        Count of buy trades with large amount.
    - 'abnormal_{trades_period}_sell_trades'
        Count of sell trades with large amount.

    Average trade amount is calculated using Exponential Moving Average algorithm.

    Parameters
    ----------
    trades_period : int
        period to compute EMA values
    threshold_factor : float
        how much trade amount should exceed average value to be considered as abnormal
    write_to : {'frame', 'state', 'both'}
        destination of where to put computed values
    """
    
    def __init__(self,
                 trades_period: int = 2000,
                 threshold_factor: float = 5,
                 write_to: Literal['state', 'frame', 'both'] = 'state'):
        assert isinstance(trades_period, int) and (trades_period > 0)
        super().__init__(trades_period=trades_period, write_to=write_to)
        self.threshold_factor = threshold_factor
        self._ema_factor = 2 / (self.trades_period + 1)
        self._n_iter = 0
        self.ema_trade_amount = None
        self.abnormal_buy_trades = 0
        self.abnormal_sell_trades = 0
        self.names = [
            f'abnormal_{trades_period}_buy_trades',
            f'abnormal_{trades_period}_sell_trades',
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(0, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def reset(self):
        self._ema_factor = 2 / (self.trades_period + 1)
        self._n_iter = 0
        self.ema_trade_amount = None
        self.abnormal_buy_trades = 0
        self.abnormal_sell_trades = 0
    
    def update(self, trades: Sequence[Trade]):
        trade = trades[-1]
        # Update average values of a trade
        self._update_average_value('ema_trade_amount', trade.amount)
        if (self._n_iter >= self.trades_period) and (trade.amount >= self.threshold_factor * self.ema_trade_amount):
            if trade.operation == 'B':
                self.abnormal_buy_trades += 1
            else:
                self.abnormal_sell_trades += 1
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if self.write_to_frame:
            frame = frames[-1]
            setattr(frame, self.names[0], self.abnormal_buy_trades)
            setattr(frame, self.names[1], self.abnormal_sell_trades)
        if self.write_to_state:
            state[self.names[0]] = self.abnormal_buy_trades
            state[self.names[1]] = self.abnormal_sell_trades
        self.abnormal_buy_trades = 0
        self.abnormal_sell_trades = 0
    
    def _update_average_value(self, name: str, new_value):
        if new_value is None:
            return
        old_average = getattr(self, name)
        if old_average is None:
            new_average = new_value
        elif self._n_iter <= self.trades_period:
            new_average = float(old_average * self._n_iter + new_value) / float(self._n_iter + 1)
        else:
            new_average = (old_average * (1 - self._ema_factor) + new_value * self._ema_factor)
        setattr(self, name, new_average)
        return new_average
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(trades_period={self.trades_period}, '
            f'threshold_factor={self.threshold_factor}, write_to={self.write_to})'
        )
