from typing import Sequence, Literal
from collections import OrderedDict
import math

import gym.spaces

from intraday.frame import Frame
from intraday.processor import Trade
from intraday.feature import TradesFeature


class AverageTrade(TradesFeature):
    """
    Computes average values, which describe a stream of trades.

    Notes
    -----
    In particular, following values are computed:

    - ema_{trades_period}_trade_tick
        Average trade direction: a value in between [-1, +1],
        where +1 = all trades were initiated by market Buy,
        -1 = all trades were initiated by market Sell order.
    - 'ema_{trades_period}_trade_spread'
        Average trade spread
    - 'ema_{trades_period}_trade_amount'
        Average trade amount
    - 'ema_{trades_period}_buy_amount'
        Average buy trade amount
    - 'ema_{trades_period}_sell_amount'
        Average sell trade amount

    Average values are calculated using Exponential Moving Average algorithm.
    When `process_trade` method is called, it outputs average values for the current moment.

    Parameters
    ----------
    trades_period : int
        period to compute EMA values
    write_to : {'frame', 'state', 'both'}
        destination of where to put computed values
    """
    
    def __init__(self,
                 trades_period: int = 2000,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        assert isinstance(trades_period, int) and (trades_period > 0)
        super().__init__(trades_period=trades_period, write_to=write_to)
        self._ema_factor = 2 / (self.trades_period + 1)
        self._n_iter = 0
        self.ema_trade_tick = None
        self.ema_trade_spread = None
        self.ema_trade_amount = None
        self.ema_buy_amount = None
        self.ema_buy_money = None
        self.ema_sell_amount = None
        self.ema_sell_money = None
        self.names = [
            f'ema_{trades_period}_trade_tick',
            f'ema_{trades_period}_trade_spread',
            f'ema_{trades_period}_trade_amount',
            f'ema_{trades_period}_buy_amount',
            f'ema_{trades_period}_sell_amount',
        ]
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(0, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def reset(self):
        self._ema_factor = 2 / (self.trades_period + 1)
        self._n_iter = 0
        self.ema_trade_tick = None
        self.ema_trade_spread = None
        self.ema_trade_amount = None
        self.ema_buy_amount = None
        self.ema_sell_amount = None
    
    def update(self, trades: Sequence[Trade]):
        trade = trades[-1]
        prev_trade = trades[-2] if (len(trades) > 1) else None
        # Update average values of a trade
        tick = (1 if (trade.operation == 'B') else -1)
        amount = trade.amount
        self._update_average_value('ema_trade_tick', tick)
        self._update_average_value('ema_trade_amount', amount)
        if trade.operation == 'B':
            self._update_average_value('ema_buy_amount', amount)
        else:
            self._update_average_value('ema_sell_amount', amount)
        # Update spread
        if (prev_trade is not None) and (prev_trade.operation != trade.operation):
            spread = abs(prev_trade.price - trade.price)
            self._update_average_value('ema_trade_spread', spread)
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if self.write_to_frame:
            frame = frames[-1]
            setattr(frame, self.names[0], self.ema_trade_tick)
            setattr(frame, self.names[1], self.ema_trade_spread)
            setattr(frame, self.names[2], self.ema_trade_amount)
            setattr(frame, self.names[3], self.ema_buy_amount)
            setattr(frame, self.names[4], self.ema_sell_amount)
        if self.write_to_state:
            state[self.names[0]] = self.ema_trade_tick
            state[self.names[1]] = self.ema_trade_spread
            state[self.names[2]] = self.ema_trade_amount
            state[self.names[3]] = self.ema_buy_amount
            state[self.names[4]] = self.ema_sell_amount
    
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
        return f'{self.__class__.__name__}(trades_period={self.trades_period}, write_to={self.write_to})'
