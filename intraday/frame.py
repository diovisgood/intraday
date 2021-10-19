from __future__ import annotations
from numbers import Real
from typing import Sequence, Union, Optional
from arrow import Arrow
from datetime import timedelta
from .provider import Trade, TradeOI


class Frame(object):
    """
    Accumulates and updates different attributes which describe a series of trades for some period of time.
    """
    
    def __init__(self, **kwargs):
        self.time_start: Optional[Arrow] = None
        """Date and time of beginning of frame. Type is the same as specified in trades."""

        self.time_end: Optional[Arrow] = None
        """Date and time of the end of frame. Type is the same as specified in trades."""
        
        self.duration: Optional[timedelta] = None
        """
        Duration of frame, typically in seconds (it depends on type of values of datetime of each trade).
        Equals to (time_end - time_start). Type is typically: timedelta or Real
        """
        
        self.prev_close: Optional[Real] = None
        """Price value of a trade just before this frame. Is used internally to calculate true_range"""

        self.open: Optional[Real] = None
        """Highest price of a trade during frame. Type: Real"""

        self.high: Optional[Real] = None
        """Highest price of a trade during frame. Type: Real"""
        
        self.low: Optional[Real] = None
        """Lowest price of a trade during frame. Type: Real"""
        
        self.close: Optional[Real] = None
        """Price of a final trade during frame. Type: Real"""

        self.hl2: Optional[Real] = None
        """Middle price: (High+Low)/2. Type: Real"""

        self.hlc3: Optional[Real] = None
        """Typical price: (High+Low+Close)/3. Type: Real"""

        self.true_range: Optional[Real] = 0
        """The difference of prices during frame (including the last price of previous frame). Type: Real"""

        self.flips: int = 0
        """Number of times stream of trades reverted from buy to sell or vise versa."""

        self._sum_spread: Real = 0
        """Used internally to calculate avg_trade_spread."""

        self.avg_trade_spread: Optional[Real] = None
        """
        There is usually a spread between buy and sell orders in order book.
        But order book contains a set of limit orders, which do not move the price.
        Price is changed when new market orders arrive and are executed using those limit orders.
        So if a buy trade is followed by a sell trade a price typically jumps down for a spread value.
        And when a sell trade is followed by a buy trade a price typically jumps up for a spread value.
        This attribute shows the average spread observed during frame.
        """

        self.trade_spread_min: Optional[Real] = None
        """
        There is usually a spread between buy and sell orders in order book.
        But order book contains a set of limit orders, which do not move the price.
        Price is changed when new market orders arrive and are executed using those limit orders.
        So if a buy trade is followed by a sell trade a price typically jumps down for a spread value.
        And when a sell trade is followed by a buy trade a price typically jumps up for a spread value.
        This attribute shows the minimum spread observed during frame.
        """

        self.trade_spread_max: Optional[Real] = None
        """As for the trade_spread_min, but this attribute shows the maximum spread observed during frame."""

        self.ticks: int = 0
        """Number of trades during frame. Type: int"""

        self.volume: Real = 0
        """Total sum of all amounts of all trades during frame. Type: Real"""
        
        self.money: Real = 0
        """Total sum of (amount*price) for all trades during frame. Type: Real"""

        self.vwap: Optional[Real] = None
        """Volume-weighted average price. Simply saying, the price at which most of volume was traded during frame."""

        self.avg_trade_tick: Optional[float] = None
        """Average direction of a trade during frame. Equals to: (imbalance_ticks / ticks)"""
        
        self.avg_trade_spread: Optional[Real] = None
        """Average difference of price from trade to trade"""

        self.avg_trade_amount: Optional[Real] = None
        """Average amount of a trade during frame. Equals to: (volume/ticks)"""

        self.buy_ticks: int = 0
        """
        Number of trades, initiated by market buy orders.
        Please do not forget, that each trade has a buyer and a seller.
        In this case seller had put limit order long ago, and buyer issued a market order to be executed immediately.
        """
        
        self.buy_volume: Real = 0
        """Total sum of amounts in all buy trades during frame. Type: Real"""

        self.buy_money: Real = 0
        """Total sum of (amount*price) in all buy trades during frame."""

        self.buy_vwap: Optional[Real] = None
        """Volume-weighted average price for all buy trades during frame."""

        self.avg_buy_amount: Optional[Real] = None
        """Average trade amount for all buy trades during frame."""

        self.avg_buy_money: Optional[Real] = None
        """Average trade money (=amount*price) for all buy trades during frame."""

        self.sell_ticks: int = 0
        """
        Number of trades, initiated by market sell orders.
        Please do not forget, that each trade has a buyer and a seller.
        In this case buyer had put limit order long ago, and seller issued a market order to be executed immediately.
        """

        self.sell_volume: Real = 0
        """Total sum of amounts in all buy trades during frame. Type: Real"""

        self.sell_money: Real = 0
        """Total sum of (amount*price) in all sell trades during frame."""
        
        self.sell_vwap: Optional[Real] = None
        """Volume-weighted average price for all sell trades during frame"""

        self.avg_sell_amount: Optional[Real] = None
        """Average trade amount for all sell trades during frame."""

        self.avg_sell_money = None
        """Average trade money (=amount*price) for all sell trades during frame."""

        self.conseq_buy_ticks: int = 0
        """
        Number of consequential buy trades.
        For instance: series of B S B B S will result in conseq_buy_ticks==1.
        """

        self.conseq_buy_volume: Real = 0
        """Total sum of amount of all consequential buy trades."""

        self.conseq_buy_money: Real = 0
        """Total sum of (amount*price) of all consequential buy trades."""

        self.conseq_sell_ticks: Real = 0
        """
        Number of consequential sell trades.
        For instance: series of B S B S S will result in conseq_sell_ticks==1.
        """
        
        self.conseq_sell_volume: Real = 0
        """Total sum of amount of all consequential sell trades."""

        self.conseq_sell_money: Real = 0
        """Total sum of (amount*price) of all consequential sell trades."""

        self.vwap_range: Real = 0
        """Difference between VWAP for buy and sell trades: (buy_vwap - sell_vwap)"""

        self.imbalance_ticks: int = 0
        """Difference between number of buy and sell trades: (buy_ticks - sell_ticks)"""

        self.imbalance_volume: Real = 0
        """Difference between summary volume of buy and sell trades: (buy_volume - sell_volume)"""

        self.imbalance_money: Real = 0
        """Difference between summary money of buy and sell trades: (buy_money - sell_money)"""

        self.imbalance_conseq_ticks: int = 0
        """Difference between number of consequential buy and sell trades: (conseq_buy_ticks - conseq_sell_ticks)"""

        self.imbalance_conseq_volume: Real = 0
        """
        Difference between summary volume of consequential buy and sell trades:
        (conseq_buy_volume - conseq_sell_volume)
        """

        self.imbalance_conseq_money: Real = 0
        """
        Difference between summary money of consequential buy and sell trades:
        (conseq_buy_money - conseq_sell_money)
        """

        self.oi_open: Optional[Real] = None
        self.oi_open: Optional[Real] = None
        """
        Open interest value as it was at the beginning of a frame.
        Available only if provider outputs TradeOI instead of Trade objects.
        As TradeOI has additional field: `open_interest`.
        """

        self.oi_high: Optional[Real] = None
        """
        The highest value of open interest during a frame.
        Available only if provider outputs TradeOI instead of Trade objects.
        As TradeOI has additional field: `open_interest`.
        """

        self.oi_low: Optional[Real] = None
        """
        The lowest value of open interest during a frame.
        Available only if provider outputs TradeOI instead of Trade objects.
        As TradeOI has additional field: `open_interest`.
        """

        self.oi_close: Optional[Real] = None
        """
        Open interest value as it was at the end of a frame.
        Available only if provider outputs TradeOI instead of Trade objects.
        As TradeOI has additional field: `open_interest`.
        """

        # Initialize values from kwargs if any
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ != other.__dict__

    def update(self, trades: Sequence[Union[Trade, TradeOI]]):
        """
        Updates frame using new trade.
        
        Please note that:
        
        - New trade should be the last in trades sequence.
        
        - You may call update only once for each new trade.
        
        Parameters
        ----------
        trades : Sequence[Union[Trade, TradeOI]]
            A sequence of trades or trades with additional open interest information.
            New trade should be the last in sequence.
        """
        # Read latest trades
        trade = trades[-1]
        prev_trade = trades[-2] if len(trades) > 1 else None
        datetime, operation, amount, price = trade.datetime, trade.operation, trade.amount, trade.price
        money = amount * price
        # Update times and duration
        if (self.time_start is None) or (self.time_start > datetime):
            self.time_start = datetime
        # Update time_end if needed
        if (self.time_end is None) or (self.time_end < datetime):
            self.time_end = datetime
        self.duration = (self.time_end - self.time_start).total_seconds()
        # Update OHLC prices for current trades_period
        if (self.prev_close is None) and (prev_trade is not None):
            self.prev_close = prev_trade.price
        if (self.open is None):
            self.open = price
        if (self.high is None) or (self.high < price):
            self.high = price
        if (self.low is None) or (self.low > price):
            self.low = price
        self.close = price
        # Update true range
        # if self.prev_close is not None:
        #     self.true_range = (max(self.high, self.prev_close) - min(self.low, self.prev_close))
        # else:
        #     self.true_range = (self.high - self.low)
        # Update spread
        if (prev_trade is not None) and (prev_trade.operation != trade.operation):
            spread = abs(prev_trade.price - trade.price)
            self._sum_spread += spread
            self.flips += 1
            self.avg_trade_spread = self._sum_spread / self.flips
            if (self.trade_spread_min is None) or (self.trade_spread_min > spread):
                self.trade_spread_min = spread
            if (self.trade_spread_max is None) or (self.trade_spread_max < spread):
                self.trade_spread_max = spread
        # Update open interest if any
        if hasattr(trade, 'open_interest'):
            open_interest = trade.open_interest
            if self.oi_open is None:
                self.oi_open = open_interest
            if (self.oi_high is None) or (self.oi_high < open_interest):
                self.oi_high = open_interest
            if (self.oi_low is None) or (self.oi_low > open_interest):
                self.oi_low = open_interest
            self.oi_close = open_interest
        # Update trades count
        self.ticks += 1
        # Update volume
        self.volume += amount
        # Update money
        self.money += money
        # Update VWAP
        # self.vwap = self.money / self.volume
        # Average amount of a trade
        # self.avg_trade_amount = self.volume / self.ticks
        # Update buy/sell counters
        if operation == 'B':
            self.buy_ticks += 1
            self.buy_volume += amount
            self.buy_money += money
            # self.buy_vwap = self.buy_money / self.buy_volume
            # self.avg_buy_amount = self.buy_volume / self.buy_ticks
            # self.avg_buy_money = self.buy_money / self.buy_ticks
        else:
            self.sell_ticks += 1
            self.sell_volume += amount
            self.sell_money += money
            # self.sell_vwap = self.sell_money / self.sell_volume
            # self.avg_sell_amount = self.sell_volume / self.sell_ticks
            # self.avg_sell_money = self.sell_money / self.sell_ticks
        # Update consequential buys or sells
        if prev_trade is not None:
            if prev_trade.operation == operation:
                # Update consequent counters
                if operation == 'B':
                    self.conseq_buy_ticks += 1
                    self.conseq_buy_volume += amount
                    self.conseq_buy_money += money
                else:
                    self.conseq_sell_ticks += 1
                    self.conseq_sell_volume += amount
                    self.conseq_sell_money += money
        # # Update imbalance values
        # self.vwap_range = _diff(self.sell_vwap, self.buy_vwap)
        # self.imbalance_ticks = (self.buy_ticks - self.sell_ticks)
        # self.imbalance_volume = (self.buy_volume - self.sell_volume)
        # self.imbalance_money = (self.buy_money - self.sell_money)
        # self.imbalance_conseq_ticks = (self.conseq_buy_ticks - self.conseq_sell_ticks)
        # self.imbalance_conseq_volume = (self.conseq_buy_volume - self.conseq_sell_volume)
        # self.imbalance_conseq_money = (self.conseq_buy_money - self.conseq_sell_money)
        # self.avg_trade_tick = (self.imbalance_ticks / self.ticks)

    def finalize(self) -> Frame:
        # Update middle and average prices
        self.hl2 = (self.high + self.low) / 2
        self.hlc3 = (self.high + self.low + self.close) / 3
        # Update true range
        if self.prev_close is not None:
            self.true_range = (max(self.high, self.prev_close) - min(self.low, self.prev_close))
            # delattr(self, 'prev_close')
        else:
            self.true_range = (self.high - self.low)
        # Average amount of a trade
        if self.ticks > 0:
            self.avg_trade_amount = self.volume / self.ticks
        if self.buy_ticks > 0:
            self.avg_buy_amount = self.buy_volume / self.buy_ticks
            self.avg_buy_money = self.buy_money / self.buy_ticks
        if self.sell_ticks > 0:
            self.avg_sell_amount = self.sell_volume / self.sell_ticks
            self.avg_sell_money = self.sell_money / self.sell_ticks
        # Update VWAP
        if self.volume > 0:
            self.vwap = self.money / self.volume
        if self.buy_volume > 0:
            self.buy_vwap = self.buy_money / self.buy_volume
        if self.sell_volume > 0:
            self.sell_vwap = self.sell_money / self.sell_volume
        # Update imbalance values
        self.vwap_range = _diff(self.sell_vwap, self.buy_vwap)
        self.imbalance_ticks = (self.buy_ticks - self.sell_ticks)
        self.imbalance_volume = (self.buy_volume - self.sell_volume)
        self.imbalance_money = (self.buy_money - self.sell_money)
        self.imbalance_conseq_ticks = (self.conseq_buy_ticks - self.conseq_sell_ticks)
        self.imbalance_conseq_volume = (self.conseq_buy_volume - self.conseq_sell_volume)
        self.imbalance_conseq_money = (self.conseq_buy_money - self.conseq_sell_money)
        if self.ticks > 0:
            self.avg_trade_tick = (self.imbalance_ticks / self.ticks)
        return self
        
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(' +
            ','.join([f'{str(k)}={str(v)}' for k, v in self.__dict__.items()]) +
            ')'
        )


def _diff(v1, v2):
    return (v1 - v2) if (v1 is not None) and (v2 is not None) else 0.0
