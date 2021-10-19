from datetime import datetime
from typing import Sequence
from abc import ABC, abstractmethod
from collections import defaultdict
from numbers import Real
import numpy as np
from gym import spaces
from .account import Account
from .exchange import Exchange, MarketOrder, StopOrder, TakeProfitOrder


class ActionScheme(ABC):
    """
    Base class for an action scheme
    
    See Also
    --------
    intraday.actions.BuySellCloseAction
    intraday.actions.PingPongAction
    """
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset is automatically invoked by Environment when it is being reset

        See Also
        --------
        intraday.MultiAgentEnv
        intraday.SingleAgentEnv
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_random_action(self):
        """
        Returns some random action
        Returns
        -------
        action : Any
            Random action
        See Also
        --------
        intraday.actions.ActionScheme.get_default_action
        """
        raise NotImplementedError()

    @abstractmethod
    def get_default_action(self):
        """
        Returns some default action.

        For example, in case of {Buy, Sell, Close} action scheme it can return Close action.

        Returns
        -------
        action : Any
            Random action
        See Also
        --------
        intraday.actions.ActionScheme.get_random_action
        """
        raise NotImplementedError()

    @abstractmethod
    def process_action(self, exchange: Exchange, account: Account, action, time: datetime):
        """
        This method is called by Environment instance to actually perform action, chosen by an agent.

        For example, in case of {Buy, Sell, Close} action scheme, if agent has chosen Buy,
        this method might create new MarketOrder to buy some fixed number of stocks.
        
        Parameters
        ----------
        exchange : Exchange
            An underlying Exchange object, which accepts and executes orders produced by an action scheme.
        account : Account
            An account instance associated with a particular agent, which issued an action.
        action : Any
            Action issued by an agent
        time : datetime
            A moment in time at which action has arrived.
        
        See Also
        --------
        intraday.MultiAgentEnv.step
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def space(self) -> spaces.Space:
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    
class BuySellCloseAction(ActionScheme):
    """
    Basic action scheme, where agent can choose from: {Buy, Sell, Close}
    
    Notes
    -----
    This scheme assumes you can open both long and short positions on some asset.
    It opens long positions of some predefined amount upon Buy signal.
    It opens short positions upon Sell signal.
    It closes any open position upon Close signal.
    
    Parameters
    ----------
    amount : Real
        The amount to be traded. Default: 1
    
    See Also
    --------
    intraday.actions.PingPongAction
    intraday.exchange.MarketOrder
    """
    space = spaces.Discrete(3)

    def __init__(self, amount: Real = 1, **kwargs):
        super().__init__(**kwargs)
        self.amount = abs(amount)

    def reset(self):
        """
        Reset is automatically invoked by Environment when it is being reset

        See Also
        --------
        intraday.MultiAgentEnv
        intraday.SingleAgentEnv
        """
        pass

    def get_random_action(self) -> int:
        """
        Returns random action, one of: {Buy, Sell, Close}.

        Returns
        -------
        action : int
            random value [0 .. 2]
        See Also
        --------
        intraday.actions.BuySellCloseAction.get_random_action
        """
        return np.random.randint(0, 3)

    def get_default_action(self) -> int:
        """
        Returns default action: Close.

        Returns
        -------
        action : int
            Value = 2
        See Also
        --------
        intraday.actions.BuySellCloseAction.get_random_action
        """
        return 2
    
    def process_action(self, exchange: Exchange, account: Account, action, time: datetime):
        """
        This method is called by Environment instance to actually perform action, chosen by an agent.

        Notes
        -----
        If agent has chosen Buy, this method creates new MarketOrder to buy some fixed number of assets.
        If agent has chosen Sell, this method creates new MarketOrder to sell some fixed number of assets.
        If agent has chosen Close and there is an open position (long or short),
        this method creates new MarketOrder to close that position.
        
        The amount for long or short positions is specified at initialization.

        Parameters
        ----------
        exchange : Exchange
            An underlying Exchange object, which accepts and executes orders produced by an action scheme.
        account : Account
            An account instance associated with a particular agent, which issued an action.
        action : Any
            Action issued by an agent
        time : datetime
            A moment in time at which action has arrived.

        See Also
        --------
        intraday.MultiAgentEnv.step
        intraday.exchange.MarketOrder
        """
        assert (0 <= action <= 2)
        # Calculate target position based on action
        target_position = self.amount if (action == 0) else -self.amount if (action == 1) else 0
        delta = (target_position - account.position)
        # Issue a market order if needed
        if delta != 0:
            order = MarketOrder(
                account=account,
                operation=('B' if (delta > 0) else 'S'),
                amount=abs(delta),
                time_init=time,
                time_kill=None
            )
            exchange.add_order(order)


class PingPongAction(ActionScheme):
    """
    A specialized action scheme for mean reversion trading
    
    Notes
    -----
    Market can be described as being in one of two modes: trend or consolidation.
    On a global scale markets are in trend mode for most of the time.
    But as you go deeper to a smaller timeframes you may find out that consolidation mode becomes significant.
    
    When market is in consolidation mode price goes up and down in some range.
    This is a good time for mean reversion trading.
    
    It means you open short position when price is near the upper bound,
    and then you open long position when price is near the lower bound.
    
    This action scheme performs it automatically for you.
    All you need is to provide four values as an action:
    - lower price bound,
    - upper price bound,
    - trail delta value to close position with profit, if price goes in your direction,
    - stop delta value to close position with stop-loss, if price goes not in your direction.
    
    When price goes up, crosses the upper bound and falls down more than trail delta:
    1. A trailing sell TakeProfitOrder is executed and short position is opened.
    2. A trailing buy TakeProfitOrder is created to automatically close (buy) position if price raises up
       more than trail delta from the local minimum.
    3. A StopOrder is created to automatically close (buy) position by stop loss if price goes up
       more than stop delta from the position price.

    When price goes down, crosses the lower bound and then rises up more than trail delta:
    1. A trailing buy TakeProfitOrder is executed and long position is opened.
    2. A trailing sell TakeProfitOrder is created to automatically close (sell) position if price falls down
       more than trail delta from the local minimum.
    3. A StopOrder is created to automatically close (sell) position by stop loss if price goes down
       more than stop delta from the position price.
    
    On each step an agent should provide an action as a tuple (or list, or numpy array) of four values:
    >>> (lower_price, upper_price, trail_delta, stop_delta)
    
    On each step agent can change these values according to its policy.
    Existing take-profit or stop-loss orders will be updated for new values.
    
    See Also
    --------
    intraday.actions.BuySellCloseAction
    intraday.exchange.TakeProfitOrder
    intraday.exchange.StopOrder
    """
    space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4,),
        dtype=np.float32
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position = defaultdict(lambda: None)
        self.lower_price = defaultdict(lambda: None)
        self.upper_price = defaultdict(lambda: None)
        self.trail_delta = defaultdict(lambda: None)
        self.stop_delta = defaultdict(lambda: None)
        self.buy_order_id = defaultdict(lambda: None)
        self.sell_order_id = defaultdict(lambda: None)
        self.buy_stop_order_id = defaultdict(lambda: None)
        self.sell_stop_order_id = defaultdict(lambda: None)

    def reset(self):
        """
        Reset is automatically invoked by Environment when it is being reset

        See Also
        --------
        intraday.MultiAgentEnv
        intraday.SingleAgentEnv
        """
        self.position.clear()
        self.lower_price.clear()
        self.upper_price.clear()
        self.trail_delta.clear()
        self.stop_delta.clear()
        self.buy_order_id.clear()
        self.sell_order_id.clear()
        self.buy_stop_order_id.clear()
        self.sell_stop_order_id.clear()

    def get_random_action(self) -> np.ndarray:
        """
        Returns random action
        
        Notes
        -----
        In fact, we can't make any realistic random action because here we don't know
        the price and its range.
        That is why using get_random_action with PingPongActionScheme is useless.
        """
        return np.array((-np.inf, np.inf, 0, 0))

    def get_default_action(self) -> np.ndarray:
        """
        Returns random action

        Notes
        -----
        In fact, we can't make any realistic default action because here we don't know
        the price and its range.
        That is why using get_default_action with PingPongActionScheme is useless.
        """
        return np.array((-np.inf, np.inf, 0, 0))

    def process_action(self, exchange: Exchange, account: Account, action, time: datetime):
        """
        This method is called by Environment instance to actually perform action, chosen by an agent.

        Notes
        -----
        On each step an agent should provide an action as a tuple (or list, or numpy array) of four values:
        >>> (lower_price, upper_price, trail_delta, stop_delta)

        The amount for long or short positions is always: 1.

        Parameters
        ----------
        exchange : Exchange
            An underlying Exchange object, which accepts and executes orders produced by an action scheme.
        account : Account
            An account instance associated with a particular agent, which issued an action.
        action : Any
            Action issued by an agent
        time : datetime
            A moment in time at which action has arrived.

        See Also
        --------
        intraday.exchange.TakeProfitOrder
        intraday.exchange.StopOrder
        """
        # Get new values for buy, sell prices and stop delta for this account
        assert isinstance(action, (Sequence, np.ndarray)) and (len(action) == 4)
        lower_price, upper_price, trail_delta, stop_delta = action
        
        # Setup account update callback if not yet
        if self.lower_price[account] is None:
            account.subscribe(self, lambda ex, acc, t: self.update(ex, acc, t))

        # Find out what values have been changed
        position_changed = (self.position[account] is None) or (self.position[account] != account.position)
        lower_price_changed = (self.lower_price[account] is None) or (self.lower_price[account] != lower_price)
        upper_price_changed = (self.upper_price[account] is None) or (self.upper_price[account] != upper_price)
        trail_delta_changed = (self.trail_delta[account] is None) or (self.trail_delta[account] != trail_delta)
        stop_delta_changed = (self.stop_delta[account] is None) or (self.stop_delta[account] != stop_delta)

        # Update values
        self.lower_price[account] = lower_price
        self.upper_price[account] = upper_price
        self.trail_delta[account] = trail_delta
        self.stop_delta[account] = stop_delta
        self.position[account] = account.position

        # Find out what orders should we change
        if account.position == 0:
            if lower_price_changed or trail_delta_changed or position_changed:
                self._open_buy(exchange, account, time)
            if upper_price_changed or trail_delta_changed or position_changed:
                self._open_sell(exchange, account, time)
        elif account.position > 0:
            if upper_price_changed or trail_delta_changed or position_changed:
                self._open_sell(exchange, account, time)
            if upper_price_changed or stop_delta_changed or position_changed:
                self._open_stop_sell(exchange, account, time)
        elif account.position < 0:
            if lower_price_changed or trail_delta_changed or position_changed:
                self._open_buy(exchange, account, time)
            if lower_price_changed or stop_delta_changed or position_changed:
                self._open_stop_buy(exchange, account, time)
            
    def update(self, exchange: Exchange, account: Account, time: datetime):
        # Get old and new position
        old_position = self.position[account] or 0
        new_position = account.position
        
        if (old_position < 0) and (new_position == 0):
            # Short position was closed by stop buy order
            self._kill_stop_buy(exchange, account, time)
        elif (old_position < 0) and (new_position > 0):
            # Short position was reverted to long by buy order
            self._kill_stop_buy(exchange, account, time)
            self._open_stop_sell(exchange, account, time)
            self._open_sell(exchange, account, time)
        elif (old_position > 0) and (new_position == 0):
            # Long position was closed by stop sell order
            self._kill_stop_sell(exchange, account, time)
        elif (old_position > 0) and (new_position < 0):
            # Long position was reverted to short by sell order
            self._kill_stop_sell(exchange, account, time)
            self._open_stop_buy(exchange, account, time)
            self._open_buy(exchange, account, time)
        elif (old_position == 0) and (new_position < 0):
            # We had no position and now we are in short
            self._open_stop_buy(exchange, account, time)
            self._open_buy(exchange, account, time)
        elif (old_position == 0) and (new_position > 0):
            # We had no position and now we are in long
            self._open_stop_sell(exchange, account, time)
            self._open_sell(exchange, account, time)
        elif (old_position == 0) and (new_position == 0):
            # We had no position and still we have none
            pass
        
        # Save new position
        self.position[account] = new_position
            
    def _open_sell(self, exchange: Exchange, account: Account, time: datetime):
        # Initialize take profit order to sell at a higher price
        if 1 + account.position > 0:
            self.sell_order_id[account] = exchange.replace_order(
                id=self.sell_order_id[account],
                new_order=TakeProfitOrder(
                    account=account,
                    operation='S',
                    amount=(1 + account.position),
                    time_init=time,
                    time_kill=None,
                    target_price=self.upper_price[account],
                    trail_delta=self.trail_delta[account],
                    best_price=None
                )
            )
            
    def _open_buy(self, exchange: Exchange, account: Account, time: datetime):
        # Initialize take profit order to buy at a lower price
        if 1 - account.position > 0:
            self.buy_order_id[account] = exchange.replace_order(
                id=self.buy_order_id[account],
                new_order=TakeProfitOrder(
                    account=account,
                    operation='B',
                    amount=(1 - account.position),
                    time_init=time,
                    time_kill=None,
                    target_price=self.lower_price[account],
                    trail_delta=self.trail_delta[account],
                    best_price=None
                )
            )
    
    def _kill_buy(self, exchange: Exchange, account: Account, time: datetime):
        # Kill take profit order to buy
        if self.buy_order_id[account] is not None:
            exchange.kill_order(self.buy_order_id[account], time_kill=time)
            self.buy_order_id[account] = None

    def _kill_sell(self, exchange: Exchange, account: Account, time: datetime):
        # Kill take profit order to sell
        if self.sell_order_id[account] is not None:
            exchange.kill_order(self.sell_order_id[account], time_kill=time)
            self.sell_order_id[account] = None
        
    def _open_stop_buy(self, exchange: Exchange, account: Account, time: datetime):
        # Initialize stop order to buy if price raises beyond some level
        self.buy_stop_order_id[account] = exchange.replace_order(
            id=self.buy_stop_order_id[account],
            new_order=StopOrder(
                account=account,
                operation='B',
                amount=abs(account.position),
                price=(account.position_price + self.stop_delta[account]),
                time_init=time,
                time_kill=None
            )
        )

    def _open_stop_sell(self, exchange: Exchange, account: Account, time: datetime):
        # Initialize stop order to sell if price drops below some level
        self.sell_stop_order_id[account] = exchange.replace_order(
            id=self.sell_stop_order_id[account],
            new_order=StopOrder(
                account=account,
                operation='S',
                amount=account.position,
                price=(account.position_price - self.stop_delta[account]),
                time_init=time,
                time_kill=None
            )
        )
        
    def _kill_stop_buy(self, exchange: Exchange, account: Account, time: datetime):
        # Kill stop order to buy
        if self.buy_stop_order_id[account] is not None:
            exchange.kill_order(self.buy_stop_order_id[account], time_kill=time)
            self.buy_stop_order_id[account] = None

    def _kill_stop_sell(self, exchange: Exchange, account: Account, time: datetime):
        # Kill stop order to sell
        if self.sell_stop_order_id[account] is not None:
            exchange.kill_order(self.sell_stop_order_id[account], time_kill=time)
            self.sell_stop_order_id[account] = None
