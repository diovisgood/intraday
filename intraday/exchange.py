import math
import random
from typing import Sequence, Union, Callable, Optional
from numbers import Real
from collections import namedtuple, OrderedDict
from datetime import timedelta, datetime
from arrow import Arrow
from .account import Account
from .provider import Trade, TradeOI


MarketOrder = namedtuple(
    'MarketOrder',
    'account operation amount time_init time_kill',
    defaults=(None,)*5
)

LimitOrder = namedtuple(
    'LimitOrder',
    'account operation amount time_init time_kill price',
    defaults=(None,)*6
)

StopOrder = namedtuple(
    'StopOrder',
    'account operation amount time_init time_kill price',
    defaults=(None,)*6
)

TrailingStopOrder = namedtuple(
    'TrailingStopOrder',
    'account operation amount time_init time_kill trail_delta best_price',
    defaults=(None,)*7
)

TakeProfitOrder = namedtuple(
    'TakeProfitOrder',
    'account operation amount time_init time_kill target_price trail_delta best_price',
    defaults=(None,)*8
)


class Exchange:
    
    def __init__(self,
                 accounts: Union[Sequence[Account], None] = None,
                 agent_order_delay: Union[Real, timedelta] = 2,
                 broker_order_delay: Union[Real, timedelta] = 0.5,
                 order_luck: float = 0.10,
                 commission: Union[Real, Callable[[str, Real, Real], Real]] = 20,
                 price_step: Real = 1,
                 ema_period: int = 20,
                 instant_balance_update=False,
                 **kwargs):
        # Setup parameters:

        # Initial balance to start trading in new episode
        assert (accounts is None) or isinstance(accounts, Sequence)
        self.accounts = []
        self.halted_accounts = set()
        if accounts is not None:
            for account in accounts:
                self.accounts.append(account)

        # Delay of execution of orders in seconds
        if isinstance(agent_order_delay, Real):
            assert (agent_order_delay > 0)
            self.agent_order_delay = timedelta(seconds=float(agent_order_delay))
        elif isinstance(agent_order_delay, timedelta):
            assert (agent_order_delay.total_seconds() > 0)
            self.agent_order_delay = agent_order_delay
        else:
            raise ValueError('agent_order_delay should be either real number of seconds or a positive timedelta object')
            
        # Delay of execution of broker-generated orders in seconds
        if isinstance(broker_order_delay, Real):
            assert (broker_order_delay > 0)
            self.broker_order_delay = timedelta(seconds=float(broker_order_delay))
        elif isinstance(broker_order_delay, timedelta):
            assert (agent_order_delay.total_seconds() > 0)
            self.broker_order_delay = broker_order_delay
        else:
            raise ValueError('broker_order_delay should be either real number of seconds or a positive timedelta object')

        # Chance of execution of orders when price reaches order price
        # but does not move further
        assert isinstance(order_luck, float) and (0 < order_luck <= 1.0)
        self.order_luck = order_luck

        # Commission for a trade: a number or a function
        assert isinstance(commission, Real) or callable(commission)
        self.commission = commission

        # Price step
        assert (price_step is None) or isinstance(price_step, Real)
        self.price_step = price_step

        # Period to measure exponential statistics
        assert isinstance(ema_period, int) and (ema_period > 0)
        self.ema_period = ema_period
        self.ema_factor = 2 / (ema_period + 1)
        
        # Balance can be updated instantly - on each new tick, or lazily, on order
        assert isinstance(instant_balance_update, bool)
        self.instant_balance_update = instant_balance_update
        
        # Initialize episode variables
        self.last_trade: Optional[Trade] = None
        self.spread_mean: Optional[float] = None
        self.spread_var: Optional[float] = None
        self.orders = OrderedDict()
        self.next_order_time: Optional[datetime] = None
    
    def reset(self, **kwargs):
        # Reset accounts
        for account in self.accounts:
            account.reset()
        self.halted_accounts.clear()
        # Reset episode variables
        self.last_trade = None
        self.spread_mean = None
        self.spread_var = None
        self.orders = OrderedDict()
        self.next_order_time = None

    def add_order(self, order: Union[MarketOrder, LimitOrder, StopOrder, TrailingStopOrder, TakeProfitOrder]) -> int:
        # Check account connected with order
        assert hasattr(order, 'account') and isinstance(order.account, Account)
        if order.account not in self.accounts:
            self.accounts.append(order.account)
        # Check order time
        if (self.last_trade is not None) and (order.time_init <= self.last_trade.datetime):
            raise ValueError('Cant add order in past time!')
        # Update price according to price_step
        if (self.price_step is not None) and hasattr(order, 'price') and (-math.inf < order.price < math.inf):
            price = math.floor(0.5 + order.price / self.price_step) * self.price_step
            order = order._replace(price=price)
        # Generate new id
        id = (max(self.orders.keys()) + 1) if (len(self.orders) > 0) else 1
        # Add order
        self.orders[id] = order
        # Check next order time
        if (self.next_order_time is None) or (self.next_order_time > order.time_init):
            self.next_order_time = order.time_init
        return id
    
    def get_order(self, id: int)-> (None, Union[MarketOrder, LimitOrder, StopOrder, TrailingStopOrder, TakeProfitOrder]):
        return self.orders[id] if (id in self.orders) else None

    def kill_order(self, id: Union[int, None], time_kill) -> (int, None):
        if id not in self.orders:
            return None
        order = self.orders[id]
        assert order.time_init <= time_kill
        self.orders[id] = order._replace(time_kill=time_kill)
        return id

    def replace_order(self, id: Union[int, None],
                      new_order: Union[MarketOrder, LimitOrder, StopOrder, TrailingStopOrder, TakeProfitOrder]
                      ) -> (int, None):
        if id in self.orders:
            self.kill_order(id, time_kill=new_order.time_init)
        return self.add_order(new_order)

    def process_trade(self, trade: Union[Trade, TradeOI]):
        """
        Updates table of orders and accounts balances due to the trade event

        Parameters
        ----------
        trade : Union[Trade, TradeOI]
            A trade with fields, at least: (datetime, operation, amount, price)
        """
        trade_datetime, price = trade.datetime, trade.price
        
        # Update accounts balance on each trade (which is slow), if instant_balance_update=True
        if self.instant_balance_update:
            self._update_balances(price, trade_datetime)
            
        # Calculate spread mean and variance
        if (self.last_trade is not None) and (self.last_trade.operation != trade.operation):
            spread = abs(self.last_trade.price - trade.price)
            self.spread_mean = (1 - self.ema_factor)*(self.spread_mean or spread) + self.ema_factor*spread
            spread_var = (spread - self.spread_mean)**2
            self.spread_var = (1 - self.ema_factor)*(self.spread_var or spread_var) + self.ema_factor*spread_var
            
        # Keep last trade info
        last_trade = self.last_trade
        self.last_trade = trade

        # Return if yet no orders to execute
        if (self.next_order_time is None) or (trade_datetime < self.next_order_time) or (len(self.orders) <= 0):
            return None
        
        # Reset next order time
        self.next_order_time = None
        
        # Calculate upper estimation bound of spread
        spread = (self.spread_mean or 0) + math.sqrt(self.spread_var or 0)/2

        # Estimate best bid/ask prices
        if trade.operation == 'B':
            best_ask_price = price
            best_bid_price = price - spread
        else:
            best_bid_price = price
            best_ask_price = price + spread
        assert (best_ask_price >= best_bid_price)
        
        # We cant modify self.orders while iterating over it
        # therefore we first make a shallow copy of it
        orders = self.orders.copy()
        callbacks = []
        updated_accounts = set()
        
        for id, order in orders.items():
            # These flag will be used in the end
            f_callback = False
            # Check order for validity
            if not self._order_is_valid(order, trade_datetime):
                # This order is invalid and should be erased
                del self.orders[id]
                continue

            # Update next order time
            if (self.next_order_time is None) or (self.next_order_time > order.time_init):
                self.next_order_time = order.time_init

            # If it is too early for order to be executed
            if trade_datetime < order.time_init:
                continue

            # Update balance for account, if not yet.
            # We need to update the balance of account because it may be subject to liquidation at current price.
            # That is why processing further orders for this account does not have sense.
            if (not self.instant_balance_update) and (order.account not in updated_accounts):
                order.account.update_balance(price)
                updated_accounts.add(order.account)
                # Halt account with negative balance and close its position
                if (order.account.balance < 0) and (order.account not in self.halted_accounts):
                    self._halt_account(order.account, trade_datetime)
                    del self.orders[id]
                    continue

            # Calculate direction and price delta
            direction = (1 if (order.operation == 'B') else -1)
            price_delta = direction * (price - order.price) if hasattr(order, 'price') else None

            if isinstance(order, MarketOrder):
                # Market order: execute immediately with best price
                execution_price = best_ask_price if (direction > 0) else best_bid_price
                commission = self._get_commission(order.operation, order.amount, execution_price)
                order.account.update(trade_datetime, order.operation, order.amount, execution_price, commission)
                del self.orders[id]
                f_callback = True

            elif isinstance(order, LimitOrder):
                # Limit order: execute if price has reached
                execution_price = None
                if (last_trade.datetime < order.time_init) and (price_delta < 0):
                    # Order has been just activated, but the price is already lower (to buy) or higher (to sell)
                    # So the order is executed at best current price
                    # (The same logic as with Market Orders)
                    if direction > 0:
                        execution_price = min(order.price, best_bid_price)
                    else:
                        execution_price = max(order.price, best_ask_price)
                if price_delta < 0:
                    # Price has passed the level of limit order.
                    # So we must consider that our limit order was executed by order price
                    execution_price = order.price
                if ((price_delta == 0) and (trade.operation != order.operation) and
                   (random.random() <= self.order_luck)):
                    # Price has reached exactly the limit order price.
                    # But order book typically has some depth, i.e. there are limit orders
                    # with the same price which are executed before our limit order.
                    # That is why we use self.order_luck as probability for order to be executed.
                    execution_price = order.price
                
                if execution_price is not None:
                    commission = self._get_commission(order.operation, order.amount, execution_price)
                    order.account.update(trade_datetime, order.operation, order.amount, execution_price, commission)
                    del self.orders[id]
                    f_callback = True

            elif isinstance(order, StopOrder):
                # Stop order: generate Market Order if price has reached
                if price_delta >= 0:
                    self.add_order(
                        MarketOrder(
                            account=order.account,
                            operation=order.operation,
                            amount=order.amount,
                            time_init=trade_datetime + self.broker_order_delay,
                            time_kill=None
                        )
                    )
                    del self.orders[id]

            elif isinstance(order, TrailingStopOrder):
                # Trailing finish order
                # 1. Start trailing immediately
                # 2. Execute order if price drops more than trail_delta from best_price
                if order.best_price is None:
                    self.orders[id] = order._replace(best_price=price)
                else:
                    price_delta = direction * (price - order.best_price)
                    if price_delta < 0:
                        # Update best price
                        self.orders[id] = order._replace(best_price=price)
                    elif price_delta >= order.trail_delta:
                        # Execute finish and generate Market Order
                        self.add_order(
                            MarketOrder(
                                account=order.account,
                                operation=order.operation,
                                amount=order.amount,
                                time_init=trade_datetime + self.broker_order_delay,
                                time_kill=None
                            )
                        )
                        del self.orders[id]

            elif isinstance(order, TakeProfitOrder):
                # Take profit order:
                # 1. Start trailing when price reaches target_price
                # 2. Execute order if price drops more than trail_delta from best_price
                if (order.best_price is None) and (direction * (price - order.target_price) <= 0):
                    # Setup first best_price kama_value
                    self.orders[id] = order._replace(best_price=price)
                elif order.best_price is not None:
                    price_delta = direction * (price - order.best_price)
                    if price_delta < 0:
                        # Update best price
                        self.orders[id] = order._replace(best_price=price)
                    elif price_delta >= order.trail_delta:
                        # Execute take profit and generate Market Order
                        self.add_order(
                            MarketOrder(
                                account=order.account,
                                operation=order.operation,
                                amount=order.amount,
                                time_init=trade_datetime + self.broker_order_delay,
                                time_kill=None
                            )
                        )
                        del self.orders[id]

            if f_callback:
                # Schedule update callback if needed
                if callable(order.account.on_update):
                    callbacks.append(order.account)
        
        # Yield callbacks
        callback_time = trade_datetime + self.agent_order_delay
        for account in callbacks:
            account.on_update(self, account, callback_time)
    
    def _order_is_valid(self, order, time: Union[datetime, Arrow]) -> bool:
        if (order is None) or (order.account in self.halted_accounts) or (order.time_init is None):
            return False
        if order.time_kill is not None:
            if order.time_kill <= order.time_init:
                return False
            elif order.time_kill <= time:
                return False
        return True
        
    def _update_balances(self, price: Real, dt: Union[Arrow, datetime]):
        for account in self.accounts:
            if (account.position != 0) and (account not in self.halted_accounts):
                account.update_balance(price)
                # Halt account with negative balance and close its position
                if account.balance < 0:
                    self._halt_account(account, dt)
                    
    def _halt_account(self, account: Account, dt: Union[Arrow, datetime]):
        # Mark account as halted
        self.halted_accounts.add(account)
        if account.position == 0:
            return None
        # Send order to close position on account
        self.add_order(
            MarketOrder(
                account=account,
                operation=('S' if (account.position > 0) else 'B'),
                amount=abs(account.position),
                time_init=dt + self.broker_order_delay,
                time_kill=None
            )
        )

    def _get_commission(self, operation, amount, price):
        if isinstance(self.commission, Real):
            return self.commission
        elif callable(self.commission):
            return self.commission(operation, amount, price)
        
    def __repr__(self):
        return (
            f'{self.__class__.__name__}{{\n'
            f'  instant_balance_update={self.instant_balance_update},\n'
            f'  agent_order_delay={self.agent_order_delay},\n'
            f'  broker_order_delay={self.broker_order_delay},\n'
            f'  order_luck={self.order_luck},\n'
            f'  commission={self.commission},\n'
            f'  ema_period={self.ema_period}\n'
            f'}}'
        )
