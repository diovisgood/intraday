from typing import Callable, Union, Any
from numbers import Real
import datetime as dt

from .report import Report, Record


class Account(object):
    def __init__(self,
                 initial_balance: Real = 10000,
                 risk_free_rate: Real = 0.0,
                 convention: str = 'raw'):
        # Save initial parameters
        self.initial_balance = initial_balance
        # Initialize account
        self.position = 0
        self.position_datetime = None
        self.position_price = None
        self.position_commission = 0
        self.position_roi = 0
        self.cash = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.min_balance = initial_balance
        self.max_drawdown = 0
        self.report = Report(initial_balance=initial_balance, risk_free_rate=risk_free_rate, convention=convention)
        self.subscribers = {}
        
    def reset(self):
        self.position = 0
        self.position_datetime = None
        self.position_price = None
        self.position_commission = 0
        self.position_roi = 0
        self.cash = self.initial_balance
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.min_balance = self.initial_balance
        self.max_drawdown = 0
        self.report.reset()
        self.subscribers.clear()
        
    def subscribe(self, who: Any, callback: Callable[..., Any]):
        self.subscribers[who] = callback
        
    def unsubscribe(self, who: Any):
        del self.subscribers[who]
        
    def on_update(self, *args, **kwargs):
        for who, callback in self.subscribers.items():
            callback(*args, **kwargs)

    def update_balance(self, price):
        # Update balance
        self.balance = self.cash + self.position * price
        if (self.position != 0) and (self.position_price != 0):
            self.position_roi = self.position * (price - self.position_price) / abs(self.position * self.position_price)
        else:
            self.position_roi = 0
        # Update max balance
        if self.max_balance < self.balance:
            self.max_balance = self.balance
            self.min_balance = self.balance
        # Update min balance
        if self.min_balance > self.balance:
            self.min_balance = self.balance
            # Update drawdown
            drawdown = (self.max_balance - self.min_balance)    
            if self.max_drawdown < drawdown:
                self.max_drawdown = drawdown
                
    def close_position(self,
                       price: Real,
                       datetime: Union[Real, dt.datetime, dt.date, Any],
                       commission: Real = 0,
                       notes: str = None):
        if self.position == 0:
            return

        # Add commission
        self.position_commission += commission
        # Create new record
        record = Record(
            operation=(1 if (self.position > 0) else -1),
            amount=abs(self.position),
            enter_date=self.position_datetime,
            enter_price=self.position_price,
            exit_date=datetime,
            exit_price=price,
            result=self.position * (price - self.position_price) - self.position_commission,
            commission=self.position_commission,
            notes=notes
        )
        # Update cash
        self.cash = self.cash + self.position * price - commission
        # Clear position
        self.position = 0
        self.position_datetime = None
        self.position_price = None
        self.position_commission = 0
        self.position_roi = 0
        # Update balance
        self.update_balance(price)
        # Add record to the list of records
        self.report.add(record)

    def update(self,
               datetime: Union[Real, dt.datetime, dt.date, Any],
               operation: str,
               amount: Real,
               price: Real,
               commission: Real = 0):
        """
        Update account by new operation.
        Each operation specifies buy or sell, amount, price and datetime of a trade.
        The position of account is updated according to the following scheme:
                            Update amount:
        Initial position:  +2   +1   -1   -2
                      +2    I    I    D    C
                      +1    I    I    C    R
                       0    S    S    S    S
                      -1    R    C    I    I
                      -2    C    D    I    I
        S - new position is set up
        I - current position is increased
        D - current position is decreased
        C - current position is closed
        R - current position is reverted (changing sign)
        :param datetime: one of: datetime, Arrow or number
        :param operation: string 'B' - buy, or 'S' - sell
        :param amount: positive number, > 0, may have floating point
        :param price: number, may have floating point
        :param commission: number, may have floating point
        :return: updated balance of account
        """
        assert isinstance(operation, str) and (operation in 'BS'), ValueError('Account:update: Invalid operation')
        assert isinstance(amount, Real) and (amount > 0), ValueError('Account:update: Invalid amount')
        assert isinstance(price, Real), ValueError('Account:update: Invalid price')
        
        # Convert amount to a signed number
        amount = amount if (operation == 'B') else -amount
        
        # Calculate new position caused by update
        new_position = (self.position + amount)

        if self.position == 0:
            # NO previous position
            # Simply setup new position
            self.position = amount
            self.position_datetime = datetime
            self.position_price = price
            self.position_commission += commission
            # Update cash
            self.cash = self.cash - amount * price - commission
            # Update balance
            self.update_balance(price)

        elif new_position == 0:
            # Position is CLOSED
            self.close_position(price, datetime, commission)
            
        elif self.position * amount > 0:
            # Current position is INCREASED by amount
            self.position_price = (self.position * self.position_price + amount * price) / new_position
            self.position_commission += commission
            self.position = new_position
            # Update cash
            self.cash = self.cash - amount * price - commission
            # Update balance
            self.update_balance(price)

        elif self.position * new_position > 0:
            # Position is DECREASED by amount
            # Create new record
            record = Record(
                operation=(1 if (self.position > 0) else -1),
                amount=abs(amount),
                enter_date=self.position_datetime,
                enter_price=self.position_price,
                exit_date=datetime,
                exit_price=price,
                result=(1 if (self.position > 0) else -1) * abs(amount) * (price - self.position_price) - commission,
                commission=commission,
                notes=None
            )
            # Note: we do not increase position_commission in this case!
            # Update cash
            self.cash = self.cash - amount * price - commission
            # Update position
            self.position = new_position
            # Update balance
            self.update_balance(price)
            # Add record to the list of records
            self.report.add(record)

        elif self.position * new_position < 0:
            # Position is REVERTED
            # Close position
            self.close_position(price, datetime, 0)
            # Setup new position
            self.position = new_position
            self.position_datetime = datetime
            self.position_commission += commission
            self.position_price = price
            # Update cash
            self.cash = self.cash - new_position * price - commission
            # Update balance
            self.update_balance(price)

        # Return updated balance
        return self.balance

    def __repr__(self):
        return f'{self.__class__.__name__}(initial_balance={self.initial_balance})'
    
    def __str__(self):
        return (
            f'{self.__class__.__name__}{{'
            f'position={self.position}, '
            f'position_datetime={str(self.position_datetime)}, '
            f'position_price={self.position_price}, '
            f'cash={self.cash}, '
            f'balance={self.balance}}}'
        )
