from collections import namedtuple
from typing import Sequence, Union, Any
from numbers import Real
import numpy as np
from datetime import date, datetime, timedelta


SECONDS_IN_GREGORIAN_YEAR = 31556952


Record = namedtuple(
    'Record',
    'operation amount enter_date enter_price exit_date exit_price result commission notes',
    defaults=(None,) * 9
)


def duration(date1: Union[Real, datetime, date, Any],
             date2: Union[Real, datetime, date, Any],
             convention: str
             ) -> float:
    if (date1 is None) or (date2 is None):
        return 0
    delta = date1 - date2
    if convention in {'actual/365', 'actual_365', 'actual365'}:
        if isinstance(delta, timedelta):
            return abs(delta.days / 365)

    elif convention in {'actual/360', 'actual_360', 'actual360'}:
        if isinstance(delta, timedelta):
            return abs(delta.days / 360)

    elif convention in {'30/360', '30_360'}:
        assert hasattr(date1, 'year') and hasattr(date1, 'month') and hasattr(date1, 'day')
        assert hasattr(date2, 'year') and hasattr(date2, 'month') and hasattr(date2, 'day')
        df = (min(date2.day, 30) + max(0, (30 - date1.day))) / 360
        mf = (date2.month - date1.month - 1) / 12
        yf = (date2.year - date1.year)
        return abs(df + mf + yf)

    else:
        if isinstance(delta, Real):
            pass
        elif isinstance(delta, timedelta):
            delta = delta.total_seconds()
        else:
            raise ValueError('Invalid date1 or date2')
        return abs(delta) / SECONDS_IN_GREGORIAN_YEAR


class Report(object):
    def __init__(self,
                 records: (None, Sequence[Record]) = None,
                 initial_balance: Real = 100000,
                 risk_free_rate: Real = 0.0,
                 convention: str = 'raw'):
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.convention = convention
        
        self.first_time = None
        self.last_time = None
        self.net_profit = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.max_net_profit = 0
        self.max_drawdown = 0
        self.max_drawdown_percent = 0
        self.total_commission = 0
        self.long_result = 0
        self.short_result = 0
        self.n_trades = 0
        self.n_trades_long = 0
        self.n_trades_short = 0
        self.n_trades_win = 0
        self.n_trades_loss = 0
        self.trades_duration = 0
        self.trades_duration_long = 0
        self.trades_duration_short = 0
        self.trades_duration_win = 0
        self.trades_duration_loss = 0
        self.returns_on_investments = []
        self.sortino_downside_returns = []
        self.returns_on_investments_annual = []
        self.sortino_downside_returns_annual = []
        self.records = []
        
        if records is not None:
            for record in records:
                self.add(record)
    
    def reset(self):
        self.first_time = None
        self.last_time = None
        self.net_profit = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.max_net_profit = 0
        self.max_drawdown = 0
        self.max_drawdown_percent = 0
        self.total_commission = 0
        self.long_result = 0
        self.short_result = 0
        self.n_trades = 0
        self.n_trades_long = 0
        self.n_trades_short = 0
        self.n_trades_win = 0
        self.n_trades_loss = 0
        self.trades_duration = 0
        self.trades_duration_long = 0
        self.trades_duration_short = 0
        self.trades_duration_win = 0
        self.trades_duration_loss = 0
        self.returns_on_investments.clear()
        self.sortino_downside_returns.clear()
        self.returns_on_investments_annual.clear()
        self.sortino_downside_returns_annual.clear()
        self.records.clear()
    
    def add(self, record: Record):
        # Add record to the list
        self.records.append(record)
        result = (record.result or 0)
        # Update first and last times
        if (self.first_time is None) or (self.first_time > record.enter_date):
            self.first_time = record.enter_date
        if (self.last_time is None) or (self.last_time < record.exit_date):
            self.last_time = record.exit_date
        # Calculate return-on-investment (roi)
        roi = result / (record.amount * record.enter_price)
        self.returns_on_investments.append(roi)
        self.sortino_downside_returns.append(min(0, roi - self.risk_free_rate))
        # Calculate annualized return-on-investment (roiann)
        d = duration(record.exit_date, record.enter_date, self.convention)
        if d != 0:
            roiann = roi / d
            self.returns_on_investments_annual.append(roiann)
            self.sortino_downside_returns_annual.append(min(0, roiann - self.risk_free_rate))
        # Calculate some integral features
        self.net_profit += result
        self.total_commission += (record.commission or 0)
        self.n_trades += 1
        self.trades_duration += d
        # Update long/short counters
        if record.operation > 0:
            self.long_result += result
            self.n_trades_long += 1
            self.trades_duration_long += d
        else:
            self.short_result += result
            self.n_trades_short += 1
            self.trades_duration_short += d
        # Update win/loss counters
        if result > 0:
            self.gross_profit += result
            self.n_trades_win += 1
            self.trades_duration_win += d
        elif result < 0:
            self.gross_loss += result
            self.n_trades_loss += 1
            self.trades_duration_loss += d
        # Calculate max drawdown
        if self.max_net_profit < self.net_profit:
            self.max_net_profit = self.net_profit
        if self.max_drawdown < (self.max_net_profit - self.net_profit):
            self.max_drawdown = (self.max_net_profit - self.net_profit)
            self.max_drawdown_percent = self.max_drawdown / (self.initial_balance + self.max_net_profit)

    @property
    def total_duration(self):
        """Total investment period duration, using specified `convention`."""
        return duration(self.last_time, self.first_time, self.convention)

    @property
    def roi_mean(self):
        """Mean value for returns on investments"""
        return np.mean(self.returns_on_investments) if (len(self.returns_on_investments) > 0) else None

    @property
    def roi_std(self):
        """Standard deviation over returns on investments"""
        return np.std(self.returns_on_investments) if (len(self.returns_on_investments) > 0) else 0

    @property
    def roi_tdd(self):
        """Target downside deviation over returns on investments"""
        return np.sqrt(np.mean(np.power(self.sortino_downside_returns, 2))) if (
            len(self.sortino_downside_returns) > 0) else 0

    @property
    def roiann_mean(self):
        """Mean value for annualized returns on investments"""
        return np.mean(self.returns_on_investments_annual) if (len(self.returns_on_investments_annual) > 0) else None
    
    @property
    def roiann_std(self):
        """Standard deviation over annualized returns on investments"""
        return np.std(self.returns_on_investments_annual) if (len(self.returns_on_investments_annual) > 0) else 0
    
    @property
    def roiann_tdd(self):
        """Target downside deviation over annualized returns on investments"""
        return np.sqrt(np.mean(np.power(self.sortino_downside_returns_annual, 2))) if (
            len(self.sortino_downside_returns_annual) > 0) else 0
    
    @property
    def sortino_ratio(self):
        """Sortino ratio over returns on investments"""
        roi_mean = self.roi_mean
        roi_tdd = self.roi_tdd
        return (
            (roi_mean - self.risk_free_rate) / roi_tdd
            if (roi_mean is not None) and (roi_tdd != 0) else None
        )

    @property
    def sortino_ratio_annual(self):
        """Sortino ratio over annualized returns on investments"""
        roiann_mean = self.roiann_mean
        roiann_tdd = self.roiann_tdd
        return (
            (roiann_mean - self.risk_free_rate) / roiann_tdd
            if (roiann_mean is not None) and (roiann_tdd != 0) else None
        )
    
    @property
    def sharpe_ratio(self):
        """Sharpe ratio over returns on investments"""
        roi_mean = self.roi_mean
        roi_std = self.roi_std
        return roi_mean / roi_std if (roi_mean is not None) and (roi_std != 0) else None

    @property
    def sharpe_ratio_annual(self):
        """Sharpe ratio over annualized returns on investments"""
        roiann_mean = self.roiann_mean
        roiann_std = self.roiann_std
        return roiann_mean / roiann_std if (roiann_mean is not None) and (roiann_std != 0) else None
    
    @property
    def calmar_ratio(self):
        """Calmar ratio over returns on investments"""
        roi_mean = self.roi_mean
        max_drawdown_percent = self.max_drawdown_percent
        return (
            (roi_mean / max_drawdown_percent)
            if (roi_mean is not None) and (max_drawdown_percent != 0)
            else None
        )

    @property
    def calmar_ratio_annual(self):
        """Calmar ratio over annualized returns on investments"""
        roiann_mean = self.roiann_mean
        max_drawdown_percent = self.max_drawdown_percent
        return (
            (roiann_mean / max_drawdown_percent)
            if (roiann_mean is not None) and (max_drawdown_percent != 0)
            else None
        )
    
    @property
    def ror(self):
        """Rate of return"""
        return (
            (self.net_profit / self.initial_balance)
            if (self.initial_balance != 0)
            else None
        )
    
    @property
    def rorann(self):
        """Annualized rate of return"""
        ror = self.ror
        d = duration(self.last_time, self.first_time, self.convention)
        return (ror / d) if (ror is not None) and (d != 0) else None
    
    @property
    def recovery_factor(self):
        rorann = self.rorann
        max_drawdown_percent = self.max_drawdown_percent
        return (rorann / (max_drawdown_percent + 1e-8)) if (rorann and max_drawdown_percent) else None
    
    @property
    def profit_factor(self):
        return abs(self.gross_profit / self.gross_loss) if (self.gross_loss != 0) else None
    
    @property
    def average_trade_result(self):
        return (self.net_profit / self.n_trades) if (self.n_trades > 0) else None
    
    @property
    def average_long_trade_result(self):
        return (self.long_result / self.n_trades_long) if (self.n_trades_long > 0) else None
    
    @property
    def average_short_trade_result(self):
        return (self.short_result / self.n_trades_short) if (self.n_trades_short > 0) else None
    
    @property
    def average_win_trade_result(self):
        return (self.gross_profit / self.n_trades_win) if (self.n_trades_win > 0) else None
    
    @property
    def average_loss_trade_result(self):
        return (self.gross_loss / self.n_trades_loss) if (self.n_trades_loss > 0) else None
    
    @property
    def average_trade_duration(self):
        return (self.trades_duration / self.n_trades) if (self.n_trades > 0) else None
    
    @property
    def average_long_trade_duration(self):
        return (self.trades_duration_long / self.n_trades_long) if (self.n_trades_long > 0) else None
    
    @property
    def average_short_trade_duration(self):
        return (self.trades_duration_short / self.n_trades_short) if (self.n_trades_short > 0) else None
    
    @property
    def average_win_trade_duration(self):
        return (self.trades_duration_win / self.n_trades_win) if (self.n_trades_win > 0) else None
    
    @property
    def average_loss_trade_duration(self):
        return (self.trades_duration_loss / self.n_trades_loss) if (self.n_trades_loss > 0) else None
    
    def __repr(self):
        return (
            f'{self.__class__.__name__}('
            f'initial_balance={self.initial_balance}, '
            f'risk_free_rate={self.risk_free_rate}, '
            f'convention={self.convention})'
        )
    
    def __str__(self):
        return '\n'.join([f'{str(k)}: {str(v)}' for (k, v) in self.__dict__.items()])
