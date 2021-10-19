import unittest
import copy
import numpy as np
from arrow import Arrow
from intraday.account import Account, Record, Report


class TestAccount(unittest.TestCase):
    Regular_Updates = [
        dict(datetime=Arrow(2020, 1, 1, 9, 0, 0).float_timestamp, operation='B', amount=2, price=100, commission=2),
        dict(datetime=Arrow(2020, 1, 1, 9, 30, 0).float_timestamp, operation='S', amount=2, price=100, commission=2),
    
        dict(datetime=Arrow(2020, 1, 1, 10, 0, 0).float_timestamp, operation='B', amount=2, price=100, commission=2),

        dict(datetime=Arrow(2020, 1, 1, 11, 0, 0).float_timestamp, operation='S', amount=4, price=110, commission=4),

        dict(datetime=Arrow(2020, 1, 1, 12, 0, 0).float_timestamp, operation='B', amount=4, price=115, commission=4),
        dict(datetime=Arrow(2020, 1, 1, 13, 0, 0).float_timestamp, operation='B', amount=1, price=110, commission=1),
        dict(datetime=Arrow(2020, 1, 1, 15, 0, 0).float_timestamp, operation='S', amount=3, price=120, commission=3),

        dict(datetime=Arrow(2020, 1, 2, 10, 0, 0).float_timestamp, operation='S', amount=2, price=150, commission=2),
        dict(datetime=Arrow(2020, 1, 2, 12, 0, 0).float_timestamp, operation='B', amount=1, price=140, commission=1),
        dict(datetime=Arrow(2020, 1, 2, 14, 0, 0).float_timestamp, operation='B', amount=1, price=130, commission=1),
    ]
    
    Regular_Records = [
        Record(operation=+1, amount=2, result=0 - 4, commission=4,
               enter_date=Arrow(2020, 1, 1, 9, 0, 0).float_timestamp, enter_price=100,
               exit_date=Arrow(2020, 1, 1, 9, 30, 0).float_timestamp, exit_price=100),
        Record(operation=+1, amount=2, result=2*(110 - 100) - 2, commission=2,
               enter_date=Arrow(2020, 1, 1, 10, 0, 0).float_timestamp, enter_price=100,
               exit_date=Arrow(2020, 1, 1, 11, 0, 0).float_timestamp, exit_price=110),
        Record(operation=-1, amount=2, result=2*(110 - 115) - 4, commission=4,
               enter_date=Arrow(2020, 1, 1, 11, 0, 0).float_timestamp, enter_price=110,
               exit_date=Arrow(2020, 1, 1, 12, 0, 0).float_timestamp, exit_price=115),
        Record(operation=+1, amount=3, result=3*(120 - 340/3) - 8, commission=8,
               enter_date=Arrow(2020, 1, 1, 12, 0, 0).float_timestamp, enter_price=340/3,
               exit_date=Arrow(2020, 1, 1, 15, 0, 0).float_timestamp, exit_price=120),
        Record(operation=-1, amount=1, result=(150 - 140) - 1, commission=1,
               enter_date=Arrow(2020, 1, 2, 10, 0, 0).float_timestamp, enter_price=150,
               exit_date=Arrow(2020, 1, 2, 12, 0, 0).float_timestamp, exit_price=140),
        Record(operation=-1, amount=1, result=(150 - 130) - 3, commission=3,
               enter_date=Arrow(2020, 1, 2, 10, 0, 0).float_timestamp, enter_price=150,
               exit_date=Arrow(2020, 1, 2, 14, 0, 0).float_timestamp, exit_price=130),
    ]
    
    def test_records(self):
        updates = copy.deepcopy(self.Regular_Updates)
        account = Account(initial_balance=1000)
        for update in updates:
            account.update(**update)
        self.maxDiff = None
        self.assertListEqual(self.Regular_Records, account.report.records)
    
    def test_report(self):
        updates = copy.deepcopy(self.Regular_Updates)
        account = Account(initial_balance=1000)
        for update in updates:
            account.update(**update)
        
        records = copy.deepcopy(self.Regular_Records)
        gross_profit = 0
        gross_loss = 0
        returns_on_investment = []
        sortino_downside_returns = []
        for record in records:
            if record.result >= 0:
                gross_profit += record.result
            else:
                gross_loss += record.result
            duration = (record.exit_date - record.enter_date)
            result = (record.result or 0)
            roi = result / (record.amount * record.enter_price)
            returns_on_investment.append(roi)
            sortino_downside_returns.append(min(0, roi))

        # Compute mean and standard deviation of returns
        roi_mean = np.mean(returns_on_investment) if (len(returns_on_investment) > 0) else None
        roi_std = np.std(returns_on_investment) if (len(returns_on_investment) > 0) else 0
        # Compute target downside deviation
        roi_tdd = np.sqrt(np.mean(np.power(sortino_downside_returns, 2))) if (len(sortino_downside_returns) > 0) else 0

        sharpe_ratio = roi_mean / roi_std if (roi_std != 0) else None
        sortino_ratio = roi_mean / roi_tdd if (roi_tdd != 0) else None
        # calmar_ratio = roi_mean / max_drawdown_percent if (max_drawdown_percent != 0) else None
        profit_factor = abs(gross_profit / gross_loss) if (gross_loss != 0) else None

        self.maxDiff = None
        self.assertEqual(gross_profit, account.report.gross_profit)
        self.assertEqual(gross_loss, account.report.gross_loss)
        self.assertEqual(sharpe_ratio, account.report.sharpe_ratio)
        self.assertEqual(sortino_ratio, account.report.sortino_ratio)
        # self.assertEqual(calmar_ratio, account.report.calmar_ratio)
        self.assertEqual(profit_factor, account.report.profit_factor)


if __name__ == '__main__':
    unittest.main()
