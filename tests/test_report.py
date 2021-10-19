import unittest
import copy
import numpy as np
from arrow import Arrow
from intraday.report import Record, Report, SECONDS_IN_GREGORIAN_YEAR

YEAR = SECONDS_IN_GREGORIAN_YEAR


class TestReport(unittest.TestCase):
    Records = [
        Record(operation=+1, amount=1, result=17, commission=2, enter_price=100, exit_price=100 + 17 + 2,
               enter_date=Arrow(2001, 1, 1, 0, 0, 0), exit_date=Arrow(2001, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
        Record(operation=+1, amount=1, result=15, commission=2, enter_price=100, exit_price=100 + 15 + 2,
               enter_date=Arrow(2002, 1, 1, 0, 0, 0), exit_date=Arrow(2002, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
        Record(operation=-1, amount=1, result=23, commission=2, enter_price=100, exit_price=100 - 23 - 2,
               enter_date=Arrow(2003, 1, 1, 0, 0, 0), exit_date=Arrow(2003, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
        Record(operation=+1, amount=1, result=-5, commission=2, enter_price=100, exit_price=100 - 5 - 2,
               enter_date=Arrow(2004, 1, 1, 0, 0, 0), exit_date=Arrow(2004, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
        Record(operation=-1, amount=1, result=12, commission=2, enter_price=100, exit_price=100 - 12 - 2,
               enter_date=Arrow(2005, 1, 1, 0, 0, 0), exit_date=Arrow(2005, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
        Record(operation=-1, amount=1, result=9, commission=2, enter_price=100, exit_price=100 - 9 - 2,
               enter_date=Arrow(2006, 1, 1, 0, 0, 0), exit_date=Arrow(2006, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
        Record(operation=+1, amount=1, result=13, commission=2, enter_price=100, exit_price=100 + 13 + 2,
               enter_date=Arrow(2007, 1, 1, 0, 0, 0), exit_date=Arrow(2007, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
        Record(operation=+1, amount=1, result=-4, commission=2, enter_price=100, exit_price=100 - 4 + 2,
               enter_date=Arrow(2008, 1, 1, 0, 0, 0), exit_date=Arrow(2008, 1, 1, 0, 0, 0).shift(seconds=YEAR)),
    ]
    
    def test_ratios(self):
        records = copy.deepcopy(self.Records)
        report = Report(initial_balance=1000, risk_free_rate=0, convention='raw')
        for record in records:
            report.add(record)
        self.maxDiff = None
        
        # Test counters
        self.assertEqual(5, report.n_trades_long)
        self.assertEqual(3, report.n_trades_short)
        self.assertEqual(6, report.n_trades_win)
        self.assertEqual(2, report.n_trades_loss)
        
        # Test duration
        total_duration = (Arrow(2008, 1, 1, 0, 0, 0).shift(seconds=YEAR) - Arrow(2001, 1, 1, 0, 0, 0)).total_seconds() / YEAR
        self.assertEqual(total_duration, report.total_duration)
        self.assertEqual(5, report.trades_duration_long)
        self.assertEqual(3, report.trades_duration_short)
        self.assertEqual(6, report.trades_duration_win)
        self.assertEqual(2, report.trades_duration_loss)
        
        # Test integral values
        gross_profit = 17 + 15 + 23 + 12 + 9 + 13
        gross_loss = -5 - 4
        self.assertEqual(gross_profit, report.gross_profit)
        self.assertEqual(gross_loss, report.gross_loss)
        self.assertEqual(gross_profit + gross_loss, report.net_profit)
        self.assertEqual(abs(gross_profit/gross_loss), report.profit_factor)
        
        rois = [0.17, 0.15, 0.23, -0.05, 0.12, 0.09, 0.13, -0.04]
        rois_mean = np.mean(rois).item()
        rois_std = np.std(rois).item()
        
        # Test Sharpe ratio
        self.assertAlmostEqual(rois_mean, report.roiann_mean, places=5)
        self.assertAlmostEqual(rois_std, report.roiann_std, places=5)
        self.assertAlmostEqual(rois_mean / rois_std, report.sharpe_ratio, places=5)

        # Test Sortino ratio
        tdds = [0, 0, 0, -0.05, 0, 0, 0, -0.04]
        tdd = np.sqrt(np.mean(np.power(tdds, 2)))
        self.assertAlmostEqual(rois_mean / tdd, report.sortino_ratio, places=5)
        
        # Test drawdown percent
        max_drawdown_percent = 5 / (1000 + 17 + 15 + 23)
        self.assertAlmostEqual(max_drawdown_percent, report.max_drawdown_percent, places=5)
        
        # Test Calmar ratio
        self.assertAlmostEqual(rois_mean / max_drawdown_percent, report.calmar_ratio, places=5)


if __name__ == '__main__':
    unittest.main()
