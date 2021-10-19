import datetime
import os
import unittest
from arrow import Arrow
from numbers import Real
from intraday.providers.moex import MoexArchiveProvider

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'data')
FILE_MASK = 'SPBFUT.RI??.*.trades.gz'


class TestMoexProvider(unittest.TestCase):
    def test_moex_archive_filter_dates(self):
        # Test date
        date_from = datetime.date(2020, 3, 20)
        date_to = datetime.date(2020, 3, 25)
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK, date_from=date_from, date_to=date_to)
        try:
            date_from = datetime.date(2019, 3, 20)
            date_to = datetime.date(2019, 3, 25)
            provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK, date_from=date_from, date_to=date_to)
            # We should never get here due to RuntimeError:
            self.assertTrue(False)
        except RuntimeError:
            pass

    def test_moex_archive_random_reset(self):
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        start1 = provider.reset()
        start2 = provider.reset()
        self.assertNotEqual(start1, start2)

    def test_moex_archive_desired_reset(self):
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        desired_start = Arrow(2020, 3, 20, 10, 30, 30, tzinfo='Europe/Moscow')
        actual_start = provider.reset(episode_start_datetime=desired_start)
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=0)
        
    def test_moex_archive_first_reset(self):
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        desired_start = Arrow(2020, 3, 20, 10, 0, 0, tzinfo='Europe/Moscow')
        actual_start = provider.reset(seek='first')
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=0)

    def test_moex_archive_last_reset(self):
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        desired_start = Arrow(2020, 3, 25, 10, 0, 0, tzinfo='Europe/Moscow')
        actual_start = provider.reset(seek='last')
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=0)

    def test_moex_archive_next_reset(self):
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        desired_start = Arrow(2020, 3, 23, 10, 0, 0, tzinfo='Europe/Moscow')
        provider.reset(seek='first')
        actual_start = provider.reset(seek='next')
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=0)

    def test_moex_archive_read(self):
        provider = MoexArchiveProvider(data_dir=DATA_DIR, files=FILE_MASK)
        expected_start = Arrow(2020, 3, 20, 10, 0, 0, tzinfo='Europe/Moscow')
        expected_end = Arrow(2020, 3, 20, 23, 49, 59, tzinfo='Europe/Moscow')
        actual_start = provider.reset(seek='first')
        self.assertAlmostEqual(actual_start.timestamp(), expected_start.timestamp(), places=0)
        last_datetime = None
        n_trades = 0
        try:
            while True:
                trade = next(provider)
                self.assertTrue(hasattr(trade, 'datetime') and isinstance(trade.datetime, (datetime.datetime, Arrow)))
                self.assertTrue(hasattr(trade, 'operation') and isinstance(trade.operation, str) and trade.operation in 'BS')
                self.assertTrue(hasattr(trade, 'amount') and isinstance(trade.amount, Real) and trade.amount > 0)
                self.assertTrue(hasattr(trade, 'price') and isinstance(trade.price, Real))
                self.assertTrue(hasattr(trade, 'open_interest') and isinstance(trade.open_interest, Real))
                self.assertTrue((last_datetime is None) or (trade.datetime >= last_datetime))
                self.assertGreaterEqual(trade.datetime, expected_start)
                self.assertGreaterEqual(expected_end, trade.datetime)
                last_datetime = trade.datetime
                n_trades += 1
        except StopIteration:
            pass
        self.assertEqual(n_trades, 538191)
        self.assertAlmostEqual(last_datetime.timestamp(), expected_end.timestamp(), places=-1)


if __name__ == '__main__':
    unittest.main()
