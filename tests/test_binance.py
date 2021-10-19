import os
import unittest
from fnmatch import fnmatch
from numbers import Real
import arrow
from datetime import date, datetime
from intraday.providers.binance import BinanceArchiveProvider

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'data')
SYMBOL = 'ONTBNB'
DATE1 = date(2018, 5, 1)
DATE2 = date(2018, 8, 31)


class TestBinanceProvider(unittest.TestCase):
    @staticmethod
    def _remove_files(file_mask: str):
        for file_name in os.listdir(DATA_DIR):
            if fnmatch(file_name, file_mask):
                os.remove(os.path.join(DATA_DIR, file_name))

    def test_binance_archive_download(self):
        self._remove_files(f'{SYMBOL}-trades-*.zip')
        self._remove_files(f'{SYMBOL}-trades-*.feather')
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=DATE1, date_to=DATE2)
        d = arrow.get(DATE1)
        while d < arrow.get(DATE2):
            file_name = os.path.join(DATA_DIR, f'{SYMBOL}-trades-{d.year:04}-{d.month:02}')
            self.assertTrue(os.path.exists(file_name + '.feather'))
            d = d.shift(months=1)
        self._remove_files(f'{SYMBOL}-trades-*.zip')

    def test_binance_archive_random_reset(self):
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=DATE1, date_to=DATE2)
        start1 = provider.reset()
        start2 = provider.reset()
        self.assertNotEqual(start1, start2)

    def test_binance_archive_desired_reset(self):
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=DATE1, date_to=DATE2)
        desired_start = arrow.get(2018, 5, 10, 10, 1, 0, tzinfo='UTC')
        actual_start = provider.reset(episode_start_datetime=desired_start)
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=-2)
        
    def test_binance_archive_first_reset(self):
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=DATE1, date_to=DATE2)
        desired_start = arrow.get(2018, 5, 1, 0, 0, 0, tzinfo='UTC')
        actual_start = provider.reset(seek='first')
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=-3)

    def test_binance_archive_last_reset(self):
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=DATE1, date_to=DATE2)
        desired_start = arrow.get(2018, 8, 1, 0, 0, 0, tzinfo='UTC')
        actual_start = provider.reset(seek='last')
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=-3)

    def test_binance_archive_next_reset(self):
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=DATE1, date_to=DATE2)
        desired_start = arrow.get(2018, 6, 1, 0, 0, 0, tzinfo='UTC')
        provider.reset(seek='first')
        actual_start = provider.reset(seek='next')
        self.assertAlmostEqual(desired_start.timestamp(), actual_start.timestamp(), places=-3)
        
    def test_binance_archive_filter_dates(self):
        # Test date
        date_from = arrow.get(DATE1.year, DATE1.month, DATE1.day, 12, 0, 0, tzinfo='utc')
        date_to = arrow.get(DATE2.year, DATE2.month, DATE2.day, 23, 59, 59, 999999, tzinfo='utc')
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=date_from, date_to=date_to)
        try:
            episode_start_datetime = arrow.get(DATE1.year, DATE1.month, DATE1.day, 0, 0, 0, tzinfo='utc')
            provider.reset(episode_start_datetime=episode_start_datetime)
            # We should never get here due to RuntimeError:
            self.assertTrue(False)
        except ValueError:
            pass

    def test_binance_archive_read(self):
        provider = BinanceArchiveProvider(data_dir=DATA_DIR, symbol=SYMBOL, date_from=DATE1, date_to=DATE2)
        expected_start = arrow.get(2018, 7, 20, 12, 0, 0, tzinfo='UTC')
        expected_end = arrow.get(2018, 8, 31, 23, 59, 59, tzinfo='UTC')
        actual_start = provider.reset(episode_start_datetime=expected_start)
        self.assertAlmostEqual(actual_start.timestamp(), expected_start.timestamp(), places=-3)
        last_datetime = None
        n_trades = 0
        try:
            while True:
                trade = next(provider)
                self.assertTrue(hasattr(trade, 'datetime') and isinstance(trade.datetime, (datetime, arrow.Arrow)))
                self.assertTrue(hasattr(trade, 'operation') and isinstance(trade.operation, str) and trade.operation in 'BS')
                self.assertTrue(hasattr(trade, 'amount') and isinstance(trade.amount, Real) and trade.amount > 0)
                self.assertTrue(hasattr(trade, 'price') and isinstance(trade.price, Real))
                self.assertTrue((last_datetime is None) or (trade.datetime >= last_datetime))
                self.assertGreaterEqual(trade.datetime, expected_start)
                self.assertGreaterEqual(expected_end, trade.datetime)
                last_datetime = trade.datetime
                n_trades += 1
        except StopIteration:
            pass
        self.assertEqual(n_trades, 104221)
        self.assertAlmostEqual(last_datetime.timestamp(), expected_end.timestamp(), places=-3)


if __name__ == '__main__':
    unittest.main()
