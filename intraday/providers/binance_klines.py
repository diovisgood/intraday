from typing import List, NamedTuple, Union, Optional, Literal, Deque
from collections import deque
import os
from numbers import Real
import arrow
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone

from intraday.provider import Trade, Kline
from intraday.simulator import Simulator


class BinanceKlines(Simulator):
    duration = 60
    
    def __init__(self,
                 data_dir: str,
                 symbol: str,
                 spread: float = 0.0005,
                 date_from: Optional[Union[date, datetime, arrow.Arrow]] = None,
                 date_to: Optional[Union[date, datetime, arrow.Arrow]] = None,
                 **kwargs):
        super().__init__(spread=spread, **kwargs)
        
        assert isinstance(data_dir, str) and (data_dir > '') and os.path.isdir(data_dir)
        self.data_dir = data_dir
        
        assert isinstance(symbol, str)
        symbol = symbol.upper()
        self.symbol = symbol
        
        if date_to is None:
            date_to = arrow.now()
        elif isinstance(date_to, date):
            date_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
        if isinstance(date_to, datetime):
            date_to = arrow.get(date_to.astimezone(timezone.utc))
        assert isinstance(date_to, arrow.Arrow)
        self.date_to = date_to
        
        if date_from is None:
            date_from = date_to.shift(months=-12)
        elif isinstance(date_from, date):
            date_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        if isinstance(date_from, datetime):
            date_from = arrow.get(date_from.astimezone(timezone.utc))
        assert isinstance(date_from, arrow.Arrow)
        self.date_from = date_from
        
        assert date_from <= date_to
        
        self.files = {}
        self._first_datetime: Optional[float] = None
        self._last_datetime: Optional[float] = None

        d = arrow.get(date_from.year, date_from.month, 1, 0, 0, tzinfo=timezone.utc)
        while d < date_to:
            # Check if we have .feather file for trades of a particular year-month
            year, month = d.year, d.month
            file_name = f'{self.symbol}-1m-{year:04}-{month:02}'
            file_path_feather = os.path.join(data_dir, file_name + '.feather')
            if not os.path.exists(file_path_feather):
                # Download .zip archive and convert it to a faster .feather format
                file_path_zip = os.path.join(data_dir, file_name + '.zip')
                if not os.path.exists(file_path_zip):
                    self.download_month_archive(symbol, year, month, file_path_zip)
                assert os.path.exists(file_path_zip)
                self.convert_month_archive(file_path_zip)
                os.remove(file_path_zip)
            # Add filename for month
            assert os.path.isfile(file_path_feather)
            self.files[year * 100 + month] = file_name + '.feather'
            # Update first and last datetime
            first_datetime, last_datetime = d.span('month')
            if (self._first_datetime is None) or (self._first_datetime > first_datetime):
                self._first_datetime = first_datetime
            if (self._last_datetime is None) or (self._last_datetime < last_datetime):
                self._last_datetime = last_datetime
            # Shift to the next month
            d = d.shift(months=1)

        self.months: List[int] = sorted(list(self.files.keys()))
        
        # Prepare episode variables
        self._df: Optional[pd.DataFrame] = None
        self._file_index: Optional[int] = None
        self._record_index: Optional[int] = None
        self._trades: Deque[Trade] = deque()
        self._episode_start_datetime: Optional[Union[datetime, arrow.Arrow]] = None
    
    @staticmethod
    def download_month_archive(symbol: str,
                               year: int,
                               month: int,
                               file_path_zip: str,
                               market: Literal['spot', 'futures'] = 'spot'):
        url = f'https://data.binance.vision/data/{market}/monthly/klines/{symbol}/1m/{symbol}-1m-{year:04}-{month:02}.zip'
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    @staticmethod
    def convert_month_archive(file_path_zip: str):
        # Read dataframe from a zipped csv file
        df = pd.read_csv(
            file_path_zip,
            compression='infer',
            sep=',',
            header=None,
            names=['time_start', 'open', 'high', 'low', 'close', 'volume',
                   'time_end', 'money', 'n_trades', 'buy_volume', 'buy_money', 'ignore'],
            dtype={'time_start': np.int64, 'open': np.float32, 'high': np.float32, 'low': np.float32,
                   'close': np.float32, 'volume': np.float32, 'time_end': np.int64,
                   'money': np.float32, 'n_trades': np.int16,
                   'buy_volume': np.float32, 'buy_money': np.float32,
                   'ignore': np.float32},
        )
        df.drop(columns=['time_end', 'ignore'], inplace=True)
        df.time_start = pd.to_datetime(df.time_start, unit='ms', utc=True)
        # Save dataframe in feather format
        file_path_feather = file_path_zip.rsplit(os.extsep, 1)[0] + '.feather'
        df.reset_index().to_feather(file_path_feather)

    @staticmethod
    def load_month_archive(file_path: str) -> pd.DataFrame:
        df = pd.read_feather(file_path)
        return df

    def reset(self,
              episode_start_datetime: Union[None, arrow.Arrow, datetime] = None,
              episode_min_duration: Union[None, Real, timedelta] = None,
              seek: Optional[Literal['first', 'next', 'last']] = None,
              rng: Optional[np.random.RandomState] = None
              ) -> datetime:
        # Check episode_min_duration
        if episode_min_duration is None:
            episode_min_duration = timedelta(seconds=0)
        elif isinstance(episode_min_duration, Real):
            episode_min_duration = timedelta(seconds=float(episode_min_duration))
        elif isinstance(episode_min_duration, timedelta):
            pass
        else:
            raise ValueError('Invalid episode_min_duration value')
        assert episode_min_duration.total_seconds() >= 0

        # Choose month for next episode
        if seek is not None:
            assert isinstance(seek, str) and (seek in {'first', 'next', 'last'})
            if seek == 'first':
                file_index = 0
                file_month = self.months[file_index]
                file_name = self.files[file_month]
            elif seek == 'last':
                file_index = len(self.files) - 1
                file_month = self.months[file_index]
                file_name = self.files[file_month]
            elif seek == 'next':
                if self._file_index is None:
                    raise RuntimeError('Can not seek to next episode without prior episode')
                file_index = self._file_index + 1
                if file_index >= len(self.files):
                    raise StopIteration('No more files')
                file_month = self.months[file_index]
                file_name = self.files[file_month]
            else:
                raise ValueError('seek can be one of: {"first", "next", "last"}')
        
        else:
            # Check episode_start_datetime
            if episode_start_datetime is None:
                # Choose random datetime
                rm = rng.random() if (rng is not None) else np.random.random()
                rt = rng.random() if (rng is not None) else np.random.random()
                # With probability 50% choose datetime in currently loaded month, if any.
                # This way we can reduce average time required to reset environment.
                if (self._df is not None) and (rm < 0.5):
                    file_month = self.months[self._file_index]
                    year = file_month // 100
                    month = file_month - year * 100
                    s, e = arrow.get(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).span('month')
                    s, e = max(s, self.date_from), min(e, self.date_to)
                    s, e = s.timestamp(), (e - episode_min_duration).timestamp()
                else:
                    # s = self._first_datetime.timestamp()
                    # e = (self._last_datetime - episode_min_duration).timestamp()
                    s, e = self.date_from.timestamp(), (self.date_to - episode_min_duration).timestamp()
                episode_start_datetime = s + rt * (e - s)
                episode_start_datetime = arrow.get(episode_start_datetime, tzinfo=timezone.utc).datetime
            elif isinstance(episode_start_datetime, datetime):
                pass
            elif isinstance(episode_start_datetime, arrow.Arrow):
                episode_start_datetime = episode_start_datetime.datetime
            else:
                raise ValueError('Invalid episode_start_datetime value')
            if ((self.date_from is not None) and (episode_start_datetime < self.date_from)) or \
               ((self.date_to is not None) and (episode_start_datetime > self.date_to)):
                raise ValueError('episode_start_datetime is outside range!')

            # Choose file of a specified month
            file_month = episode_start_datetime.year * 100 + episode_start_datetime.month
            assert file_month in self.files
            file_index = self.months.index(file_month)
            file_name = self.files[file_month]
        
        # Load appropriate month of archive trades, unless it is already loaded
        if (self._file_index != file_index) or not isinstance(self._df, pd.DataFrame):
            self.close()
            self._file_index = file_index
            self._df = self.load_month_archive(os.path.join(self.data_dir, file_name))
        
        # Get starting kline index
        if seek is not None:
            # Start from the first record
            record_index = 0
        else:
            # Seek to the trade right before the episode start datetime
            record_index = self._df.time_start.searchsorted(episode_start_datetime, side='left')
            
        # Get actual episode start datetime as it may be later than demanded datetime
        self._episode_start_datetime = self._df.time_start.iloc[record_index].to_pydatetime()
        self._record_index = record_index

        return self._episode_start_datetime
    
    def __next__(self) -> NamedTuple:
        while len(self._trades) <= 0:
            assert (self._df is not None)
            # Read next kline record (i.e. - "enhanced" candle)
            if self._record_index >= len(self._df):
                # Open next file
                self.reset(seek='next')
            record = self._df.iloc[self._record_index]
            kline = Kline(
                time_start=record.time_start,
                time_end=record.time_start + timedelta(seconds=self.duration),
                open=record.open,
                high=record.high,
                low=record.low,
                close=record.close,
                volume=record.volume,
                money=record.money,
                buy_volume=record.buy_volume,
                buy_money=record.buy_money
            )
            self._trades.extend(self.simulate_trades_from_kline(kline))
            self._record_index += 1
        return self._trades.popleft()
    
    def close(self):
        self._df = None
        self._file_index = None
        self._record_index = None
        self._trades.clear()
        self._episode_start_datetime = None

    @property
    def name(self) -> str:
        return self.symbol + '@BinanceKlines'

    @property
    def session_start_datetime(self) -> Union[datetime, None]:
        return self._first_datetime.datetime

    @property
    def session_end_datetime(self) -> Union[datetime, None]:
        return self._last_datetime.datetime

    @property
    def episode_start_datetime(self) -> Union[datetime, None]:
        return self._episode_start_datetime
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Clear current episode information, if any
        state['_df'] = None
        state['_file_index'] = None
        state['_record_index'] = None
        state['_trades'] = deque()
        state['_episode_start_datetime'] = None
        return state
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'data_dir={self.data_dir}, symbol={self.symbol}, spread={self.spread}, '
            f'date_from={self.date_from}, date_to={self.date_to})'
        )
