from typing import Sequence, List, NamedTuple, Tuple, Union, Optional, Literal, BinaryIO
import os
import gzip
import fnmatch
import bisect
from numbers import Real
import arrow
import numpy as np
from datetime import date, datetime, timedelta, timezone
from struct import calcsize, unpack
import re

from intraday.provider import Provider, TradeOI


class MoexArchiveProvider(Provider):
    Stream_Trade_v1_Fmt = '=cBHIIII'
    Stream_Trade_v1_Size = calcsize(Stream_Trade_v1_Fmt)
    Stream_Trade_v2_Fmt = '=cBHIQII'
    Stream_Trade_v2_Size = calcsize(Stream_Trade_v2_Fmt)
    MAX_DWORD = float(4294967295)
    tzinfo = 'Europe/Moscow'
    
    def __init__(self,
                 data_dir: str,
                 files: Union[Sequence[str], str, re.Pattern],
                 date_from: Optional[date] = None,
                 date_to: Optional[date] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        assert isinstance(data_dir, str) and (data_dir > '') and os.path.isdir(data_dir)
        self.data_dir = data_dir
        
        if isinstance(date_to, (datetime, arrow.Arrow)):
            date_to = date_to.date()
        assert (date_to is None) or isinstance(date_to, date)
        self.date_to: Optional[date] = date_to

        if isinstance(date_from, (datetime, arrow.Arrow)):
            date_from = date_from.date()
        assert (date_from is None) or isinstance(date_from, date)
        self.date_from: Optional[date] = date_from
        
        assert (date_to is None) or (date_from is None) or (date_from <= date_to), 'Invalid date_from or date_to'

        # Find all files in Streams_Dir with file_prefixes from Streams
        self.files = {}
        if isinstance(files, (str, re.Pattern)):
            # Find all files in data_dir which satisfy files mask
            for file_name in os.listdir(data_dir):
                if isinstance(files, str) and (not fnmatch.fnmatch(file_name, files)):
                    continue
                elif isinstance(files, re.Pattern) and (not files.match(file_name)):
                    continue
                # Get file date from file_name
                _, file_date = _parse_filename(file_name)
                if file_date is None:
                    continue
                # Filter dates
                if ((date_from is None) or (date_from <= file_date)) and ((date_to is None) or (file_date <= date_to)):
                    self.files[file_date] = file_name
        elif isinstance(files, Sequence):
            # Add all files from list
            for file_name in files:
                # Check file name and date
                assert isinstance(file_name, str)
                _, file_date = _parse_filename(file_name)
                if file_date is None:
                    continue
                # Check file exists
                file_path = os.path.join(data_dir, file_name)
                if (not os.path.exists(file_path)) or (not os.path.isfile(file_path)):
                    continue
                # Add file
                self.files[file_date] = file_name
        else:
            raise ValueError('Invalid `files` argument')
        
        if len(self.files) <= 0:
            raise RuntimeError('Stream files not found for specified file_prefix')
        
        self.dates: List[date] = sorted(list(self.files.keys()))
        
        # Prepare episode variables
        self._file: Optional[BinaryIO] = None
        self._file_version: Optional[int] = None
        self._file_index: Optional[int] = None
        self._symbol: Optional[str] = None
        self._records_count: Optional[int] = None
        self._session_start_datetime: Optional[Union[datetime, arrow.Arrow]] = None
        self._session_end_datetime: Optional[Union[datetime, arrow.Arrow]] = None
        self._episode_start_datetime: Optional[Union[datetime, arrow.Arrow]] = None
    
    def __len__(self):
        return self._records_count
    
    def __getitem__(self, index):
        assert (self._file is not None) and (self._file_version is not None)
        # Read record at item position
        record_size = self.Stream_Trade_v1_Size if (self._file_version == 1) else self.Stream_Trade_v2_Size
        self._file.seek(index * record_size, os.SEEK_SET)
        buffer = self._file.read(record_size)
        if not buffer:
            raise RuntimeError(f'Failed to seek to the trade: {index}')
        # Unpack record
        if self._file_version == 1:
            _, _, _, _, _, dtime_ms, *_ = unpack(self.Stream_Trade_v1_Fmt, buffer)
        elif self._file_version == 2:
            _, _, _, _, _, dtime_ms, *_ = unpack(self.Stream_Trade_v2_Fmt, buffer)
        else:
            raise RuntimeError('Invalid file version')
        assert 0 <= dtime_ms <= 24 * 60 * 60 * 1000
        datetime = self._session_start_datetime + timedelta(seconds=float(dtime_ms) / 1000.0)
        return datetime
    
    def reset(self,
              episode_start_datetime: Union[None, arrow.Arrow, datetime] = None,
              episode_min_duration: Union[None, Real, timedelta] = None,
              seek: Optional[Literal['first', 'next', 'last']] = None,
              rng: Optional[np.random.RandomState] = None
              ) -> datetime:
        # Check episode_start_datetime
        if episode_start_datetime is None:
            pass
        elif isinstance(episode_start_datetime, (arrow.Arrow, datetime)):
            if self.date_from is not None:
                assert episode_start_datetime >= self.date_from
            if self.date_to is not None:
                assert episode_start_datetime <= self.date_to
        else:
            raise ValueError('Invalid episode_start_datetime value')
        
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
        
        # Choose date for next episode
        if seek is not None:
            assert isinstance(seek, str) and (seek in {'first', 'next', 'last'})
            if seek == 'first':
                file_index = 0
                file_date = self.dates[file_index]
                file_name = self.files[file_date]
            elif seek == 'last':
                file_index = len(self.files) - 1
                file_date = self.dates[file_index]
                file_name = self.files[file_date]
            elif seek == 'next':
                if self._file_index is None:
                    raise RuntimeError('Can not seek to next episode without prior episode')
                file_index = self._file_index + 1
                if file_index >= len(self.files):
                    raise StopIteration('No more files')
                file_date = self.dates[file_index]
                file_name = self.files[file_date]
            else:
                raise ValueError('seek can be one of: {"first", "next", "last"}')

            episode_start_datetime = None
        
        elif episode_start_datetime is not None:
            # Get datetime as Arrow object
            episode_start_datetime = arrow.get(episode_start_datetime)
            # Choose file session of a specified date
            # Set episode start time of a specified datetime
            file_date = date(episode_start_datetime.year, episode_start_datetime.month, episode_start_datetime.day)
            assert file_date in self.files
            file_index = self.dates.index(file_date)
            file_name = self.files[file_date]
        
        else:
            # Choose random day from preloaded list
            # Generate random non-uniformly
            file_index = rng.randint(len(self.files)) if (rng is not None) else np.random.randint(len(self.files))
            file_index = max(0, min(len(self.files) - 1, file_index))
            file_date = self.dates[file_index]
            file_name = self.files[file_date]
            episode_start_datetime = None
        
        # Close previous episode if any
        if self._file is not None:
            self.close()
        
        # Keep file index
        self._file_index = file_index
        
        # Keep symbol name
        self._symbol, *_ = _parse_filename(file_name)

        # Prepend file_name with full path to Streams_Dir
        file_path = os.path.join(self.data_dir, file_name)
        
        # Load stream from input file
        if file_name[-3:].lower() == '.gz':
            self._file = gzip.open(file_path, 'rb')
        else:
            self._file = open(file_path, 'rb')
        
        # Estimate file format version
        op = self._file.read(1)
        if op in {b'B', b'S'}:
            self._file_version = 1
        elif op in {b'b', b's'}:
            self._file_version = 2
        else:
            raise RuntimeError(f'Failed to estimate file version: {op}')
        
        # Get file size in bytes and compute number of records in file
        self._file.seek(0, os.SEEK_END)
        file_size = self._file.tell()
        record_size = self.Stream_Trade_v1_Size if (self._file_version == 1) else self.Stream_Trade_v2_Size
        self._records_count = (file_size // record_size)
        
        # Use day start and day end as session start and session end respectively
        self._session_start_datetime, self._session_end_datetime =\
            arrow.get(file_date.year, file_date.month, file_date.day, 0, 0, 0, tzinfo=self.tzinfo).span('day')
        self._session_start_datetime = self._session_start_datetime.datetime
        self._session_end_datetime = self._session_end_datetime.datetime

        # Get starting trade index
        if seek is not None:
            trade_index = 0
        elif episode_start_datetime is not None:
            # Seek to the trade right after the episode start datetime
            trade_index = bisect.bisect_left(self, episode_start_datetime)
            trade_index = max(0, min(self._records_count - 1, trade_index))
        else:
            # Get start and end indices
            start_index, end_index = 0, (self._records_count - 1)
            first_datetime = self.__getitem__(start_index)
            last_datetime = self.__getitem__(end_index)
            # Check if there is enough duration
            if (last_datetime - first_datetime) < episode_min_duration:
                raise RuntimeError(f'Session {file_date} can not satisfy episode_min_duration={episode_min_duration}')
            # Update end index
            if episode_min_duration.total_seconds() > 0:
                end_index = bisect.bisect_left(self, last_datetime - episode_min_duration)
                end_index = min(self._records_count - 1, end_index)
            # Choose random episode start to satisfy episode_min_duration
            trade_index = rng.randint(start_index, end_index + 1) if (rng is not None) else np.random.randint(
                start_index, end_index + 1)
            trade_index = max(0, min(self._records_count - 1, trade_index))
            # Shift to the leftmost trade with the same datetime value
            episode_start_datetime = self.__getitem__(trade_index)
            while (trade_index > 0) and (self.__getitem__(trade_index - 1) == episode_start_datetime):
                trade_index -= 1
        
        # Get actual episode start datetime as it may be later than demanded datetime
        self._episode_start_datetime = self.__getitem__(trade_index)
        if ((self.date_from is not None) and (self._episode_start_datetime < self.date_from)) or \
           ((self.date_to is not None) and (self._episode_start_datetime > self.date_to)):
            raise ValueError('episode_start_datetime is outside range!')

        # Seek to the first record
        self._file.seek(trade_index * record_size, os.SEEK_SET)
        
        return self._episode_start_datetime
    
    def close(self):
        self._file.close()
        self._file = None
        self._file_index = None
        self._file_version = None
        self._symbol = None
        self._records_count = None
        self._session_start_datetime = None
        self._session_end_datetime = None
        self._episode_start_datetime = None
    
    def __next__(self) -> NamedTuple:
        assert (self._file is not None)
        # Read next buffer
        if self._file_version == 1:
            buffer = self._file.read(self.Stream_Trade_v1_Size)
            if not buffer:
                raise StopIteration
            # Unpack next record
            operation, period, amount, price, id, dtime_ms, open_interest = unpack(self.Stream_Trade_v1_Fmt, buffer)
        elif self._file_version == 2:
            buffer = self._file.read(self.Stream_Trade_v2_Size)
            if not buffer:
                raise StopIteration
            # Unpack next record
            operation, period, amount, price, id, dtime_ms, open_interest = unpack(self.Stream_Trade_v2_Fmt, buffer)
        else:
            raise RuntimeError(f'Invalid file format version: {self._file_version}')
        operation = operation.decode().upper()
        assert operation in 'BS'
        assert 0 <= dtime_ms <= 24 * 60 * 60 * 1000
        datetime = self._session_start_datetime + timedelta(seconds=float(dtime_ms) / 1000.0)

        # Return next trade
        return TradeOI(
            datetime=datetime,
            operation=operation,
            amount=amount,
            price=price,
            open_interest=open_interest
        )
    
    @property
    def name(self) -> str:
        if self._symbol is None:
            return ''
        return self._symbol + '@MOEX'
    
    @property
    def session_start_datetime(self) -> Union[datetime, None]:
        return self._session_start_datetime

    @property
    def session_end_datetime(self) -> Union[datetime, None]:
        return self._session_end_datetime

    @property
    def episode_start_datetime(self) -> Union[datetime, None]:
        return self._episode_start_datetime
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Clear current episode information, if any
        state['_file'] = None
        state['_file_index'] = None
        state['_file_version'] = None
        state['_records_count'] = None
        state['_symbol'] = None
        state['_session_start_datetime'] = None
        state['_session_end_datetime'] = None
        state['_episode_start_datetime'] = None
        return state
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'data_dir={self.data_dir}, '
            f'date_from={self.date_from}, date_to={self.date_to})'
        )


def _parse_filename(filename: str) -> Optional[Tuple[str, date]]:
    # Extract date part from filename
    if filename.endswith('.gz'):
        filename = filename[0:-3]
    if filename.endswith('.trades'):
        filename = filename[0:-7]
    str_name = filename[:-9]
    str_date = filename[-8:]
    if not str_date.isdigit():
        return None
    # Calc _session_start_datetime for this exact day
    year = int(str_date[0:4])
    month = int(str_date[4:6])
    day = int(str_date[6:])
    if (1900 <= year <= 2100) and (1 <= month <= 12) and (1 <= day <= 31):
        return str_name, date(year, month, day)
    return None
