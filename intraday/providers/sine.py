import math
from typing import NamedTuple, Union, Optional, Tuple
from numbers import Real
import arrow
import numpy as np
from datetime import datetime, date, timedelta, timezone

from intraday.provider import Provider, Trade


class SineProvider(Provider):
    """
    Generates fake stream of trades to move price in a sinusoidal form

    Notes
    -----
    If you want to train a trading bot, this sine generator
    would be a good test for it.
    
    If your algorithm fails to learn to make profit even on such simple data,
    it will never find any profit in a real market data.
    
    Also, please keep in mind that with default values (mean=0, amplitude=100)
    price will take both positive and negative values.
    It may be useful to test your trading bot and your features for robustness.
    
    Parameters
    ----------
    mean : float
        Mean value of a sinusoid. Default: 0.0.
    amplitude: float
        Amplitude of a sinusoid. Default: 100.0
    frequency: Optional[Union[float, Tuple[float, float]]]
        Frequency of a sinusoid in Hertz, i.e. (1 / seconds).
        Or a range specified by two values, to take a random frequency at each episode.
        You must specify either `frequency` or `period`.
        Default: None
    period: Optional[Union[timedelta, Tuple[timedelta, timedelta]]]
        Period of a sinusoid in seconds.
        Or a range specified by two values, to take a random period at each episode.
        You must specify either `frequency` or `period`.
        Default: None
    SNRdb: float
        Signal to noise ratio in Db.
        The less this value - the more noise is added to sinusoid.
        Default: 15.0
    date_from: Optional[Union[date, datetime, arrow.Arrow]]
        Specify starting date for simulated trades.
        If None - uses the date a year ago from current date.
        Default: None
    date_to: Optional[Union[date, datetime, arrow.Arrow]]
        Specify ending date for simulated trades.
        If None - uses the current date.
        Default: None
    """
    def __init__(self,
                 mean: float = 0.0,
                 amplitude: float = 100.0,
                 frequency: Optional[Union[float, Tuple[float, float]]] = None,
                 period: Optional[Union[timedelta, Tuple[timedelta, timedelta]]] = None,
                 SNRdb: float = 15.0,
                 date_from: Optional[Union[date, datetime, arrow.Arrow]] = None,
                 date_to: Optional[Union[date, datetime, arrow.Arrow]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.mean = mean
        self.amplitude = amplitude
        self.SNRdb = SNRdb
        self.noise_amplitude = math.sqrt(amplitude ** 2 / (math.pow(10, SNRdb/10)))
        if isinstance(frequency, float):
            self.freq1 = self.freq2 = frequency
        elif isinstance(frequency, Tuple):
            assert (len(frequency) == 2) and isinstance(frequency[0], Real) and isinstance(frequency[1], Real)
            self.freq1, self.freq2 = min(*frequency), max(*frequency)
        elif isinstance(period, timedelta):
            self.freq1 = self.freq2 = 1 / period.total_seconds()
        elif isinstance(period, Tuple):
            assert (len(period) == 2) and isinstance(period[0], timedelta) and isinstance(period[1], timedelta)
            freq1, freq2 = (1 / period[0].total_seconds()), (1 / period[1].total_seconds())
            self.freq1, self.freq2 = min(freq1, freq2), max(freq1, freq2)
        else:
            raise ValueError('Specify either frequency or period!')
            
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
        
        # Prepare episode variables
        self._freq: Optional[float] = None
        self._datetime: Optional[datetime] = None
        self._last_price: Optional[float] = None
        self._episode_start_datetime: Optional[Union[datetime, arrow.Arrow]] = None
    
    def reset(self,
              episode_start_datetime: Union[None, arrow.Arrow, datetime] = None,
              episode_min_duration: Union[None, Real, timedelta] = None,
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

        # Check episode_start_datetime
        if episode_start_datetime is None:
            # Choose random datetime
            rt = rng.random() if (rng is not None) else np.random.random()
            s = self.date_from.timestamp()
            e = (self.date_to - episode_min_duration).timestamp()
            episode_start_datetime = s + rt * (e - s)
            episode_start_datetime = arrow.get(episode_start_datetime, tzinfo=timezone.utc).datetime
        elif isinstance(episode_start_datetime, datetime):
            pass
        elif isinstance(episode_start_datetime, arrow.Arrow):
            episode_start_datetime = episode_start_datetime.datetime
        else:
            raise ValueError('Invalid episode_start_datetime value')
        self._episode_start_datetime = episode_start_datetime

        # Generate random frequency
        r = rng.random() if (rng is not None) else np.random.random()
        self._freq = self.freq1 + r * (self.freq2 - self.freq1)

        self._datetime = episode_start_datetime
        self._last_price = 0
        
        return self._episode_start_datetime
    
    def __next__(self) -> NamedTuple:
        self._datetime += timedelta(seconds=5 * np.random.random())
        t = (self._datetime - self._episode_start_datetime).total_seconds()
        sine = math.sin(2 * math.pi * self._freq * t)
        noise = np.random.randn()
        price = self.amplitude * sine + self.noise_amplitude * noise
        operation = 'S' if (price < self._last_price) else 'B'
        amount = np.random.randint(10) + 1
        self._last_price = price
        
        # Return next trade
        return Trade(
            datetime=self._datetime,
            operation=operation,
            amount=amount,
            price=price,
        )
    
    def close(self):
        self._freq = None
        self._datetime = None
        self._last_price = None
        self._episode_start_datetime = None

    @property
    def name(self) -> str:
        return 'Sine'

    @property
    def session_start_datetime(self) -> Union[datetime, None]:
        return self.date_from.datetime

    @property
    def session_end_datetime(self) -> Union[datetime, None]:
        return self.date_to.datetime

    @property
    def episode_start_datetime(self) -> Union[datetime, None]:
        return self._episode_start_datetime
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'frequency=({self.freq1}, {self.freq1}), '
            f'mean={self.mean}, amplitude={self.amplitude}, SNRdb={self.SNRdb}, '
            f'date_from={self.date_from}, date_to={self.date_to})'
        )
