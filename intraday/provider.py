from typing import Union, Optional, Type, NamedTuple
from numbers import Real
from collections import namedtuple
import numpy as np
from datetime import timedelta, datetime


Trade = namedtuple('Trade', 'datetime operation amount price')
"""Information about a simple trade"""


TradeOI = namedtuple('TradeOI', 'datetime operation amount price open_interest')
"""Information about a trade with additional field open_interest, used in futures markets"""


Candle = namedtuple('Candle', 'time_start time_end open high low close volume')
"""Aggregated information of price movement and volume traded over some period of time"""


Kline = namedtuple('Kline', 'time_start time_end open high low close volume money buy_volume buy_money')
"""Aggregated information of price movement and volume traded over some period of time,
including additional information about at which prices the most buys and sells were concentrated"""


class Provider(object):
    def __init__(self, **kwargs):
        pass

    def reset(self,
              episode_min_duration: Union[None, Real, timedelta] = None,
              rng: Optional[np.random.RandomState] = None,
              **kwargs
              ) -> datetime:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        raise NotImplementedError()
    
    @property
    def kind(self) -> Type[NamedTuple]:
        """Returns type of objects which this provider outputs. One of of namedtuple: Trade, TradeOI, Candle, Kline"""
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def session_start_datetime(self) -> Union[datetime, None]:
        raise NotImplementedError()

    @property
    def session_end_datetime(self) -> Union[datetime, None]:
        raise NotImplementedError()

    @property
    def episode_start_datetime(self) -> Union[datetime, None]:
        raise NotImplementedError()
