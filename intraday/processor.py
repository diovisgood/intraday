from __future__ import annotations
from typing import Tuple, Sequence, Union, Literal, Optional
import math
from numbers import Real
import arrow
import datetime as dt
from .provider import Trade
from .frame import Frame


class Processor(object):
    """
    Base abstract class for IntervalProcessor, ImbalanceProcessor and RunProcessor
    """
    def __init__(self, **kwargs):
        pass
    
    def reset(self):
        raise NotImplementedError()

    def process(self, trades: Sequence[Trade]) -> Optional[Frame]:
        raise NotImplementedError()
    
    def finish(self) -> Optional[Frame]:
        raise NotImplementedError()
    
    @property
    def name(self):
        raise NotImplementedError()


class IntervalProcessor(Processor):
    
    FramingMethods = [
        'time',
        'tick',
        'volume',
        'money',
    ]
    
    def __init__(self,
                 method: Literal['time', 'tick', 'volume', 'money'],
                 interval: Real,
                 duration: Tuple[Real, Real] = (1, 4*60*60),
                 **kwargs):
        super().__init__(**kwargs)
        # Method of how to group the flow of records into frames
        # Frame initial_threshold which depends on selected method
        assert isinstance(method, str) and (method in IntervalProcessor.FramingMethods)
        assert isinstance(interval, int) and (interval > 0)
        assert (isinstance(duration, tuple) and (len(duration) == 2) and
                isinstance(duration[0], Real) and isinstance(duration[1], Real))
        
        # Save parameters
        self.method = method
        self.interval = interval
        self.duration = duration
        
        self.frame = None
        
    @property
    def name(self):
        return f'{self.method}@{self.interval}'

    def reset(self):
        self.frame: Optional[Frame] = None

    def process(self, trades: Sequence[Trade]) -> Optional[Frame]:
        result = None
        trade = trades[-1]
        frame = self.frame

        if (frame is None) or ((self.method == 'time') and (trade.datetime > frame.time_end)):
            # Init new frame
            result = frame.finalize() if (frame is not None) else None
            frame = Frame(time_start=trade.datetime)
            self.frame = frame
            if self.method == 'time':
                # Round time_start to specified initial_threshold and also get time_end
                time_start, time_end = _get_time_span(trade.datetime, self.interval)
                frame.time_start = time_start
                frame.time_end = time_end

        # Update frame
        frame.update(trades)
        
        # Check for the end of frame
        if (
            (frame.duration >= self.duration[0]) and (
                ((self.method == 'tick') and (frame.ticks >= self.interval)) or
                ((self.method == 'volume') and (frame.volume >= self.interval)) or
                ((self.method == 'money') and (frame.money >= self.interval))
            ) or
            (frame.duration >= self.duration[1])
        ):
            # Reset current frame
            result = frame.finalize() if (frame is not None) else None
            self.frame = None
            
        return result
    
    def finish(self) -> Optional[Frame]:
        result = self.frame.finalize()
        self.frame = None
        return result

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'method={self.method}, interval={self.interval}, duration={self.duration})'
        )


class ImbalanceProcessor(Processor):
    
    FramingMethods = {
        'ti': 'tick-imbalance',
        'vi': 'volume-imbalance',
        'mi': 'money-imbalance',
    }
    
    def __init__(self,
                 method: str,
                 initial_threshold: Real,
                 ema_period_frames: int = 200,
                 ema_period_trades: int = 1000,
                 duration: Tuple[Real, Real] = (1, 30*60),
                 **kwargs):
        super().__init__(**kwargs)
        # Method of how to group the flow of records into frames
        # Frame initial_threshold which depends on selected method
        assert isinstance(method, str) and (method in ImbalanceProcessor.FramingMethods)
        assert isinstance(initial_threshold, int) and (initial_threshold > 0)
        assert isinstance(ema_period_frames, int) and (ema_period_frames > 0)
        assert isinstance(ema_period_trades, int) and (ema_period_trades > 0)
        assert (isinstance(duration, tuple) and (len(duration) == 2) and
                isinstance(duration[0], Real) and isinstance(duration[1], Real))

        # Save parameters
        self.method = method
        self.initial_threshold = initial_threshold
        self.ema_period_frames = ema_period_frames
        self.ema_period_trades = ema_period_trades
        self.duration = duration

        # Initialize episode variables
        self.frame = None
        self.n_trades = 0
        self.n_frames = 0
        
        # Average values
        self.avg_trade_tick = None
        self.avg_trade_buy_amount = None
        self.avg_trade_buy_money = None
        self.avg_trade_sell_amount = None
        self.avg_trade_sell_money = None
        self.avg_frame_ticks = None
        # These will hold threshold values
        self.threshold_imbalance_ticks = None
        self.threshold_imbalance_volume = None
        self.threshold_imbalance_money = None

    @property
    def name(self):
        return self.method

    def reset(self):
        self.frame = None
        self.n_trades = 0
        self.n_frames = 0
        self.avg_trade_tick = None
        self.avg_trade_buy_amount = None
        self.avg_trade_buy_money = None
        self.avg_trade_sell_amount = None
        self.avg_trade_sell_money = None
        self.avg_frame_ticks = None
        self.threshold_imbalance_ticks = None
        self.threshold_imbalance_volume = None
        self.threshold_imbalance_money = None

    def process(self, trades: Sequence[Trade]) -> Optional[Frame]:
        result = None
        trade = trades[-1]
        frame = self.frame

        if frame is None:
            frame = Frame(time_start=trade.datetime)
            self.frame = frame
            
        # Update frame
        frame.update(trades)
        
        # Update average values of a trade
        tick = (1 if (trade.operation == 'B') else -1)
        price = trade.price
        amount = trade.amount
        money = amount * price
        self._update_average_value('avg_trade_tick', tick, self.n_trades, self.ema_period_trades)
        if trade.operation == 'B':
            self._update_average_value('avg_trade_buy_amount', amount, self.n_trades, self.ema_period_trades)
            self._update_average_value('avg_trade_buy_money', money, self.n_trades, self.ema_period_trades)
        else:
            self._update_average_value('avg_trade_sell_amount', amount, self.n_trades, self.ema_period_trades)
            self._update_average_value('avg_trade_sell_money', money, self.n_trades, self.ema_period_trades)
            
        # Check for the end of frame
        if (
            (frame.duration >= self.duration[0]) and (
                ((self.n_frames <= 0) and (
                    ((self.method == 'ti') and (frame.ticks >= self.initial_threshold)) or
                    ((self.method == 'ti') and (frame.volume >= self.initial_threshold)) or
                    ((self.method == 'ti') and (frame.money >= self.initial_threshold))
                )) or
                ((self.n_frames > 0) and (
                    ((self.method == 'ti') and (abs(frame.imbalance_ticks) >= self.threshold_imbalance_ticks)) or
                    ((self.method == 'vi') and (abs(frame.imbalance_volume) >= self.threshold_imbalance_volume)) or
                    ((self.method == 'mi') and (abs(frame.imbalance_money) >= self.threshold_imbalance_money))
                ))
            ) or
            (frame.duration >= self.duration[1])
        ):
            # Reset current frame
            result = frame.finalize()
            self._process_frame(frame)
            self.frame = Frame()

        # Update number of processed trades
        self.n_trades += 1

        return result

    def finish(self) -> Optional[Frame]:
        if self.frame is None:
            return None
        result = self.frame.finalize()
        if result is not None:
            self._process_frame(result)
            self.frame = None
        return result

    def _process_frame(self, frame: Frame):
        # Update average number of ticks in frame
        self._update_average_value('avg_frame_ticks', frame.ticks, self.n_frames, self.ema_period_frames)
        
        # Calculate buy and sell probability
        Pbuy = (self.avg_trade_tick + 1.0) / 2.0
        Psell = (1.0 - Pbuy)

        # Calculate imbalance thresholds for the next frame
        self.threshold_imbalance_ticks = self.avg_frame_ticks * abs(self.avg_trade_tick or 0)
        self.threshold_imbalance_volume = (self.avg_frame_ticks *
            abs(Pbuy * (self.avg_trade_buy_amount or 0) - Psell * (self.avg_trade_sell_amount or 0)))
        self.threshold_imbalance_money = (self.avg_frame_ticks *
            abs(Pbuy * (self.avg_trade_buy_money or 0) - Psell * (self.avg_trade_sell_money or 0)))
        
        # Update number of processed frames
        self.n_frames += 1

    def _update_average_value(self, name: str, new_value, n_iter: int, period: int):
        if new_value is None:
            return
        old_average = getattr(self, name)
        if old_average is None:
            setattr(self, name, new_value)
        elif n_iter <= period:
            setattr(self, name, float(old_average * n_iter + new_value) / float(n_iter + 1))
        else:
            ema_factor = 2 / (period + 1)
            setattr(self, name, (old_average * (1 - ema_factor) + new_value * ema_factor))
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'method={self.method}, initial_threshold={self.initial_threshold}, '
            f'ema_period_frames={self.ema_period_frames}, ema_period_trades={self.ema_period_trades},'
            f'duration={self.duration})'
        )


class RunProcessor(Processor):
    
    FramingMethods = {
        'tr': 'tick-runs',
        'vr': 'volume-runs',
        'mr': 'money-runs',
    }
    
    def __init__(self,
                 method: str,
                 initial_threshold: Real,
                 ema_period_frames: int = 200,
                 ema_period_trades: int = 1000,
                 duration: Tuple[Real, Real] = (1, 30*60),
                 **kwargs):
        super().__init__(**kwargs)
        # Method of how to group the flow of records into frames
        # Frame initial_threshold which depends on selected method
        assert isinstance(method, str) and (method in RunProcessor.FramingMethods)
        assert isinstance(initial_threshold, int) and (initial_threshold > 0)
        assert isinstance(ema_period_frames, int) and (ema_period_frames > 0)
        assert isinstance(ema_period_trades, int) and (ema_period_trades > 0)
        assert (isinstance(duration, tuple) and (len(duration) == 2) and
                isinstance(duration[0], Real) and isinstance(duration[1], Real))

        # Save parameters
        self.method = method
        self.initial_threshold = initial_threshold
        self.ema_period_frames = ema_period_frames
        self.ema_period_trades = ema_period_trades
        self.duration = duration
        
        # Initialize episode variables
        self.frame = None
        self.n_trades = 0
        self.n_frames = 0
        
        # Average values
        self.avg_trade_tick = None
        self.avg_trade_buy_amount = None
        self.avg_trade_buy_money = None
        self.avg_trade_sell_amount = None
        self.avg_trade_sell_money = None
        self.avg_frame_ticks = None
        
        # These will hold threshold values
        self.threshold_run_ticks = None
        self.threshold_run_volume = None
        self.threshold_run_money = None

    @property
    def name(self):
        return self.method

    def reset(self):
        self.frame = None
        self.n_trades = 0
        self.n_frames = 0
        self.avg_trade_tick = None
        self.avg_trade_buy_amount = None
        self.avg_trade_buy_money = None
        self.avg_trade_sell_amount = None
        self.avg_trade_sell_money = None
        self.avg_frame_ticks = None
        self.threshold_run_ticks = None
        self.threshold_run_volume = None
        self.threshold_run_money = None
    
    def process(self, trades: Sequence[Trade]) -> (Frame, None):
        result = None
        trade = trades[-1]
        frame = self.frame

        if frame is None:
            frame = Frame(time_start=trade.datetime)
            self.frame = frame
            
        # Update frame
        frame.update(trades)
        
        # Update average values of a trade
        tick = (1 if (trade.operation == 'B') else -1)
        price = trade.price
        amount = trade.amount
        money = amount * price
        self._update_average_value('avg_trade_tick', tick, self.n_trades, self.ema_period_trades)
        if trade.operation == 'B':
            self._update_average_value('avg_trade_buy_amount', amount, self.n_trades, self.ema_period_trades)
            self._update_average_value('avg_trade_buy_money', money, self.n_trades, self.ema_period_trades)
        else:
            self._update_average_value('avg_trade_sell_amount', amount, self.n_trades, self.ema_period_trades)
            self._update_average_value('avg_trade_sell_money', money, self.n_trades, self.ema_period_trades)
        
        # Check for the end of frame
        if (
            ((frame.duration >= self.duration[0]) and (
                ((self.n_frames <= 0) and (
                    ((self.method == 'tr') and (max(frame.buy_ticks, frame.sell_ticks) >= self.initial_threshold)) or
                    ((self.method == 'vr') and (max(frame.buy_volume, frame.sell_volume) >= self.initial_threshold)) or
                    ((self.method == 'mr') and (max(frame.buy_money, frame.sell_money) >= self.initial_threshold))
                )) or
                ((self.n_frames > 0) and (
                    ((self.method == 'tr') and (max(frame.buy_ticks, frame.sell_ticks) >= self.threshold_run_ticks)) or
                    ((self.method == 'vr') and (
                        max(frame.buy_volume, frame.sell_volume) >= self.threshold_run_volume)) or
                    ((self.method == 'mr') and (max(frame.buy_money, frame.sell_money) >= self.threshold_run_money))
                ))
            )
        ) or
            (frame.duration >= self.duration[1])
        ):
            # Reset current frame
            result = frame.finalize()
            self._process_frame(frame)
            self.frame = Frame()
            
        # Update number of processed trades
        self.n_trades += 1
        
        return result
    
    def finish(self) -> (Frame, None):
        result = self.frame.finalize()
        if result is not None:
            self._process_frame(result)
            self.frame = None
        return result

    def _process_frame(self, frame: Frame):
        # Update average values
        self._update_average_value('avg_frame_ticks', frame.ticks, self.n_frames, self.ema_period_frames)
        
        # Calculate buy and sell probability
        Pbuy = (self.avg_trade_tick + 1.0) / 2.0
        Psell = (1.0 - Pbuy)
        
        # Calculate run thresholds for the next frame
        self.threshold_run_ticks = self.avg_frame_ticks * max(Pbuy, Psell)
        self.threshold_run_volume = (self.avg_frame_ticks *
                                     max(Pbuy * (self.avg_trade_buy_amount or 0),
                                         Psell * (self.avg_trade_sell_amount or 0)))
        self.threshold_run_money = (self.avg_frame_ticks *
                                    max(Pbuy * (self.avg_trade_buy_money or 0),
                                        Psell * (self.avg_trade_sell_money or 0)))
        
        # Update number of processed frames
        self.n_frames += 1
    
    def _update_average_value(self, name: str, new_value, n_iter: int, period: int):
        if new_value is None:
            return
        old_average = getattr(self, name)
        if old_average is None:
            setattr(self, name, new_value)
        elif n_iter <= period:
            setattr(self, name, float(old_average * n_iter + new_value) / float(n_iter + 1))
        else:
            ema_factor = 2 / (period + 1)
            setattr(self, name, (old_average * (1 - ema_factor) + new_value * ema_factor))
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'method={self.method}, initial_threshold={self.initial_threshold}, '
            f'ema_period_frames={self.ema_period_frames}, ema_period_trades={self.ema_period_trades},'
            f'duration={self.duration})'
        )


def _get_time_span(datetime: Union[dt.datetime, arrow.Arrow], interval: int) -> Tuple[arrow.Arrow, arrow.Arrow]:
    if not isinstance(datetime, arrow.Arrow):
        datetime = arrow.get(datetime)
    # Calculate start of frame
    if interval <= 24*60*60:
        # Get start of day
        s = datetime.floor('day')
    else:
        # Get start of year
        s = datetime.floor('year')
    # Calculate start of frame with respect to start time s and frame initial_threshold
    seconds = math.floor((datetime.timestamp() - s.timestamp()) / interval) * interval
    return (s + dt.timedelta(seconds=seconds)).span('second', interval)
