import math
import numpy as np
from typing import Sequence, MutableSequence, Tuple, Any, Union
from collections import OrderedDict, namedtuple
from numbers import Real
from .frame import Frame

Event = namedtuple('Event', 'frame sign level', defaults=(None, 0, None))


class Labels:
    
    @staticmethod
    def apply_gaussian_filter(frames: Sequence[Frame], radius: int, source: str = 'vwap') -> str:
        kernel = Labels._get_gaussian_kernel(radius)
        N = len(frames)
        R = radius
        P = 2 * R + 1
        dest = f'gauss_{radius}_{source}'
        for i, frame in enumerate(frames):
            value = 0.0
            for k in range(P):
                if i - R + k < 0:
                    v = getattr(frames[0], source)
                elif i - R + k >= N:
                    v = getattr(frames[-1], source)
                else:
                    v = getattr(frames[i - R + k], source)
                value += kernel[k] * v
            setattr(frame, dest, value)
        return dest
                
    _gaussian_kernels = {}

    @staticmethod
    def _get_gaussian_kernel(radius: int):
        # kernel will have a middle cell, and radius cells on each side
        assert isinstance(radius, int) and (radius > 0)
        # Check if we already have kernel in cache
        if radius in Labels._gaussian_kernels:
            return Labels._gaussian_kernels[radius]
        # Apparently this is all you need to get a good approximation
        sigma = radius / 2
        # Normalization constant makes sure total of matrix is 1
        norm = 1.0 / (math.sqrt(2 * math.pi) * sigma)
        # the bit you divide x^2 by in the exponential
        coeff = 2 * sigma * sigma
        total = 0.0
        kernel = []
        for x in range(-radius, radius + 1):
            g = norm * math.exp(-x * x / coeff)
            kernel.append(g)
            total += g
        
        # Rescale values to get a total of 1, because of discretization error
        for i in range(len(kernel)):
            kernel[i] = kernel[i] / total
        
        # Convert kernel to immutable tuple and save it to cache
        kernel = tuple(kernel)
        Labels._gaussian_kernels[radius] = kernel
        
        return kernel
    
    @staticmethod
    def calculate_standard_deviation(frames: Sequence[Frame],
                                     expected: str,
                                     variable: Union[str, Sequence[str]] = ('low', 'high', 'close')) -> float:
        dy = []
        for frame in frames:
            expected_value = getattr(frame, expected)
            if isinstance(variable, str):
                variable_value = getattr(frame, variable)
                dy.append(variable_value - expected_value)
            else:
                for field in variable:
                    variable_value = getattr(frame, field)
                    dy.append(variable_value - expected_value)
        # Calculate standard deviation
        dy = np.array(dy)
        return np.sqrt(np.mean(dy**2))
    
    @staticmethod
    def _calculate_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dx = (x - x_mean)
        dy = (y - y_mean)
        k = np.dot(dx, dy) / np.dot(dx, dx)
        b = y_mean - k * x_mean
        return k, b

    @staticmethod
    def calculate_linear_regression(frames: Sequence[Frame], x_name: str, y_name: str) -> Tuple[float, float]:
        x, y = [], []
        for frame in frames:
            x.append(getattr(frame, x_name))
            y.append(getattr(frame, y_name))
        x, y = np.array(x), np.array(y)
        return Labels._calculate_linear_regression(x, y)
    
    @staticmethod
    def fractal_extremums_filter(frames: Sequence[Frame],
                                 radius: int = 2,
                                 source: Union[str, Tuple[str, str]] = ('low', 'high')) -> MutableSequence[Event]:
        assert isinstance(radius, int) and (radius > 0)
        R = radius
        N = len(frames)
        if isinstance(source, str):
            source_low = source_high = source
            # dest = f'fractal_{radius}_{source}'
        elif isinstance(source, Tuple):
            source_low, source_high = source
            # dest = f'fractal_{radius}_{source_low}_{source_high}'
        else:
            raise ValueError('Invalid source')
    
        extremums = []
        for i, central_frame in enumerate(frames):
            # Find highest and lowest values for last period
            highest, lowest = None, None
            for frame in frames[max(0, i - R):min(N - 1, i + R + 1)]:
                low = getattr(frame, source_low)
                high = getattr(frame, source_high)
                if (lowest is None) or (lowest > low):
                    lowest = low
                if (highest is None) or (highest < high):
                    highest = high
            # Read high and low of central point
            Le = getattr(central_frame, source_low)
            He = getattr(central_frame, source_high)
            # Check for extremums
            if He >= highest:
                # We have got new candidate for upper extremum
                # We should compare it with previous extremums
                discard = False
                for j in range(len(extremums) - 1, -1, -1):
                    e = extremums[j]
                    if e.sign < 0:
                        if e.level > highest:
                            # Previous extremum had a higher price, so we discard current candidate
                            discard = True
                            break
                        else:
                            # Previous extremum had a lower price, so we discard it
                            del extremums[j]
                    elif e.sign > 0:
                        # We stop comparison when we meet first lower extremum
                        break
                if not discard:
                    extremums.append(Event(frame=central_frame, sign=-1, level=highest))
            elif Le <= lowest:
                # We have got new candidate for lower extremum
                # We should compare it with previous extremums
                discard = False
                for j in range(len(extremums) - 1, -1, -1):
                    e = extremums[j]
                    if e.sign > 0:
                        if e.level < lowest:
                            # Previous extremum had a lower price, so we discard current candidate
                            discard = True
                            break
                        else:
                            # Previous extremum had a higher price, so we discard it
                            del extremums[j]
                    elif e.sign < 0:
                        # We stop comparison when we meet first higher extremum
                        break
                if not discard:
                    extremums.append(Event(frame=central_frame, sign=1, level=lowest))
            else:
                pass
        return extremums

    @staticmethod
    def cumulative_sum_filter(frames: Sequence[Frame],
                              threshold: Real,
                              source: str = 'close',
                              prediction: Union[str, None] = None) -> MutableSequence[Event]:
        assert isinstance(threshold, Real) and (threshold > 0.0)
        # Let the prediction for the first value be equal to the value itself
        expected = getattr(frames[0], source)
        # These are cumulative sum counters
        pos, neg = 0.0, 0.0
        # Cycle through frames and collect breakthrough events
        events = []
        for i, frame in enumerate(frames):
            # Read source value and prediction for next value
            value = getattr(frame, source)
            # Update cumulative counters
            pos = max(0, pos + value - expected)
            neg = min(0, neg + value - expected)
            # Check for threshold breakthrough
            if pos >= threshold:
                # We probably go up
                events.append(Event(frame=frame, sign=1, level=value))
                pos = 0
            if neg <= -threshold:
                # We probably go down
                events.append(Event(frame=frame, sign=-1, level=value))
                neg = 0
            # Get prediction for next value or use current value as prediction for next one
            expected = getattr(frame, prediction) if (prediction is not None) else value
        return events
    
    @staticmethod
    def apply_triple_barrier(frames: Sequence[Frame],
                             events: Sequence[Event],
                             barrier: Union[float, Tuple[float, float]]) -> MutableSequence[Event]:
        lower_k, upper_k = barrier if isinstance(barrier, Tuple) else (barrier, barrier)
        out = []
        for j, event in enumerate(events):
            # Get starting and ending time of period of interest
            t0 = event.end
            t1 = events[j + 1].end if (j + 1 < len(events)) else frames[-1].time_end
            # Calculate upper and lower barriers
            base_price = frames[event.i].close
            upper_barrier = base_price + upper_k
            lower_barrier = base_price - lower_k
            # Find out indexes of frames where price hits upper and lower barriers
            upper_hit_i, lower_hit_i = None, None
            for i in range(i0, i1):
                frame = frames[i]
                if (upper_hit_i is None) and (frame.high > upper_barrier):
                    upper_hit_i = i
                if (lower_hit_i is None) and (frame.low < lower_barrier):
                    lower_hit_i = i
                if (upper_hit_i is not None) and (lower_hit_i is not None):
                    break
