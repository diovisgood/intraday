from typing import Sequence, Union
from collections import OrderedDict
import math

from intraday.frame import Frame
from intraday.feature import Feature


class GaussianSmooth(Feature):
    """Gaussian smoothing"""
    
    def __init__(self, radius: int, source: Union[str, Sequence[str]] = 'vwap'):
        """
        Initializes `GaussianSmooth` feature processor

        Parameters
        ----------
        radius : int
            Radius of gaussian kernel to smooth values. Period will be: (2*radius + 1).
        source : str or Sequence[str]
            Names of Frame's attributes to encode.
        """
        super().__init__(write_to='frame', period=2 * radius + 1)
        
        assert isinstance(radius, int) and (radius > 0)
        self.radius = radius
        
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        
        for name in self.source:
            assert isinstance(name, str)
            self.names.append(f'gauss_{radius}_{name}')
        self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        kernel = self._get_gaussian_kernel(self.radius)
        N = len(frames)
        R = self.radius
        P = 2 * R + 1
        # Get central frame index i
        i = N - R - 1
        if i < 0:
            return
        # Process each name in source
        for j, name in enumerate(self.source):
            value = 0.0
            for k in range(P):
                if i - R + k < 0:
                    v = getattr(frames[0], name)
                elif i - R + k >= N:
                    v = getattr(frames[-1], name)
                else:
                    v = getattr(frames[i - R + k], name)
                value += kernel[k] * v
            # Write smoothed value into central frame i
            setattr(frames[i], self.names[j], value)
    
    _gaussian_kernels = {}
    
    @staticmethod
    def _get_gaussian_kernel(radius: int):
        # kernel will have a middle cell, and radius cells on each side
        assert isinstance(radius, int) and (radius > 0)
        # Check if we already have kernel in cache
        if radius in GaussianSmooth._gaussian_kernels:
            return GaussianSmooth._gaussian_kernels[radius]
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
        GaussianSmooth._gaussian_kernels[radius] = kernel
        
        return kernel
    
    def __repr__(self):
        return f'{self.__class__.__name__}(radius={self.radius}, source={self.source})'
