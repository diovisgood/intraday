from typing import Sequence, Literal, List
from collections import OrderedDict
from abc import ABC, abstractmethod

import gym.spaces

from .frame import Frame
from .processor import Trade


class Feature(ABC):
    """
    Base class for all feature processors
    
    Attributes
    ----------
    write_to : {'state', 'frame', 'both'}
        Specifies where you should put your computed values into.
        state - put values into state's OrderedDict
        frame - put values into latest (newest) frame
        both - put values into both state and frame
    write_to_frame : bool
        Derived from `write_to` value. If True you should write to frame.
    write_to_state : bool
        Derived from `write_to` value. If True you should write to state.
    period : int
        How many latest frames you need to compute you feature's values?
        Note: not all features require previous frames. Specify 1 or 0 or None in this case.
    names : List[str]
        A list of all names of values this feature instance produces.
        The order of names should match the order of their appearance in state (OrderedDict)
    spaces : OrderedDict[str, gym.Space]
        An ordered dict of all names of values this feature instance outputs to state
        and their Space definitions.
        The order of names should match the order of their appearance in state (OrderedDict)
        Note: this dict must be empty in case when `write_to='frame'`.
        As it contains only names which are actually being written into state.
    
    See Also
    --------
    intraday.Frame
    gym.spaces.Space
    gym.spaces.Box
    """
    def __init__(self, write_to: Literal['state', 'frame', 'both'] = 'state', **kwargs):
        assert isinstance(write_to, str) and (write_to in {'state', 'frame', 'both'})
        self.write_to = write_to
        self.write_to_frame: bool = write_to in {'frame', 'both'}
        self.write_to_state: bool = write_to in {'state', 'both'}
        self.period: int = kwargs['period'] if ('period' in kwargs) else None
        self.names: List[str] = []
        self.spaces: OrderedDict[str, gym.Space] = OrderedDict()
    
    def reset(self):
        """
        Perform some clean up and reset internal state between episodes
        
        Notes
        -----
        This method is automatically invoked by Environment instance upon its reset.
        
        See Also
        --------
        intraday.MultiAgentEnv.reset
        """
        pass

    @abstractmethod
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        """
        Compute and write your features values when new frame arrives
        
        Notes
        -----
        Important: this method is invoked only ONCE for each new frame.
        
        Parameters
        ----------
        frames : Sequence[Frame]
            Contains a list of latest frames.
            frames[-1] is the newest frame.
            And frames[-2] is the frame before the newest one.
        state : OrderedDict
            The common state dict to write feature values to.
            This dict collects values from all the features of features pipeline,
            and then passes those values to an agent for it to make next action.
            Write your values directly into this dict, for example:
            >>> state['ema_price'] = self.ema_price
            Ensure that you only use names as they are specified in `self.names` and `self.spaces`
            
        """
        raise NotImplementedError()
    
    def __str__(self):
        return self.__repr__()


class TradesFeature(Feature):
    """
    Base class for all feature processors which extract values from stream of trades instead of frames
    
    Attributes
    ----------
    trades_period : int
        How many latest trades you need to compute you feature's values?
        Note: not all features require previous trades. Specify 1 or 0 or None in this case.
    
    See Also
    --------
    intraday.Feature
    intraday.Trade
    """
    def __init__(self, write_to: Literal['state', 'frame', 'both'] = 'state', **kwargs):
        super().__init__(write_to=write_to, **kwargs)
        self.trades_period = kwargs['trades_period'] if ('trades_period' in kwargs) else None
    
    @abstractmethod
    def update(self, trades: Sequence[Trade]):
        """
        Compute your feature values when new trade arrives

        Notes
        -----
        Important: this method is invoked only ONCE for each new trade.

        Parameters
        ----------
        trades : Sequence[Trade]
            Contains a list of latest trades.
            trades[-1] is the newest trade.
            And trades[-2] is the trade before the newest one.
        """
        raise NotImplementedError()
