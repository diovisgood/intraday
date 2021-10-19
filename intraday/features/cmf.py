from typing import Sequence, Tuple, Literal
from collections import OrderedDict

import gym.spaces

from intraday.frame import Frame
from intraday.feature import Feature


class CMF(Feature):
    """
    Chaikin Money Flow (CMF)

    Notes
    -----
    As described here: https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf

    CMF is an oscillator that fluctuates between -1 and +1.
    Rarely, if ever, will the indicator reach these extremes.
    Typically, this oscillator fluctuates between -0.50 and +0.50 with 0 as the centerline.
    """
    
    def __init__(self,
                 period: int = 20,
                 source: Tuple[str, str, str, str] = ('low', 'high', 'close', 'volume'),
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `ChaikinMoneyFlow` feature processor

        Parameters
        ----------
        period : int
            Period to compute money flow. A typical values are: 20 or 21.
        source : Tuple[str, str, str, str]
            Names for `low`, `high`, `close` and `volume` of frame's attributes.
        write_to : str {'frame','state','both'}
            Destination of where to put computed values.
        """
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, Tuple) and (len(source) == 4)
        self.source = source
        self.names = [f'cmf_{period}_{self.source[0]}_{self.source[1]}_{self.source[2]}_{self.source[3]}']
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-1.0, 1.0, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        self.money_flow_volumes = []
        self.volumes = []
    
    def reset(self):
        self.money_flow_volumes.clear()
        self.volumes.clear()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        low, high = getattr(frame, self.source[0]), getattr(frame, self.source[1])
        close, volume = getattr(frame, self.source[2]), getattr(frame, self.source[3])
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low) if (high != low) else 0.0
        money_flow_volume = volume * money_flow_multiplier
        self.money_flow_volumes.append(money_flow_volume)
        self.volumes.append(volume)
        if len(self.volumes) > self.period:
            self.money_flow_volumes = self.money_flow_volumes[-self.period:]
            self.volumes = self.volumes[-self.period:]
        result = sum(self.money_flow_volumes) / sum(self.volumes)
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result
