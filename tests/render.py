import time
from typing import List, OrderedDict, Tuple, Callable, Union, Optional
import os
import numpy as np
import pandas as pd

import pyglet

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# import matplotlib as plt
# import mplfinance as mpf
# plt.interactive(True)

# import pyglet
from gym.envs.classic_control import rendering

# import matplotlib.animation as animation
from datetime import timedelta, date
import gym

from intraday.frame import Frame
from intraday.providers.binance import BinanceArchiveProvider
from intraday.processor import IntervalProcessor
from intraday.features import CopyAttributes, FractalExtremums
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from intraday.env import SingleAgentEnv

BINANCE_DATA_DIR = 'D:\\Database\\BINANCE\\'
CPU_LIMIT = max(1, os.cpu_count() // 2)

# Setup logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
    level=logging.DEBUG,
)
log = logging.getLogger('')


def get_env() -> gym.Env:
    provider = BinanceArchiveProvider(
        data_dir=BINANCE_DATA_DIR,
        symbol='BTCUSDT',
        date_from=date(2018,1,1),
        date_to=date(2020,1,1)
    )
    processor = IntervalProcessor(method='time', interval=60)
    features_pipeline = [
        FractalExtremums(
            radius=3,
            source='close',
            threshold=None,
            write_to='frame'
        ),
        CopyAttributes(
            source=(
                'time_start',
                'time_end',
                'duration',
                'open',
                'high',
                'low',
                'close',
                'vwap',
                'volume',
            )
        ),
    ]
    action_scheme = BuySellCloseAction()
    reward_scheme = BalanceReward()
    return SingleAgentEnv(
        provider=provider,
        processor=processor,
        features_pipeline=features_pipeline,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        initial_balance=1000000000,
        warm_up_time=timedelta(minutes=15)
    )


def add_record(df: pd.DataFrame, state: OrderedDict, action: int):
    time = state['time_start'].datetime
    buy = state['low'] if (action == 1) else np.nan
    sell = state['high'] if (action == 2) else np.nan
    stop = state['close'] if (action == 3) else np.nan
    values = [state[k] for k in ('open', 'high', 'low', 'close', 'volume')] + [buy, sell, stop]
    df.loc[time] = values


class Label(rendering.Geom):
    def __init__(self, text: str, pos=(0.0, 0.0)):
        super().__init__()
        self.text = text
        self.pos = pos

    def render1(self):
        label = pyglet.text.Label(
            self.text,
            font_name='Arial',
            font_size=12,
            x=self.pos[0],
            y=self.pos[1],
            anchor_x='center',
            anchor_y='center',
            color=self._color,
        )
        label.x
        label.draw()


def run_episode(env: gym.Env,
                model: Optional[Callable] = None,
                **kwargs
                ) -> Union[None, Tuple[List[OrderedDict], List[Frame]]]:
    # Start new episode
    state = env.reset(episode_min_duration=timedelta(hours=1), episode_max_duration=timedelta(hours=3))
    states = [state]
    frames = []

    # Prepare window
    width, height = 600, 400
    margin = 10
    
    viewer = rendering.Viewer(width, height)
    
    area1 = (margin, viewer.height*1/3 + margin/2, viewer.width - margin, viewer.height - margin)
    area2 = (margin, margin, viewer.width - margin, viewer.height*1/3 - margin/2)
    for a in (area1, area2):
        box = rendering.FilledPolygon([(a[0], a[1]), (a[0], a[3]), (a[2], a[3]), (a[2], a[1])])
        box.set_color(.9, .9, .9)
        viewer.add_geom(box)

    trans1 = rendering.Transform()
    scale1 = rendering.Transform()
    trans2 = rendering.Transform()
    scale2 = rendering.Transform()
    min_x, max_x, min_p, max_p, min_b, max_b = None, None, None, None, None, None

    balance_chart = rendering.PolyLine([], close=False)
    balance_chart.set_color(.3, .9, .3)
    balance_chart.linewidth.stroke = 2.0
    balance_chart.add_attr(scale2)
    balance_chart.add_attr(trans2)
    viewer.add_geom(balance_chart)
    
    prev_action = None
    signals = []
    
    def add_bar(state: OrderedDict, action: int, balance: float):
        bar_width, bar_space = 5, 1
        price_margin = 0.05
        xc = float((bar_width + bar_space) * len(states) + (bar_width / 2))
        xl, xr = xc - (bar_width / 2), xc + (bar_width / 2)
        o, h, l, c = state['open'], state['high'], state['low'], state['close']
        green, red = (0.1, 0.9, 0.1), (1.0, 0.2, 0.2)
        clr = green if (c >= o) else red
        vert = rendering.Line((xc, l), (xc, h))
        open = rendering.Line((xl, o), (xc, o))
        close = rendering.Line((xr, c), (xc, c))
        vert.set_color(*clr)
        open.set_color(*clr)
        close.set_color(*clr)
        vert.linewidth.stroke = 2.0
        open.linewidth.stroke = 2.0
        close.linewidth.stroke = 2.0
        vert.add_attr(scale1)
        open.add_attr(scale1)
        close.add_attr(scale1)
        vert.add_attr(trans1)
        open.add_attr(trans1)
        close.add_attr(trans1)
        viewer.add_geom(vert)
        viewer.add_geom(open)
        viewer.add_geom(close)
        nonlocal prev_action
        if (prev_action is None) or (prev_action != action):
            prev_action = action
            label = pyglet.text.Label(
                '▲' if (action == 0) else '▼' if (action == 1) else '●',
                font_size=12,
                x=xc,
                y=l if (action == 0) else h if (action == 1) else c,
                anchor_x='center',
                anchor_y='center',
                color=green if (action == 0) else red if (action == 1) else (0.0, 0.0, 0.0),
            )
            signals.append(label)
        nonlocal min_x, max_x, min_p, max_p, min_b, max_b
        if (min_x is None) or (min_x > xl):
            min_x = xl
        if (max_x is None) or (max_x < xr):
            max_x = xr
        if (min_p is None) or (min_p > l):
            min_p = l
        if (max_p is None) or (max_p < h):
            max_p = h
        if (min_b is None) or (min_b > balance):
            min_b = balance
        if (max_b is None) or (max_b < balance):
            max_b = balance
        scale_x = min(1.0, (area1[2] - area1[0]) / (max_x - min_x + 2 * bar_width))
        scale_p = (area1[3] - area1[1]) / ((max_p - min_p) * (1 + 2 * price_margin))
        scale1.set_scale(scale_x, scale_p)
        trans1.set_translation(
            (bar_width - min_x) * scale_x + area1[0],
            ((max_p - min_p) * price_margin - min_p) * scale_p + area1[1]
        )
        if max_b > min_b:
            scale_b = (area2[3] - area2[1]) / ((max_b - min_b) * (1 + 2 * price_margin))
        else:
            scale_b = (area2[3] - area2[1]) / max_b
        scale2.set_scale(scale_x, scale_b)
        trans2.set_translation(
            (bar_width - min_x) * scale_x + area2[0],
            ((max_b - min_b) * price_margin - min_b) * scale_b + area2[1]
        )
        balance_chart.v.append((xc, balance))

    # Run episode
    while True:
        if callable(model):
            action = model(state)
        else:
            action = env.action_space.sample()

        add_bar(state, action, env.accounts[0].balance)
        viewer.render()
        
        state, reward, done, frame = env.step(action)
        if (state is not None) and (frame is not None):
            state['time'] = state['time_start'].timestamp()
            states.append(state)
            frames.append(frame)
        if done:
            break
        
        time.sleep(0.01)

    if viewer:
        viewer.close()

    return states, frames


if __name__ == '__main__':
    env = get_env()
    run_episode(env)
    