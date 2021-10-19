from __future__ import annotations
from typing import Union
from numbers import Real
from .account import Account


class RewardScheme(object):
    def __init__(self, **kwargs):
        pass
    
    def reset(self):
        raise NotImplementedError()
    
    def get_reward(self, env, account: Account) -> float:
        raise NotImplementedError()
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ConstantReward(RewardScheme):
    def __init__(self, value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.value = value
    
    def reset(self):
        pass
    
    def get_reward(self, env, account: Account) -> float:
        return self.value


class BalanceReward(RewardScheme):
    def __init__(self, norm_factor: Union[None, Real, str] = None, **kwargs):
        super().__init__(**kwargs)
        self.norm_factor = norm_factor
        self.last_balance = {}
    
    def reset(self):
        self.last_balance.clear()
        
    def get_reward(self, env, account: Account) -> float:
        reward = (account.balance - self.last_balance[account]) if (account in self.last_balance) else 0
        self.last_balance[account] = account.balance
        if isinstance(self.norm_factor, Real):
            reward = reward / float(self.norm_factor)
        elif isinstance(self.norm_factor, str):
            last_frame = env.frames[-1]
            norm = getattr(last_frame, self.norm_factor)
            reward = (reward / float(norm)) if (norm > 1e-8) else 0.0
        return reward
