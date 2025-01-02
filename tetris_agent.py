import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import gymnasium as gym
import typing


class TetrisAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
                 ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    # gets the next action to be performed using an epsilon greedy policy
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update_qvalues(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool]
    ) -> None: 
        future_q_value = (not terminated) * np.max(self.q_values[self.totuple(next_obs[0])])
        temporal_diff = (
            reward + self.discount_factor * future_q_value - self.q_values[self.totuple(obs[0])][action]
        )
        self.q_values[self.totuple(obs[0])][action] = (
            self.q_values[self.totuple(obs[0])][action] + self.lr * temporal_diff
        )
        self.training_error.append(temporal_diff)
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
    def totuple(self, a):
        try:
            return tuple(self.totuple(i) for i in a)
        except TypeError:
            return a
    