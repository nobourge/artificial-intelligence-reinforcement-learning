from rlenv import RLEnv, Observation
import numpy as np
import numpy.typing as npt

import sys

print(sys.path)


class QLearning:
    """Tabular QLearning"""

    def __init__(self, learning_rate: float, discount_factor: float, epsilon: float):
        # Initialize parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # Initialize Q-table
        self.q_table = {}

    def choose_action(self, state):
        # Implement action selection using epsilon-greedy strategy
        pass

    def update(self, state, action, reward, next_state):
        # Implement Q-table update
        pass

if __name__ == "__main__":
    print("Hello World")
    print(sys.path)