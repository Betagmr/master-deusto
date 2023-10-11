from typing import Any

import numpy as np


class QLearning:
    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

        self.table = np.zeros((n_states, n_actions))

    def __getitem__(self, state: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.table[state]

    def update_values(
        self, state: int, next_state: int, action: int, alpha: float, reward: float, gamma: float
    ) -> None:
        self.table[state, action] += alpha * (
            reward + gamma * np.max(self.table[next_state]) - self.table[state, action]
        )
