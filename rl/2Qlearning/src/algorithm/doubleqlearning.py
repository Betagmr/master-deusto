import numpy as np


class DoubleQLearning:
    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

        self.table_1 = np.zeros((n_states, n_actions))
        self.table_2 = np.zeros((n_states, n_actions))

    def __getitem__(self, state: int) -> list[float]:
        return self.table_1[state] + self.table_2[state]

    def update_values(
        self, state: int, next_state: int, action: int, alpha: float, reward: float, gamma: float
    ) -> None:
        if np.random.random() < 0.5:
            self.table_1[state, action] += alpha * (
                reward
                + gamma * self.table_2[next_state][np.argmax(self.table_1[next_state])]
                - self.table_1[state][action]
            )
        else:
            self.table_2[state, action] += alpha * (
                reward
                + gamma * self.table_1[next_state][np.argmax(self.table_2[next_state])]
                - self.table_2[state][action]
            )
