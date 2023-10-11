import numpy as np


class Sarsa:
    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

        self.table = np.zeros((n_states, n_actions))

    def __getitem__(self, state: int) -> list[float]:
        return self.table[state]

    def update_values(
        self, state: int, next_state: int, action: int, alpha: float, reward: float, gamma: float
    ) -> None:
        average_qvalue = self.calculate_average_qvalue(self.table[next_state], epsilon=0.1)
        self.table[state, action] += alpha * (reward + average_qvalue - self.table[state, action])

    def calculate_average_qvalue(self, values, epsilon=0):
        max_value = max(values)
        n_actions = len(values)
        n_greedy_actions = 0
        for v in values:
            if v == max_value:
                n_greedy_actions += 1

        non_greedy_action_probability = epsilon / n_actions
        greedy_action_probability = (
            (1 - epsilon) / n_greedy_actions
        ) + non_greedy_action_probability

        result = 0
        for v in values:
            if v == max_value:
                result += v * greedy_action_probability
            else:
                result += v * non_greedy_action_probability

        return result
