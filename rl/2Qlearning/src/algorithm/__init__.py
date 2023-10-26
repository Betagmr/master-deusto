from .doubleqlearning import DoubleQLearning
from .qlearning import QLearning
from .sarsa import Sarsa

algorithm_dict = {
    "sarsa": Sarsa,
    "qlearning": QLearning,
    "doubleqlearning": DoubleQLearning,
}


def get_algorithm_instance(algorithm_name: str, q_size: int = 16, n_actions: int = 4):
    return algorithm_dict[algorithm_name](q_size, n_actions)
