import random
import warnings

import numpy as np

from src.algorithm.doubleqlearning import DoubleQLearning
from src.algorithm.qlearning import QLearning
from src.algorithm.sarsa import Sarsa
from src.test import test_agent
from src.train import train_agent


def main() -> None:
    size = 8 * 8
    n_actions = 4
    seed = 42

    q_table = QLearning(size, n_actions)
    sarsa = Sarsa(size, n_actions)
    double_q_table = DoubleQLearning(size, n_actions)

    print("Training Q-Learning agent...")
    np.random.seed(seed)
    random.seed(seed)
    train_agent(q_table)

    print("\nTraining sarsa agent...")
    np.random.seed(seed)
    random.seed(seed)
    train_agent(sarsa)

    print("\nTraining Double Q-Learning agent...")
    np.random.seed(seed)
    random.seed(seed)
    train_agent(double_q_table)

    print("\nTesting Q-Learning agent...")
    test_agent(q_table)
    print("\nTesting Sarsa agent...")
    test_agent(sarsa)
    print("\nTesting DoubleQ-Learning agent...")
    test_agent(double_q_table)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
