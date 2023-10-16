import warnings

from gym.envs.toy_text.frozen_lake import generate_random_map

from src.algorithm.doubleqlearning import DoubleQLearning
from src.algorithm.qlearning import QLearning
from src.algorithm.sarsa import Sarsa
from src.environment import FrozenLake
from src.test import test_agent
from src.train import train_agent


def main() -> None:
    map_size = 16
    n_actions = 4
    q_size = map_size**2
    n_games = 5_000

    env_map = generate_random_map(size=map_size, p=0.7)
    env_train = FrozenLake(env_map)
    env_test = FrozenLake(env_map, render_mode="human")

    q_table = QLearning(q_size, n_actions)
    sarsa = Sarsa(q_size, n_actions)
    double_q_table = DoubleQLearning(q_size, n_actions)

    print("Training Q-Learning agent...")
    train_agent(env_train, q_table, n_games=n_games, alpha=0.5, gamma=0.9)
    train_agent(env_train, sarsa, n_games=n_games, alpha=0.5, gamma=0.9)
    train_agent(env_train, double_q_table, n_games=n_games, alpha=0.1, gamma=0.9)
    env_train.close()

    print("Testing Q-Learning agent...")
    test_agent(env_test, q_table, n_games=1)
    test_agent(env_test, sarsa, n_games=1)
    test_agent(env_test, double_q_table, n_games=1)
    env_test.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
