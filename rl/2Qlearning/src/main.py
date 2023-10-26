import warnings

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

from src.algorithm.doubleqlearning import DoubleQLearning
from src.algorithm.qlearning import QLearning
from src.algorithm.sarsa import Sarsa
from src.environment import FrozenLake
from src.test import test_agent
from src.train import train_agent


def main() -> None:
    map_size = 16
    n_games = 25_000

    q_size = map_size**2
    n_actions = 4
    env_map = generate_random_map(size=map_size, p=0.8)
    env_train = FrozenLake(env_map)
    env_test = FrozenLake(env_map, render_mode="human")

    # q_size = 50
    # n_actions = 6
    # env_train = gym.make("Taxi-v3")
    # env_test = gym.make("Taxi-v3", render_mode="human")

    # q_table = QLearning(q_size, n_actions)
    sarsa = Sarsa(q_size, n_actions)
    double_q_table = DoubleQLearning(q_size, n_actions)

    print("Training Q-Learning agent...")
    # train_agent(env_train, q_table, n_games=n_games, alpha=0.5, gamma=0.9)
    train_agent(
        env_train,
        double_q_table,
        n_games=n_games,
        alpha=0.2,
        gamma=0.8,
        epsilon_decay=0.01,
    )
    # train_agent(env_train, double_q_table, n_games=n_games, alpha=0.9, gamma=0.45, epsilon_decay=0.85)
    env_train.close()

    print("Testing Q-Learning agent...")
    # test_agent(env_test, q_table, n_games=1)
    # test_agent(env_test, sarsa, n_games=1)
    test_agent(env_test, double_q_table, n_games=1)
    env_test.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
