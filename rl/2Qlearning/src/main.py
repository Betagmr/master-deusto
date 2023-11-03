import random
import warnings

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

from src.algorithm import get_algorithm_instance
from src.algorithm.doubleqlearning import DoubleQLearning
from src.algorithm.qlearning import QLearning
from src.algorithm.sarsa import Sarsa
from src.environment import FrozenLake
from src.test import test_agent
from src.train import train_agent


def launch_frozen_lake(agent_name, map_size: int = 16, n_games: int = 15_000) -> None:
    q_size = map_size**2
    n_actions = 4
    env_map = generate_random_map(size=map_size, p=0.8)
    env_train = FrozenLake(env_map)
    env_test = FrozenLake(env_map, render_mode="human")

    agent = get_algorithm_instance(agent_name, q_size, n_actions)
    train_agent(env_train, agent, n_games=n_games, alpha=0.2, gamma=0.9, epsilon_decay=0.01)
    test_agent(env_test, agent)
    env_test.close()


def launch_taxi(agent_name, n_games: int = 15_000) -> None:
    q_size = 500
    n_actions = 6
    env_train = gym.make("Taxi-v3")
    env_test = gym.make("Taxi-v3", render_mode="human")

    agent = get_algorithm_instance(agent_name, q_size, n_actions)
    train_agent(env_train, agent, n_games=n_games, alpha=0.2, gamma=0.9, epsilon_decay=0.01)
    test_agent(env_test, agent)


def main() -> None:
    # agent_name -> sarsa, qlearning, doubleqlearning
    agent_name = "doubleqlearning"
    launch_frozen_lake(agent_name)
    launch_taxi(agent_name)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
