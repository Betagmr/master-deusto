import gym
import optuna
from gym.envs.toy_text.frozen_lake import generate_random_map
from optuna.trial import Trial

from src.algorithm import get_algorithm_instance
from src.environment import FrozenLake
from src.test import test_agent
from src.train import train_agent


def create_objective_function(env, q_size: int, n_actions: int):
    def objetive(trial: Trial):
        algorithm_name = trial.suggest_categorical(
            "algorithm_name", ["sarsa", "qlearning", "doubleqlearning"]
        )
        lr = trial.suggest_float("learning_rate", 0.1, 1, step=0.05)
        lr_decay = trial.suggest_float("learning_rate_decay", 0.1, 1, step=0.05)
        epsilon_decay = trial.suggest_float("epsiolon_decay", 0.00001, 1, log=True)

        agent = get_algorithm_instance(algorithm_name, q_size, n_actions)

        steps = train_agent(
            env=env,
            agent=agent,
            alpha=lr,
            gamma=lr_decay,
            epsilon_decay=epsilon_decay,
        )

        test_rewards = test_agent(env, agent, n_games=1)

        return steps if test_rewards > 1 else 100_000

    return objetive


if __name__ == "__main__":
    env_frozen = FrozenLake(generate_random_map(size=16, p=0.8))
    env_taxi = gym.make("Taxi-v3")

    study_frozen = optuna.create_study(
        study_name="frozen-study",
        storage="sqlite:///frozen-study.db",
        direction="minimize",
    )
    study_frozen.optimize(
        create_objective_function(env_frozen, q_size=16**2, n_actions=4),
        n_trials=50,
    )

    # study_taxi = optuna.create_study(
    #     study_name="taxi-study",
    #     storage="sqlite:///taxi-study.db",
    #     direction="minimize",
    # )
    # study_taxi.optimize(
    #     create_objective_function(env_taxi, q_size=500, n_actions=6),
    #     n_trials=100,
    # )
