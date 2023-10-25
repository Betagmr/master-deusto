from optuna.trial import Trial
from algorithm import algorithm_dict
from src.environment import FrozenLake


def objetive(trial: Trial):
    algorithm_name = trial.suggest_categorical(
        "algorithm_name", ["sarsa", "qlearning", "dobleqlearning"]
    )
    lr = trial.suggest_float("learning_rate", 0.1, 1, log=True)
    lr_decay = trial.suggest_float("learning_r  ate_decay", 0.1, 1.0, step=0.1)
    epsilon_decay = trial.suggest_float("epsiolon_decay", 1e-5, 1e-1, log=True)

    map_size = 16
    n_actions = 4
    q_size = map_size**2
    n_games = 5_000

    agent = algorithm_dict[algorithm_name](q_size, n_actions)

    return 0
