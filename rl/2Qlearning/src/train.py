import gym
import numpy as np


def train_agent(agent, n_games: int = 5000, alpha: float = 0.7, gamma: float = 0.9) -> None:
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
    )

    for episodes in range(n_games):
        done = False
        state, _ = env.reset()

        while not done:
            if np.max(agent[state]) > 0:
                action = np.argmax(agent[state])
            else:
                action = env.action_space.sample()

            new_state, reward, done, *_ = env.step(action)
            agent.update_values(state, new_state, action, alpha, reward, gamma)
            state = new_state

        if episodes % 500 == 0:
            print(f"Game {episodes} ended with reward = {reward}.")

    env.close()
