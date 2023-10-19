import numpy as np


def train_agent(
    env,
    agent,
    n_games: int = 3000,
    alpha: float = 0.7,
    gamma: float = 0.9,
    epsilon_decay: float = 0.001,
) -> None:
    epsilon = 1.0
    steps = 0

    for episodes in range(n_games):
        done = False
        state, _ = env.reset()
        total_reward = 0

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent[state])
            else:
                action = env.action_space.sample()

            new_state, reward, done, _, info = env.step(action)
            agent.update_values(state, new_state, action, alpha, reward, gamma)
            state = new_state
            total_reward += reward
            steps += 1

        epsilon = epsilon - epsilon_decay if epsilon > 0.01 else 0.01

        if episodes % 100 == 0:
            print(f"Game {episodes} ended with reward = {total_reward}.")
            print(f"Steps: {steps}")
