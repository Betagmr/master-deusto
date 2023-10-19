import numpy as np


def test_agent(env, agent, n_games: int = 1) -> None:
    total_reward: float = 0
    for _ in range(n_games):
        done = False
        state, _ = env.reset()

        while not done:
            action = np.argmax(agent[state])
            new_state, reward, done, _, info = env.step(action)
            state = new_state
            total_reward += reward

    print(f"Total reward = {total_reward}.")
