import gym
import numpy as np


def test_agent(agent, n_games: int = 5) -> None:
    env = gym.make(
        "FrozenLake-v1",
        render_mode="human",
        map_name="8x8",
        is_slippery=False,
    )

    total_reward: float = 0
    for _ in range(n_games):
        done = False
        state, _ = env.reset()

        while not done:
            action = np.argmax(agent[state])
            new_state, reward, done, *_ = env.step(action)
            state = new_state
            total_reward += reward

    print(f"Average reward score {n_games} games: {total_reward / n_games * 100}%")
    env.close()
