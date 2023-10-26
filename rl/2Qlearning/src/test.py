import numpy as np


def test_agent(env, agent) -> int:
    total_reward = 0
    done = False
    state, _ = env.reset()
    steps = 0
    while not done:
        action = np.argmax(agent[state])
        new_state, reward, done, _, info = env.step(action)
        state = new_state
        total_reward += reward
        steps += 1

        if steps > 1_000:
            return total_reward

    return total_reward
