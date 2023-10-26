import gym


class FrozenLake(gym.Wrapper):
    def __init__(self, env_map, render_mode=None, max_steps=300):
        env = gym.make(
            "FrozenLake-v1",
            desc=env_map,
            is_slippery=False,
            render_mode=render_mode,
        )
        env._max_episode_steps = max_steps

        super().__init__(env)

        self.env = env
        self.n_rows = env.nrow
        self.n_columns = env.nrow
        self.list_of_steps = []

    def step(self, action):
        new_state, reward, done, aux, info = self.env.step(action)

        if reward == 1:
            reward = 20
        elif reward == 0 and done:
            reward = -1
        elif new_state in self.list_of_steps:
            reward = -1
            done = True
        else:
            distance = self.get_player_distance(new_state)
            reward = (30 - distance) / 1000

        self.list_of_steps.append(new_state)
        info["n_steps"] = len(self.list_of_steps)

        return new_state, reward, done, aux, info

    def get_player_distance(self, state):
        state_row = state // (self.n_rows - 1)
        state_column = state % (self.n_columns - 1)
        distance = self.n_rows - 1 - state_row + self.n_columns - 1 - state_column

        return distance

    def reset(self):
        state, info = self.env.reset()
        self.list_of_steps = [state]

        return state, info
