from stable_baselines3 import PPO

from src.snakeenv import SnakeEnv


def train():
    n_steps = 100_000
    lr = 0.1

    env = SnakeEnv(render_mode=False)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        verbose=1,
        n_steps=n_steps,
        batch_size=10_000,
        tensorboard_log="./logs/",
    )
    model.learn(total_timesteps=n_steps, progress_bar=True)


if __name__ == "__main__":
    train()
