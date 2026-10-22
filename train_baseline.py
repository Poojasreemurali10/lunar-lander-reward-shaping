from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

env = make_vec_env("LunarLander-v3", n_envs=16)

model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

# Saves a checkpoint every ~100k steps
checkpoint = CheckpointCallback(
    save_freq=6250,  # counted per env, x16 envs = ~100k steps
    save_path="./checkpoints/",
    name_prefix="ppo_baseline",
)

try:
    model.learn(total_timesteps=1_000_000, callback=checkpoint)
except KeyboardInterrupt:
    print("Interrupted, saving progress...")

model.save("ppo_baseline")
print("Saved ppo_baseline.zip")
