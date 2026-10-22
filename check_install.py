import gymnasium as gym
import stable_baselines3, torch

print("gymnasium", gym.__version__)
print("sb3", stable_baselines3.__version__)
print("torch", torch.__version__)

env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=0)
print("LunarLander works. First observation:", obs.round(2))