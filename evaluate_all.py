import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C

name = sys.argv[1]                       # e.g. ppo_fixed_s0 or a2c_fixed_s0
episodes = 50

Algo = A2C if name.startswith("a2c") else PPO
model = Algo.load(name)
env = gym.make("LunarLander-v3")         # original game, no wrapper

scores, lengths = [], []
for ep in range(episodes):
    obs, info = env.reset(seed=1000 + ep)
    done, total, steps = False, 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        steps += 1
        done = terminated or truncated
    scores.append(total)
    lengths.append(steps)

scores = np.array(scores)
print(f"Model: {name}")
print(f"Mean score:   {scores.mean():.1f} (std {scores.std():.1f})")
print(f"Success rate: {(scores >= 200).mean() * 100:.0f}% (score >= 200)")
print(f"Mean length:  {np.mean(lengths):.0f} steps")