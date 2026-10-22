import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

name = sys.argv[1]   # e.g. ppo_hover_hack_s0
model = PPO.load(name)

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix=name,
                  episode_trigger=lambda ep: True)

for ep in range(2):
    obs, info = env.reset(seed=ep)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
env.close()
print("Saved to the videos folder")