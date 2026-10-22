import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_baseline")
env = gym.make("LunarLander-v3", render_mode="human")

scores = []
for ep in range(10):
    obs, info = env.reset()
    done, total = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    scores.append(total)
    print(f"Episode {ep + 1}: score = {total:.1f}")

print(f"Average over 10 episodes: {sum(scores) / len(scores):.1f}")
env.close()