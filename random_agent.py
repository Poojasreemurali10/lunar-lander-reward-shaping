import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=0)

for step in range(500):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    if step < 5:
        print(f"step {step}: obs={obs.round(2)}, reward={reward:.2f}")

    if terminated or truncated:
        print(f"Episode ended at step {step}, last reward = {reward:.1f}")
        obs, info = env.reset()

env.close()