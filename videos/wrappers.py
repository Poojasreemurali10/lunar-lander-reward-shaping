import gymnasium as gym

class TimePenalty(gym.Wrapper):
    def __init__(self, env, penalty=0.03):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.penalty
        return obs, reward, terminated, truncated, info