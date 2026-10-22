import gymnasium as gym


class TimePenalty(gym.Wrapper):
    def __init__(self, env, penalty=0.03):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.penalty
        return obs, reward, terminated, truncated, info


class HoverBonus(gym.Wrapper):
    """Deliberately bad reward: +1 per frame while upright and slow."""

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        upright = abs(obs[4]) < 0.2
        slow = abs(obs[2]) < 0.3 and abs(obs[3]) < 0.3
        if upright and slow:
            reward += self.bonus
        return obs, reward, terminated, truncated, info


class FixedShaping(gym.Wrapper):
    """Same idea as HoverBonus, but pays for the CHANGE in 'upright and slow'
    (potential-based shaping) plus a small time fee."""

    def __init__(self, env, scale=10.0, penalty=0.03, gamma=0.999):
        super().__init__(env)
        self.scale = scale
        self.penalty = penalty
        self.gamma = gamma
        self.prev = 0.0

    def _potential(self, obs):
        upright = abs(obs[4]) < 0.2
        slow = abs(obs[2]) < 0.3 and abs(obs[3]) < 0.3
        return self.scale if (upright and slow) else 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new = 0.0 if terminated else self._potential(obs)
        reward += self.gamma * new - self.prev
        reward -= self.penalty
        self.prev = new
        return obs, reward, terminated, truncated, info