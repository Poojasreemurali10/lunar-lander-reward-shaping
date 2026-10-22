import sys
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from wrappers import TimePenalty, HoverBonus, FixedShaping

variant = sys.argv[1]       # baseline, time_penalty, hover_hack or fixed
seed = int(sys.argv[2])     # e.g. 0
algo = sys.argv[3].lower() if len(sys.argv) > 3 else "ppo"   # ppo or a2c

if variant == "time_penalty":
    wrapper, kwargs = TimePenalty, {"penalty": 0.03}
elif variant == "hover_hack":
    wrapper, kwargs = HoverBonus, {"bonus": 1.0}
elif variant == "fixed":
    wrapper, kwargs = FixedShaping, {}
else:
    wrapper, kwargs = None, {}

env = make_vec_env("LunarLander-v3", n_envs=16, seed=seed,
                   wrapper_class=wrapper, wrapper_kwargs=kwargs)

if algo == "a2c":
    model = A2C("MlpPolicy", env, n_steps=5, gamma=0.995,
                learning_rate=0.00083, ent_coef=0.00001, seed=seed, verbose=1)
else:
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4,
                gamma=0.999, gae_lambda=0.98, ent_coef=0.01, seed=seed, verbose=1)

try:
    model.learn(total_timesteps=1_000_000)
except KeyboardInterrupt:
    print("Interrupted, saving progress...")

model.save(f"{algo}_{variant}_s{seed}")
print(f"Saved {algo}_{variant}_s{seed}.zip")