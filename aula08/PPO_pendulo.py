import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================
# Callback para armazenar recompensas dos episódios
# ============================================================
class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:

        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])

        return True

# ============================================================
# Função para criar ambientes
# ============================================================
def make_env():
    def _init():
        env = gym.make("CartPole-v1")
        env = Monitor(env)
        return env
    return _init

# ============================================================
# Criando 10 ambientes paralelos
# ============================================================
num_envs = 10
env = DummyVecEnv([make_env() for _ in range(num_envs)])

# ============================================================
# Modelo PPO
# ============================================================
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1
)

# ============================================================
# Treinamento
# ============================================================
callback = RewardCallback()

total_timesteps = 250_000

model.learn(
    total_timesteps=total_timesteps,
    callback=callback
)


# ============================================================
# Processando recompensas
# ============================================================
rewards = np.array(callback.episode_rewards)
window = 50

moving_avg = np.convolve(
    rewards,
    np.ones(window) / window,
    mode='valid'
)

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(8, 3))

plt.plot(rewards, 'r', alpha=0.2, linewidth=1)
plt.plot(moving_avg, 'r', linewidth=2, label="PPO           ")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=12, facecolor='w')
plt.xlabel("Episódios")
plt.ylabel("Recompensa")
plt.xlim([0, 600])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.gcf().patch.set_alpha(0)

plt.show()
