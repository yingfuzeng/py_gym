import gym
import os, inspect,time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from pybullet_envs.bullet.kukaGymCamEnv import KukaGymCamEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from stable_baselines import PPO1,PPO2
#env = KukaGymCamEnv(renders=False, isDiscrete=False, maxSteps=15)
kwargs = {'renders': False, 'isDiscrete': False, 'maxSteps': 15}
# multiprocess environment
env = make_vec_env(KukaDiverseObjectEnv, n_envs=8, env_kwargs = kwargs)

model = PPO2(CnnPolicy, env, verbose=1)
start_time = time.time()
model.learn(total_timesteps=1000000)
print(time.time() - start_time)
model.save("ppo2_diverse_1M")

# del model # remove to demonstrate saving and loading
#
# model = PPO2.load("ppo2_cartpole")
#
# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()