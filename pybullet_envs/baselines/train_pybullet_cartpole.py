#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DDPG


def callback(lcl, glb):
  # stop training if reward exceeds 199
  is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
  return is_solved


def main():

  env = CartPoleBulletEnv(renders=False)
  model = DDPG(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=500000)
  model.save("deepq_cartpole")
  print("Saving model to deepq_cartpole")



if __name__ == '__main__':
  main()
