#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

import datetime


def callback(lcl, glb):
  # stop training if reward exceeds 199
  total = sum(lcl['episode_rewards'][-101:-1]) / 100
  totalt = lcl['t']
  #print("totalt")
  #print(totalt)
  is_solved = totalt > 2000 and total >= 10
  return is_solved


def main():

  env = KukaGymEnv(renders=False, isDiscrete=True)
  model = DQN(MlpPolicy, env, verbose=1,buffer_size=50000,exploration_final_eps=0.02,exploration_initial_eps=0.1)
  model.learn(1000000,
                    log_interval=100)
  print("Saving model to kuka_model.pkl")
  model.save("kuka_model")


if __name__ == '__main__':
  main()
