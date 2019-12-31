#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import numpy as np
import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.kukaGymFullEnv import KukaGymFullEnv
from pybullet_envs.bullet.kukaGymCamEnv import KukaGymCamEnv
from pybullet_envs.bullet.kukaGymRotationEnv import KukaGymRotationEnv

from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO1

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG,TRPO,DQN

import datetime


def main():
  #env = KukaGymFullEnv(renders=False, isDiscrete=False, maxSteps=15)
  env = KukaGymCamEnv(renders=False, isDiscrete=True, maxSteps=15)
  #env = KukaGymRotationEnv(renders=False, isDiscrete=False,maxSteps=15)
  #model = DQN.load("deepq_kuka_rotation_100K")
  #model.set_env(env)
  model = PPO1(MlpPolicy, env, verbose=1,tensorboard_log="./PPO_cam_test")
  #model = DQN(MlpPolicy, env, verbose=1,exploration_final_eps=0.05,tensorboard_log="./DQN_r_1M")
  #model.tensorboard_log = "./DQN_r_500k"
  model.learn(total_timesteps=100000, log_interval = 1000)
  model.save("ppo_cam_test")

  print("Saving model to kuka_model.pkl")


if __name__ == '__main__':
  main()
