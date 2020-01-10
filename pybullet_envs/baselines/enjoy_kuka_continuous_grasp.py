#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import time
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.kukaGymFullEnv import KukaGymFullEnv
from pybullet_envs.bullet.kukaGymRotationEnv import KukaGymRotationEnv
from stable_baselines.common.policies import MlpPolicy
from pybullet_envs.bullet.kukaGymCamEnv import KukaGymCamEnv

from stable_baselines import PPO1,PPO2

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG,TRPO,DQN



def main():
  random_policy = False
  env = KukaGymFullEnv(renders=True, isDiscrete=False,maxSteps=15)
  #env = KukaGymCamEnv(renders=True, isDiscrete=False, maxSteps=15)

  model = PPO1.load("ppo_kuka_full_5M")
  total_grasps = 0
  success_grasps = 0
  #obs = env.reset()
  while True:
    obs, dones = env.reset(), False
    print("===================================")
    print("obs")
    #print(obs)
    episode_rew = 0
    while not dones:
      if random_policy:
          action = env.action_space.sample()
      else:
          action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.debug_text("PPO-3M",str(success_grasps)+"/"+str(total_grasps))
      #env.render()
      #obs, rew, done, _ = env.step(a)
      episode_rew += rewards
    time.sleep(0.5)
    total_grasps += 1
    if episode_rew > 9000:
    	success_grasps += 1
    print("Episode reward", episode_rew)


if __name__ == '__main__':
  main()
