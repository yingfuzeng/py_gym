#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

from baselines import deepq


def main():

  env = KukaGymEnv(renders=True, isDiscrete=True)
  #act = deepq.load("kuka_model.pkl")
  #print(act)
  while True:
    obs, done = env.reset(), False
    print("===================================")
    print("obs")
    print(obs)
    episode_rew = 0
    while not done:
      env.render()
      a = env.action_space.sample()
      print(a)
      #obs, rew, done, _ = env.step(act(obs[None])[0])
      obs, rew, done, _ = env.step(a)
      episode_rew += rew
    print("Episode reward", episode_rew)


if __name__ == '__main__':
  main()
