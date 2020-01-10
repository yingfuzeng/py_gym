#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from PIL import Image
import gym
import time
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.kukaGymRotationEnv import KukaGymRotationEnv
from pybullet_envs.bullet.kukaGymCamEnv import KukaGymCamEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG,TRPO,DQN



def main():
  random_policy = True
  env = KukaDiverseObjectEnv(renders=True, isDiscrete=True,maxSteps=15)
  #env = KukaGymRotationEnv(renders=True, isDiscrete=True, maxSteps=15)
  #env = KukaGymEnv(renders=True, isDiscrete=True, maxSteps=15)
  model = DQN.load("deepq_kuka_cam_1M")
  total_grasps = 0
  success_grasps = 0
  recording = False


  while True:
  #obs = env.reset()
    obs, dones = env.reset(), False
    print("===================================")
    print("obs")

    episode_rew = 0
    counter = 0
    while not dones:
      if random_policy:
          action = env.action_space.sample()
      else:
          action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      # print(obs.shape)
      # img = Image.fromarray(obs, 'RGB')
      # img.save('./obs/ob' +str(counter)+'.png')
      # counter += 1
      print(rewards)
      #env.debug_text("DQN-100K",str(success_grasps)+"/"+str(total_grasps))
      #env.render()
      #obs, rew, done, _ = env.step(a)
      episode_rew += rewards
    time.sleep(0.5)
    total_grasps += 1
    if episode_rew ==1:
    	success_grasps += 1
    print("Episode reward", episode_rew)


if __name__ == '__main__':
  main()
