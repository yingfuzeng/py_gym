#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import numpy as np
import gym
import tensorflow as tf
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.kukaGymCamEnv import KukaGymCamEnv
from pybullet_envs.bullet.kukaGymRotationEnv import KukaGymRotationEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG,TRPO,DQN
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import FeedForwardPolicy

import datetime


def modified_cnn(scaled_images, **kwargs):
  #print(scaled_images)
  activ = tf.nn.relu
  layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
  layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
  layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
  layer_3 = conv_to_fc(layer_3)
  return activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))


class CustomPolicy(FeedForwardPolicy):
  def __init__(self, *args, **kwargs):
    super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")
def main():

  env = KukaDiverseObjectEnv(renders=False, isDiscrete=True,maxSteps=15)
  #model = DQN.load("deepq_kuka_rotation_100K")
  #model.set_env(env)
  #model = DQN(MlpPolicy, env, verbose=1,exploration_final_eps=0.05,tensorboard_log="./DQN_r_1M")
  model = DQN(CnnPolicy, env, verbose=1, exploration_final_eps=0.02, tensorboard_log="./DQN_diverse_100K")
  #model.tensorboard_log = "./DQN_r_500k"
  model.learn(total_timesteps=100000)
  model.save("deepq_kuka_diverse_100K")

  print("Saving model to kuka_model.pkl")


if __name__ == '__main__':
  main()
