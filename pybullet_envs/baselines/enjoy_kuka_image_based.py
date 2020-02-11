#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from PIL import Image
import gym
import time
from pybullet_envs.bullet.kukaImageEnv import KukaImageEnv




def main():
  random_policy = True
  env = KukaImageEnv(renders=True, isDiscrete=True,maxSteps=15)
  total_grasps = 0
  random_policy = True
  success_grasps = 0
  recording = False
  # while True:
  #   pass
  #logid = env._p.startStateLogging(env._p.STATE_LOGGING_VIDEO_MP4, "~/Desktop/kuka_auto_data2.mp4")
  for i in range(3):
  #obs = env.reset()
    env.trajectory_control([0.3,-0.3,0.5],0,0,30)

    env.controller_roundingBox()
    for i in range(100):
      env._p.stepSimulation()
      if env._renders:
        time.sleep(env._timeStep)
    env._get_observation()
    obs, dones = env.reset(), False

  #env._p.stopStateLogging(logid)


if __name__ == '__main__':
  main()
