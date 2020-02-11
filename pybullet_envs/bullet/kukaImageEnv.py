import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import matplotlib.pyplot as plt
import pybullet as p
from . import kuka
import cv2
import random
import pybullet_data
from pkg_resources import parse_version
from itertools import product


largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
object_path = '/home/yingfu/Documents/gym/pybullet_data/blocks/'

class KukaImageEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=70,
               object_num =3,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=1000):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._object_num = object_num
    self._object_ids = []
    self._object_box_points = {}
    self._debug_lines = []
    self._width = 300
    self._height = 300
    self._far = 0.8
    self._near = 0.1

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0, -0.3, -0.43])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self.seed()
    self.reset()
    #observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    #observation_high = np.array([largeValObservation] * observationDim)
    # if (self._isDiscrete):
    #   self.action_space = spaces.Discrete(7)
    # else:
    #   action_dim = 4
    #   space_low = np.array([-1,-1,-1,-1])
    #   space_high = np.array([1, 1, 1,1])
    #   self._action_bound = 1
    #   self.action_space = spaces.Box(space_low, space_high, dtype=np.float16)
    # self.observation_space = spaces.Box(-observation_high, observation_high)
    # self.viewer = None

  def _get_observation(self):
    """Return the observation as an image.
    """
    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._view_matrix,
                               projectionMatrix=self._proj_matrix)
    rgb = img_arr[2]
    far = self._far
    near = self._near
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    np_depth_arr = img_arr[3]#np.reshape(img_arr[3], (self._height, self._width))
    depth_buffer_opengl = np.reshape(img_arr[3], [self._width, self._height])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    seg_opengl = np.reshape(img_arr[4], [self._width, self._height])
    print(seg_opengl)
    # plt.subplot(2, 2, 1)
    # plt.imshow(depth_opengl, cmap='gray', vmin=0, vmax=0.8)
    # plt.title('Depth OpenGL3')
    # plt.subplot(2, 2, 2)
    # plt.imshow(seg_opengl, cmap='gray', vmin=-1, vmax=1)
    # plt.title('Seg OpenGL3')
    # plt.show()

    return np_img_arr[:, :, :3]

  def reset(self):
    #print("KukaGymEnv _reset")
    self._object_ids =[]
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)

    # Camera related
    look = [0.3, 0.3, 0.7]
    distance = 0.7
    pitch = 0
    yaw = 245
    roll = 0
    mul = 0.5
    mul2 = 1
    self._far = 0.8
    self._near = 0.1
    self._view_matrix = p.computeViewMatrix(
      cameraEyePosition=[0, 0, 0.8],
      cameraTargetPosition=[0, 0, 0],
      cameraUpVector=[0, 1, 0])
    self._proj_matrix = p.computeProjectionMatrixFOV(
      fov=45.0,
      aspect=1.0,
      nearVal=self._near,
      farVal=self._far)
    #self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    #self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    random_objs = random.sample(range(0, 14), self._object_num)
    print("Loading objects..", random_objs)
    x_offset= [-0.1, 0, 0.1,0.7,0.6]
    y_offset= [0.1, 0, -0.1,0.25,0.3]
    iter = 0
    #random_objs = [0]
    for n in random_objs:
      obj_file = object_path + str(n)+".urdf"
      xpos = x_offset[iter] + 0.05 * random.random()
      ypos = y_offset[iter] + 0.05 * random.random()
      iter += 1
      ang = 3.14 * 0.5 + 3.1415925438 * random.random()
      orn = p.getQuaternionFromEuler([0, 0, ang])
      print("orn",orn)
      id = p.loadURDF(obj_file, 0, 0, 0,
                      0,0,0,1)

      ps = self.getAABBPoints(self._p.getAABB(id))
      self._object_box_points[id] = ps
      p.resetBasePositionAndOrientation(id, [xpos, ypos, 0.2],
                                              [orn[0], orn[1], orn[2], orn[3]])
      self._object_ids.append(id)
      #current_pos = self._kuka.current_pos()

    self.blockUid = self._object_ids[0]

    self._envStepCounter = 0
    p.stepSimulation()
    #self._observation = self.getExtendedObservation()
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, 0])
    #p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.850000,
    #          0.000000, 0.000000, 0.0, 1.0)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    p.setGravity(0, 0, -10)
    for i in range(100):
      time.sleep(self._timeStep)
      p.stepSimulation()
    self._observation = self._get_observation()

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._kuka.getObservation()
    gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
    dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
    dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

    gripperEul = p.getEulerFromQuaternion(gripperOrn)
    #print("gripperEul")
    #print(gripperEul)
    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    #print("projectedBlockPos2D")
    #print(projectedBlockPos2D)
    #print("blockEulerInGripper")
    #print(blockEulerInGripper)

    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

    self._observation.extend(list(blockInGripperPosXYEulZ))
    return self._observation

  def step(self, action):
    if (self._isDiscrete):
      dv = 0.03
      # dx = dv*action[0]
      # dy = dv*action[1]
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
      f = 0.3
      realAction = [dx, dy, -0.03, da, f]
    else:
      #print("action[0]=", str(action[0]))
      dv = 0.03
      dx = action[0] * dv
      dy = action[1] * dv
      dz = action[2] * dv
      da = action[3] * 0.25
      f = 0.3
      realAction = [dx, dy, dz-0.9*dv, da, f]
    return self.step2(realAction)

  def move_to_object(self):
    self.roundingBox()

  def trajectory_control(self,target,angle, fangle,steps):
    self._get_observation()
    current_pos = self._kuka.current_pos()
    dis = np.array(target) - np.array(current_pos)
    print("translation", dis)
    current_angle = self._kuka.endEffectorAngle
    print("orientation", current_angle,angle)
    for i in range(steps):
      print( (angle-current_angle)/steps)
      self._kuka.applyAction([dis[0] / steps, dis[1] / steps, dis[2] / steps, (angle-current_angle)/steps, fangle])
      for i in range(10):
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)

  def grap_action(self, angle):

    for i in range(30):
      self._kuka.gripperControl(angle)
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)


  def controller_roundingBox(self):
    done_objects = []
    done_p = 0
    while True:

      max_height = -999
      next_obj = None
      # Determine the highest object
      for id in self._object_ids:
        if id not in done_objects:
          pos, angle = self.getObjectAcessPoint(id)
          if pos[2] > max_height:
            next_obj = id
            max_height = pos[2]

      # Remove lines
      self.remove_debug_lines()

      if next_obj != None:
        pos, angle = self.getObjectAcessPoint(next_obj)
        self.primitive_move(pos, angle)

        obj_height = p.getBasePositionAndOrientation(next_obj)[0][2]
        if obj_height > 0.1:
          p.resetBasePositionAndOrientation(next_obj, [0+done_p, -0.8, 0],
                                            [0, 0, 0, 1])
          done_p += 0.2
          done_objects.append(next_obj)
          for i in range(10):
            p.stepSimulation()
            time.sleep(self._timeStep)
        self.remove_debug_lines()
      else:
        break





  def primitive_move(self, pos,angle):
    offset_y = 0.02
    # First move the point above object
    target = [pos[0], pos[1], pos[2] + 0.5]
    self.trajectory_control(target,angle, 0,20)
    # Pregrap
    target = [pos[0], pos[1], pos[2]+0.15 ]
    self.trajectory_control(target,angle, 0,20)
    # Grap it
    self.grap_action(0.1)
    # Move it to
    self.trajectory_control([target[0],target[1],0.5],angle, 0.06,20)
    self.trajectory_control([0.5, -0.5, 0.5], -1, 0.06,30)
    for i in range(100):
      time.sleep(self._timeStep)
      self._kuka.get_joint_info()
      p.stepSimulation()
    # Check if successful


  def step2(self, action):
    self._kuka.applyAction(action)
    for i in range(self._actionRepeat):
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      if self._termination():
        break
    self._envStepCounter += 1

    self._observation = self.getExtendedObservation()

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()
    npaction = np.array([
        action[3]
    ])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    actionCost = np.linalg.norm(npaction) * 10.
    #print("actionCost")
    #print(actionCost)
    reward = self._reward() #- actionCost
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def debug_text(self, text1,text2):
    self._p.addUserDebugText(text2,[0.9,-0.2,0.4],textSize = 2,textColorRGB = [0.2,0.2,0.7])
    self._p.addUserDebugText(text1, [0.9, -0.2, 0.5], textSize=2, textColorRGB=[0.2, 0.2, 0.7])


  def getObjectAcessPoint(self,id):
    def dist(p1, p2):
      pairs = list(zip(p1,p2))
      return sum([(pair[0] - pair[1])*(pair[0] - pair[1]) for pair in pairs])
    def get_mid_point(p1, p2):
      return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2,(p1[2]+p2[2])/2]
    #id = self._object_ids[0]
    # Original positions
    ori_pos = self._object_box_points[id]
    # Current positions and z-axis angle
    current_pos, orn = p.getBasePositionAndOrientation(id)[0:2]
    rotation_matrix = p.getMatrixFromQuaternion(orn)

    ang = p.getEulerFromQuaternion(orn)[2]
    p_r_s = [self.rotate_translate_z(p1, ang,current_pos,rotation_matrix) for p1 in ori_pos]

    self.drawAABB(p_r_s)
    top = sorted(p_r_s, key=lambda x: x[2])[4:8]
    top = sorted(top, key=lambda x: x[0])
    mid = list(zip(top[0], top[1], top[2], top[3]))
    mid_point = [sum(mid[0])/4,sum(mid[1])/4,sum(mid[2])/4]

    # 4 lines
    lines = [(top[0],top[1]),(top[0],top[2]),(top[1],top[3]),(top[2],top[3])]
    # line lengths
    lines_l = sorted(lines, key=lambda x: dist(x[0], x[1]))
    l_l = lines_l[-1]
    theta = math.atan((l_l[1][1] - l_l[0][1])/(l_l[1][0] - l_l[0][0]))

    # self.drawAABB(self._object_box_points[id])
    # print(theta)
    grasp_pointa = get_mid_point(l_l[0], l_l[1])
    ray_length = 1
    beta = math.pi/2 - theta
    grasp_pointb = get_mid_point(lines_l[-2][0], lines_l[-2][1])
    a,b = self.extendLine(grasp_pointa,grasp_pointb)
    l1 = self._p.addUserDebugLine(a, b, [0, 1, 0])
    self._debug_lines.append(l1)
    return mid_point,beta

  # Angle perpenticula to the long line
  #def get_rect_angle(self, pos):
  def extendLine(self, p1, p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    mul = 0.1
    x3  = x2 + mul*(x2-x1)
    y3  = y2 + mul*(y2-y1)
    z3 =  z2 + mul*(z2 - z1)
    x4 =  x1 + mul*(x1 - x2)
    y4 =  y1 + mul*(y1 - y2)
    z4 =  z1 + mul*(z1 - z2)
    return [x3,y3,z3],[x4,y4,z4]
  def getAABBPoints(self,aabb):
    aabbMin = aabb[0]
    aabbMax = aabb[1]
    points_zip  = list(zip(aabbMin,aabbMax))
    return list(product(points_zip[0],points_zip[1],points_zip[2]))
  def remove_debug_lines(self):
    for lid in self._debug_lines:
      self._p.removeUserDebugItem(lid)
    self._debug_lines = []

  def drawAABB(self,points):
    #print("points", points, len(points))
    for i in range(0,len(points),2):
      f = [i for i in points[i]]
      t = [i for i in points[i+1]]
      l1=self._p.addUserDebugLine(f, t, [1, 1, 1])
      self._debug_lines.append(l1)
    # Top surface 4 points
    top = sorted(points, key=lambda x: x[2])[4:8]
    top = sorted(top, key=lambda x: x[0])

    f = [i for i in top[0]]
    t = [i for i in top[1]]
    l1 = self._p.addUserDebugLine(f, t, [1, 1, 1])
    self._debug_lines.append(l1)

    f = [i for i in top[0]]
    t = [i for i in top[2]]
    l1 =self._p.addUserDebugLine(f, t, [1, 1, 1])
    self._debug_lines.append(l1)

    f = [i for i in top[3]]
    t = [i for i in top[2]]
    l1 =self._p.addUserDebugLine(f, t, [1, 1, 1])
    self._debug_lines.append(l1)

    f = [i for i in top[3]]
    t = [i for i in top[1]]
    l1 =self._p.addUserDebugLine(f, t, [1, 1, 1])
    self._debug_lines.append(l1)

  def rotate_translate_z(self, point,q,t,matrix):
    # x,y = point[0],point[1]
    # x1 = x*math.cos(q) - y*math.sin(q)
    # y1 = x*math.sin(q) + y*math.cos(q)
    # return (x1+t[0],y1+t[1],point[2]+t[2])
    r = np.array(matrix).reshape(3,3)
    newp = r.dot(np.array(point))
    print(r)
    return (newp[0] + t[0], newp[1] + t[1], newp[2] + t[2])

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    #renderer=self._p.ER_TINY_RENDERER)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.03
    closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range(100):
        graspAction = [0, 0, 0.0001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if self._renders:
          time.sleep(self._timeStep)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(1000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        if self._renders:
          time.sleep(self._timeStep)
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):

    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                       self._kuka.kukaEndEffectorIndex)

    reward = -1000

    numPt = len(closestPoints)
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      reward = -closestPoints[0][8] * 10
    if (blockPos[2] > 0.2):
      reward = reward + 10000
      #print("successfully grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
    #print("reward")
    #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
