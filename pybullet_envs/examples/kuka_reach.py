import time
import math
import numpy as np
import random
from math import sin, cos
from random import randint
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pdb
import pybullet as p
import pybullet_data
urdfRootPath=pybullet_data.getDataPath()
maxForce = 500
maxVelocity = 0.2

fingerAForce = 1
fingerBForce = 1.5
fingerTipForce = 1
serverMode = p.GUI # GUI/DIRECT
physicsClient = p.connect(serverMode)
p.setGravity(0,0,-10)
objects = p.loadSDF(os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
robotID = objects[0]
planeID = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), [0, 0, 0])
# cubeID = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), [0.6, 0.05, 0])
kukaEndEffectorIndex = 6
kukaGripperIndex = 7

numJoints = p.getNumJoints(robotID)
print("Number of joints: {}".format(numJoints))
jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
for i in range(numJoints):
    jointInfo = p.getJointInfo(robotID, i)
    jointID = jointInfo[0]
    jointName = jointInfo[1].decode("utf-8")
    jointType = jointTypeList[jointInfo[2]]
    jointLowerLimit = jointInfo[8]
    jointUpperLimit = jointInfo[9]
    print("\tID: {}".format(jointID))
    print("\tname: {}".format(jointName))
    print("\ttype: {}".format(jointType))
    print("\tlower limit: {}".format(jointLowerLimit))
    print("\tupper limit: {}".format(jointUpperLimit))
print("------------------------------------------")


def reset():

    p.resetBasePositionAndOrientation(robotID, [-0.100000, 0.000000, 0.070000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])
    jointPositions = [
        0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
    ]
    numJoints = p.getNumJoints(robotID)
    for jointIndex in range(numJoints):
        p.resetJointState(robotID, jointIndex, jointPositions[jointIndex])
        p.setJointMotorControl2(robotID,
                                jointIndex,
                                p.POSITION_CONTROL,
                                targetPosition=jointPositions[jointIndex],
                                force=maxForce)

# Control end effector position via POSITION_CONTROL
def endEffectControl(target):
    # use standard IK algorithm to calculate poses
    jointPoses = p.calculateInverseKinematics(robotID, kukaEndEffectorIndex, target,[0,1,0,0])
    for i in range(kukaEndEffectorIndex + 1):
        # print(i)
        p.setJointMotorControl2(bodyUniqueId=robotID,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=maxForce,
                                maxVelocity=maxVelocity,
                                positionGain=0.3,
                                velocityGain=1)

def gripperControl(fingerAngle):
    p.setJointMotorControl2(robotID,
                            8,
                            p.POSITION_CONTROL,
                            targetPosition=-fingerAngle,
                            force=fingerAForce)
    p.setJointMotorControl2(robotID,
                            11,
                            p.POSITION_CONTROL,
                            targetPosition=fingerAngle,
                            force=fingerBForce)

    p.setJointMotorControl2(robotID,
                            10,
                            p.POSITION_CONTROL,
                            targetPosition=0,
                            force=fingerTipForce)
    p.setJointMotorControl2(robotID,
                            13,
                            p.POSITION_CONTROL,
                            targetPosition=0,
                            force=fingerTipForce)


def circular_path(center, t):

    R = 0.5
    print([R*sin(t), 0, R*cos(t)])
    return [0.2 * cos(2*t), 0.5 , 0.2 * sin(2*t)+0.4 ]
    #return [x+y for x,y in zip(center, [R*sin(t), 0, R*cos(t)])]

random_target_timer = 0
t = 0.01
prevPose = None
reset()
gripperControl(0.6)
ls = p.getLinkState(robotID, kukaEndEffectorIndex+1)

endEffectControl((0.6, 0, 0.5))

cube1ID = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), [0.5, -0.25, 0])
#cube2ID = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), [0.52, -0.15, 0])
sphereID = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"), [0.52, -0.43, 0])
trayID = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"), [0.52, 0.5, 0])
p.setRealTimeSimulation(1)
j = 0
targetPos = (([0.53, -0.25, 0.50], 0.6, 0.6),([0.53, -0.28, 0.30], 0.1, 0.6),([0.6, 0.4, 0.5], 0.1, 0.1),([0.6, 0.5, 0.5], 0.6, 0.1) ,([0.52, -0.45, 0.5],0.6,0.6),([0.52, -0.46, 0.29],0.1,0.6), ([0.6, 0.4, 0.5],0.6,0.1),([0.6, 0, 0.5],0.6,0.6))
#endEffectControl(targetPos[j])
mode = "move"


def closeTo(target, orientation):
    ls = p.getLinkState(robotID, kukaEndEffectorIndex)
    print(ls[4], ls[5])
    dis = sum(abs(np.array(ls[4]) - target))+sum(abs(np.array(ls[5]) - orientation))
    print(dis)
    if dis < 0.0005:
        return True
    else:
        return False

def moveto(target,j):
    endEffectControl(target)
    gripperControl(targetPos[j][2])
    if closeTo(target, [0,1,0,0]):
        t = 0
        return "grab",j
    else:
        return "move",j
def grab(angle, j,t):
    gripperControl(angle)
    if t > 1:
        return "move", j+1
        t = 0
    else:
        return "grab", j


def indirect(mode,j):
    switcher={
            "move":lambda:moveto(targetPos[j][0],j),
            "grab":lambda:grab(targetPos[j][1], j,t),
            }
    return switcher.get(mode,lambda :'Invalid')()
#logid = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "~/Desktop/vid1.mp4")
p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0,0,0])

for i in range(100000):
    mode,j = indirect(mode, j)
    if j > 7:
        break
    t += 1./240.
    time.sleep(1 / 240.)
#p.stopStateLogging(logid)
