import time
import random
from random import randint
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pdb
import pybullet as p
import pybullet_data
urdfRootPath=pybullet_data.getDataPath()
maxForce = 1000
maxVelocity = 3
serverMode = p.GUI # GUI/DIRECT
physicsClient = p.connect(serverMode)
p.setGravity(0,0,-30)
objects = p.loadSDF(os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
robotID = objects[0]
planeID = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), [0, 0, -1])
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
    jointPoses = p.calculateInverseKinematics(robotID, kukaEndEffectorIndex, target, [0,1,0,0])
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


def circular_path(center, t):
    from math import sin,cos
    R = 0.5
    #print([R*sin(t), 0, R*cos(t)])
    return [0.2 * cos(3*t), 0.5 , 0.2 * sin(3*t)+0.4 ]
    #return [x+y for x,y in zip(center, [R*sin(t), 0, R*cos(t)])]

random_target_timer = 0
target = [0.8,0.8,.5]
t = 0.01
prevPose = None
logid = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "~/Desktop/kuka_circle.mp4")
p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=180, cameraPitch=-20, cameraTargetPosition=[0,0,0])

for i in range(1000):
    ls = p.getLinkState(robotID, kukaEndEffectorIndex)
    t += 1./240.
    print(i)
    # if random_target_timer > 400:
    #     target = [0.3+random.uniform(-2,1),0.3+random.uniform(-2,1),0.3+random.uniform(0.5,2) ]
    #     random_target_timer = 0
    # else:
    #     random_target_timer += 1
    target = circular_path([0.8,0.8,0.5], t)
    if prevPose != None:
        p.addUserDebugLine(prevPose, ls[4], [0, 0, 0.3], 1, 60)
    prevPose = ls[4]
    p.stepSimulation()
    endEffectControl(target)
    time.sleep(1/240.)

p.stopStateLogging(logid)
# robotUrdfPath = "./urdf/sisbot.urdf"
# #robotUrdfPath = "./urdf/robotiq_c2.urdf"
# #robotUrdfPath = "./urdf/ur5.urdf"
# tablePath = "./urdf/table/table.urdf"


# tablePositon = [0,0.5,0]
# # define world


# table = p.loadURDF(tablePath,tablePositon)
# # define robot
# robotStartPos = [0,-0.2,1]
# robotStartOrn = p.getQuaternionFromEuler([0,0,0])
# print("----------------------------------------")
# print("Loading robot from {}".format(robotUrdfPath))
# robotID = p.loadURDF(robotUrdfPath, robotStartPos, robotStartOrn)
#
# # get joint information
# numJoints = p.getNumJoints(robotID)

# # get links
# linkIDs = list(map(lambda linkInfo: linkInfo[1], p.getVisualShapeData(robotID)))
# linkNum = len(linkIDs)
#
# # start simulation
# endEffectorIndex = 6
# counter = 0
# dir = 1
#
# endPosition = [1,1,1]
#
# sphereUid = p.createVisualShape(shapeType = p.GEOM_SPHERE,radius=0.05,rgbaColor = [1,0,0,2])
# collisionShapeId = p.createCollisionShape(
#     shapeType=p.GEOM_SPHERE,)
#
# multiBodyId = p.createMultiBody(
#     baseVisualShapeIndex=sphereUid,
#     basePosition=[1, 1, 1])
# print(sphereUid)
# for i in range(100000):
#     p.stepSimulation()
#     time.sleep(1/240.)
#     if counter > 500:
#         p.removeBody(multiBodyId)
#         #endPosition = [1,1, 2]
#         endPosition = [0.3*randint(-1,1),0.1+0.5*randint(0,1),0.9 + 0.1*randint(-1,1) ]
#         multiBodyId = p.createMultiBody(
#             baseVisualShapeIndex=sphereUid,
#             basePosition=endPosition)
#         counter = 0
#     else:
#
#         counter += 1
#
#     maxForce = 500
#     jointPoses = p.calculateInverseKinematics(robotID,
#                                               endEffectorIndex,
#                                               endPosition)
#     #print(jointPoses)
#     p.setJointMotorControlArray(bodyUniqueId=robotID,
#                             jointIndices=[1,2,3,4,5,6],
#                             controlMode=p.POSITION_CONTROL,
#                             positionGains = ([0.05]*6),
#                             targetPositions=jointPoses)


