import pybullet as p
import pybullet_data as pd
import time
p.connect(p.GUI)
path = "/home/yingfu/Documents/gym/pybullet_data/"

useFixedBase = True
flags = p.URDF_INITIALIZE_SAT_FEATURES#0#p.URDF_USE_SELF_COLLISION

#plane_pos = [0,0,0]
#plane = p.loadURDF("plane.urdf", plane_pos, flags = flags, useFixedBase=useFixedBase)
table_pos = [0,0,-0.625]
#table = p.loadURDF("table/table.urdf", table_pos, flags = flags, useFixedBase=useFixedBase)
#object = p.loadSDF(path + "gripper/wsg50_with_r2d2_gripper.sdf")
object = p.loadSDF(path + "kuka_iiwa/kuka_with_gripper.sdf")
kukaUid = object[0]
a = 0
l = -0.1
while (1):
	numJoints = p.getNumJoints(kukaUid)
	print(numJoints)
	p.setJointMotorControl2(kukaUid,
							10,
							p.POSITION_CONTROL,
							targetPosition=-1,
							force=300)
	p.setJointMotorControl2(kukaUid,
							8,
							p.POSITION_CONTROL,
							targetPosition=1,
							force=500)
	p.setJointMotorControl2(kukaUid,
							7,
							p.POSITION_CONTROL,
							targetPosition=a,
							force=500)
	a += 0.1
	for i in range(20):
		p.stepSimulation()
		time.sleep(1./240.)
	
