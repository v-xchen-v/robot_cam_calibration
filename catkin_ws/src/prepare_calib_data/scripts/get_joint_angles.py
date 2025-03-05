#!/usr/bin/env python3
import os, sys
# pip install numpy==1.23.1
# sys.path.append("/home/xichen/Documents/repos/Luca_Catkin/src")
"""roslaunch rm_driver rm_75_driver.launch"""
"""
rostopic echo /joint_states
---
header: 
  seq: 128494
  stamp: 
    secs: 1721632769
    nsecs: 522392034
  frame_id: ''
name: 
  - joint1
  - joint2
  - joint3
  - joint4
  - joint5
  - joint6
  - joint7
position: [1.3857220000000003, 1.4104104, 0.0, 0.0, 0.0, 0.0, 0.0]
velocity: []
effort: []
---
rostopic hz /joint_states # ~150hz
"""

import rospy
from sensor_msgs.msg import JointState
import threading
from urdfpy import URDF
from urdfpy.utils import matrix_to_xyz_rpy
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

# class RMExampleControl:
#     def __init__(self):
#         # for joint states
#         self.jointsAng = []
#         self.jointsVel = []
#         self.thread = threading.Thread(target=self.joint_states_listener)
#         self.lock = threading.Lock()
#         # end for joint states

#         # forward kinematic
#         self.robot_model = URDF.load('/home/xichen/Documents/repos/Luca_Catkin/src/realman_inspire_R_description/urdf/Realman_Inspire_R.urdf')
#         print("robot loaded")
#         self.thread.start()
        
#     def joint_states_listener(self):
#         rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
#         rospy.spin()

#     def joint_states_callback(self, msg: JointState):
#         # self.lock.acquire()
#         self.name = msg.name
#         self.jointsAng = msg.position
#         print(self.jointsAng)

#         # doing forward kinematic to get xyzrpy of links
#         if self.robot_model is not None:
#           fk = self.robot_model.link_fk(cfg=
#                                   {
#                                       "joint1": self.jointsAng[0],
#                                       "joint2": self.jointsAng[1],
#                                       "joint3": self.jointsAng[2],
#                                       "joint4": self.jointsAng[3],
#                                       "joint5": self.jointsAng[4],
#                                       "joint6": self.jointsAng[5],
#                                       "joint7": self.jointsAng[6],
#                                   })
#           tf_link7 = fk[self.robot_model.links[-1]]
#           # print(f"link7 xyzrpy: {matrix_to_xyz_rpy(tf_link7)}")
          
#           plt.plot(self.jointsAng)
#           plt.draw()
#           plt.pause(0.0001)
#           plt.clf()
        
#         self.jointsVel = msg.velocity
        # self.lock.release()

def joint_states_callback(msg: JointState):
    # self.lock.acquire()
    name = msg.name
    jointsAng = msg.position
    print(jointsAng)

    # doing forward kinematic to get xyzrpy of links
    # if robot_model is not None:
    #   fk = robot_model.link_fk(cfg=
    #                           {
    #                               "joint1": jointsAng[0],
    #                               "joint2": jointsAng[1],
    #                               "joint3": jointsAng[2],
    #                               "joint4": jointsAng[3],
    #                               "joint5": jointsAng[4],
    #                               "joint6": jointsAng[5],
    #                               "joint7": jointsAng[6],
    #                           })
    #   tf_link7 = fk[robot_model.links[-1]]
      # print(f"link7 xyzrpy: {matrix_to_xyz_rpy(tf_link7)}")
      
    plt.plot(np.array(jointsAng)*180.0/np.pi)
    plt.ylim(-360, 360)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
    
    jointsVel = msg.velocity

if __name__ == "__main__":
    rospy.init_node("rm_example_control")
    # realman = RMExampleControl()
    rospy.Subscriber("/joint_states", JointState, callback=joint_states_callback, queue_size=1)

    while not rospy.is_shutdown():
        rospy.spin()