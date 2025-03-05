#!/usr/bin/env python3
"""
first terminal:
sr1
roscore

second termimal:
sr2
# ros2 launch realsense2_camera rs_camera.launch
ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=1280x720x30

thrid termial:
sr1
sr2
ros2 run ros1_bridge dynamic_bridge

(optional)
sr1
rqt_image_view /camera/color/image_raw
"""

# TODO:
# 1. [Done] Show camera frame in window and Press 's' in keyboard and save camera rgb image as file
# 2. When press 's' save arm joints as well

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
import numpy as np
from urdfpy import URDF
from urdfpy.utils import matrix_to_xyz_rpy
from typing import Tuple
import os, datetime

rospy.init_node("prepare_cam_calibration_data", anonymous=True)

rospy.loginfo("Starting...")


# 角点的个数以及棋盘格间距
XX = 11 #标定板的中长度对应的角点的个数
YY = 8  #标定板的中宽度对应的角点的个数
L = 0.02 #标定板一格的长度  单位为米]

def _find_chessboard_corners(gray: np.array, XX: int, YY:int, flags: int=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK
                                            +cv2.CALIB_CB_NORMALIZE_IMAGE
                                            , criteria: Tuple[int, int ,float]=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001), winsize: Tuple[int, int]=(11, 11)):
    ret, corners = cv2.findChessboardCorners(gray, (XX, YY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK
                                            +cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        # refining pixel coordinates for given 2d points
        corners2 = cv2.cornerSubPix(gray, corners, winsize, (-1, -1), criteria)
        return True, corners2
    else:
        return False, []
    
# Initialize the CVBridge class
cv_bridge = CvBridge()
save_idx = 0
current_joint_angles = None
current_ee_xyzrpy = None

import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
description_file_path = os.path.join(current_dir, "urdf", "rm_75_6f_description.urdf")
# robot_model = URDF.load('/home/xichen/Documenxts/repos/Luca_Catkin/src/realman/realman_example_control/urdf/rm_75_6f_description.urdf')
robot_model = URDF.load(description_file_path)

def show_and_save_image(img, img_draw, output_dir):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(3)
    if key == ord('s'):
        global save_idx
        image_save_path = f'{output_dir}/{save_idx}.png'
        rospy.loginfo(f'Saved image at {image_save_path}')
        cv2.imwrite(image_save_path, rgb_img)
        global current_joint_angles
        if current_joint_angles is not None:
            np.save(f'{output_dir}/{save_idx}_angles', np.array(current_joint_angles))
        if current_ee_xyzrpy is not None:
            # np.save(f'/home/xichen/Documents/repos/Luca_Catkin/src/output/{save_idx}_xyzrpy', np.array(current_ee_xyzrpy))
            np.save(f'{output_dir}/{save_idx}_xyzrpy', np.array(current_ee_xyzrpy))
        save_idx+=1

# define a callback for the Image message
def image_callback(img_msg):
    # log some info about the image topic
    # rospy.loginfo(img_msg.header)

    try:
        cv_image = cv_bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        
    
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    ret, corneres = _find_chessboard_corners(gray, XX, YY)
    img_draw = cv_image.copy()
    if ret:
        img_draw = cv2.drawChessboardCorners(cv_image.copy(), (XX, YY), corneres, ret)

    # cv2.imshow('draw', img_draw)'
    # cv2.waitKey(5)

    # Get the directory of the script's parent folder
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Generate a timestamp string
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # Construct the output directory path
    output_dir = os.path.join(current_dir, "output", timestamp)

    print(f"Output directory: {output_dir}")
    show_and_save_image(cv_image, img_draw, output_dir)

def joint_states_callback(msg: JointState):
    name = msg.name
    jointsAng = msg.position
    # print(jointsAng)
    global current_joint_angles 
    current_joint_angles = jointsAng

    # doing forward kinematic to get xyzrpy of links
    global robot_model
    if robot_model is not None:
        link_names = [x.name for x in robot_model.links]
        fk = robot_model.link_fk(cfg=
                              {
                                  "joint1": jointsAng[0],
                                  "joint2": jointsAng[1],
                                  "joint3": jointsAng[2],
                                  "joint4": jointsAng[3],
                                  "joint5": jointsAng[4],
                                  "joint6": jointsAng[5],
                                  "joint7": jointsAng[6],
                              })
        tf_link7 = fk[robot_model.links[-1]]
    #   print(f"link7 xyzrpy: {matrix_to_xyz_rpy(tf_link7)}")
        global current_ee_xyzrpy
        current_ee_xyzrpy = matrix_to_xyz_rpy(tf_link7)
      
    
sub_image = rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
sub_jointstate = rospy.Subscriber("/joint_states", JointState, callback=joint_states_callback, queue_size=1)
 
while not rospy.is_shutdown():
    rospy.spin()