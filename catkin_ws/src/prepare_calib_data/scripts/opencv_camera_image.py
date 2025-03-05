#!/usr/bin/env python3

"""roslaunch realsense2_camera rs_camera.launch"""
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

print("Hello")

rospy.init_node("opencv_camera_image", anonymous=True)

rospy.loginfo("Hello ROS CV!")

# Initialize the CVBridge class
cv_bridge = CvBridge()

# define a function to show the image in an opencv window
def show_image(img):
    cv2.imshow("Frame", img)
    # cv2.waitKey(3)
    key = cv2.waitKey(3)
    if key == ord('s'):
        cv2.imwrite('frame.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# define a callback for the Image message
def image_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo(img_msg.header)

    try:
        cv_image = cv_bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

    show_image(cv_image)

# Initialize a subscriber to the 'camera/color/image_raw' topic with the function "image_callback" as a callback
sub_image = rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

while not rospy.is_shutdown():
    rospy.spin()