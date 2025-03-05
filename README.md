# robot_cam_calibration

## Installation

Ensure you have python 3.8 Installed. 
1. Install ROS-noetic
Follow the official ROS Noetic installation guide:
[ROS Noetic Installation Guide](http://wiki.ros.org/noetic/Installation/Ubuntu)

2. Then install the dependent python packages via:
    ```
    python -m pip install -r requirements.txt
    ```
3. pull the Git Submodule by 
    ```
    git submodule update --init --recursive
    ```
## Setup
- Robot Base (robot_base): Defined in URDF as the base of the robotic arm.

- End Effector (ee): The last transformed attached to the robot arm, named joint7.

- Chessboard (target): Mounted on the robot's arm.

- Camera (cam): Mounted externally, fixed in a rigid position relative to the robot.

## Usage Guide

### 1. Collect Calibration Data
This step gathers images from a RealSense D455 camera and robot joint angles(DoF) for calibration.

Example Usage:
1. Make sure you camera is connected physically. You can use command ```cheese``` in terminal to check it visually.
2. Make sure Realman robot arm is powered, and laucnh the driver ROS package by ```roslaunch rm_driver rm_75_driver.launch```
3. Run the calibration data collection python ros package `prepare_cam_calibration_data.py`

Ensure your robot is positioned in different poses during data collection.

### 2. Compute Robot-Eye Tranformation Matrix
This step calculates the transformation matrix between the robot base, end-effector, and camera using collected data.
```
python scripts/run_eyehand_opt_calibration.py
```
#### Output

4 .npy files and a txt file

- **`cam2base_4x4.npy`** - The optimized camera-to-robot base 4×4 matrix.
- **`target2ee_4x4.npy`** - The optimized calibration board-to-end effector 4×4 matrix.
- **`init_cam2base_4x4.npy`** - Initial camera-to-robot base transformation (computed with OpenCV, no optimization).
- **`init_target2ee_4x4.npy`** - Initial calibration board-to-end effector transformation (computed with OpenCV, no optimization).
- **`reprojection_error.txt`** - Records the mean reprojection error in pixels.

Each `.npy` file contains a 4×4 transformation matrix stored in NumPy format.
