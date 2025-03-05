from robot_cam_calibration.robot_handeye_opt_calibrator import HandEyeOptCalibrator
import numpy as np
from robot_cam_calibration.camera_intrinsic_params_calibrator import CameraIntrinsicParamsCalibrator

# # The configurations for camera calibration
# INTRINSIC_CALIB_DATA_DIR = "./data/demo_data/robot_cam_calibration_data_0906/handeye"
# # 角点的个数以及棋盘格间距
# XX = 11 #标定板的中长度对应的角点的个数
# YY = 8  #标定板的中宽度对应的角点的个数
# L = 0.02 #标定板一格的长度  单位为米
# calibrator = CameraIntrinsicParamsCalibrator(XX=XX, YY=YY, L=L)
# calibrator.load_intrinsic_calib_images(INTRINSIC_CALIB_DATA_DIR)
# calibrator.calibrate(verbose=True)
# calibrator.save('./output/calib_data/robot_cam_calibration_data_0906/intrinsic')


# import argparse
# TODO: use argparser here, to specfic input and output dir as cli parameters

HANDEYE_CALIB_DATA_DIR= "input" # /home/xichen/Documents/repos/Luca_Setup/data/output/calib_data/1011_d455_no2_intrinsic
HANDEYE_CALIB_RESULT_OUTPUT_DIR = "output" #'/home/xichen/Documents/repos/Luca_Setup/output/calib_data/0116_right_output/handeye_opt'


calibrator = HandEyeOptCalibrator(XX=11, YY=8, L=0.02)

extrinsic_calib_image_idxs = np.arange(0, 50)
excluded_image_idxs = []
extrinsic_calib_image_idxs = np.delete(extrinsic_calib_image_idxs, excluded_image_idxs)


calibrator.load_camera_intrinsics('data/output/calib_data/1011_d455_no2_intrinsic/intrinsic')
calibrator.load_calibration_images(HANDEYE_CALIB_DATA_DIR,
                                   extrinsic_calib_image_idxs)
calibrator.load_calibration_ee_xyzrpy(HANDEYE_CALIB_DATA_DIR, extrinsic_calib_image_idxs)
calibrator.calibrate()
# ensure the intrinsics if correct
cam_calib_errors = calibrator.cam_calib_reproject_error(vis=False)

# errors = calibrator.handeye_calib_reproject_error(vis=True)
# # find errors > 1
# extrinsic_calib_image_idxs = extrinsic_calib_image_idxs[errors<1]
calibrator.load_calibration_images(HANDEYE_CALIB_DATA_DIR, extrinsic_calib_image_idxs)
calibrator.load_calibration_ee_xyzrpy(HANDEYE_CALIB_DATA_DIR, extrinsic_calib_image_idxs)
calibrator.calibrate()
cam_calib_errors = calibrator.handeye_calib_reproject_error(vis=False)
argmin_error_idx = np.argmin(cam_calib_errors)
print(f'argmin_error_idx: {argmin_error_idx}')

# calibrator.load_initial_eyehand_calibration_result('./output/calib_data/handeye')
calibrator.set_initial_imageid(argmin_error_idx)
calibrator.optmize()
errors = calibrator.opt_handeye_calib_reproject_error(vis=True)
calibrator.save()

# calibrator.handeye_calib_reproject_error(vis=True)
print(np.array(errors).mean())
# calibrator.print_result()