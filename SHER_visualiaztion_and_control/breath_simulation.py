# For breath simulation
# !/usr/bin/env python
from __future__ import print_function
import cv2 # for saving/loading images - see online page for details
import time # for accessing current time
from PIL import Image # apparently the fastest tool to save images
import rospy # for writing a ROS node
from geometry_msgs.msg import Vector3, Transform
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import csv
import torch
from torchvision import models
from image_conversion_without_using_ros import image_to_numpy
import numpy as np
import torchvision.transforms.functional as TF
from ultralytics import YOLO
import socket
import struct
import torch.nn as nn
import os
from scipy.spatial.transform import Rotation as R
import math

vel_x = None
vel_y = None
vel_z = None
breath_simuation_b_scan_raw = None
x = None
y = None
z = None
rx = None
ry = None
rz = None
rw = None

class image_converter:

  def __init__(self):
    self.velocity_sub = rospy.Subscriber('/eyerobot2/desiredTipVelocities', Vector3, self.get_tip_velocity)
    self.breath_simulation_b_scan_sub = rospy.Subscriber("/breath_simulation_b_scan", Image, self.get_breath_simulation_b_scan)
    self.tool_tip_position_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.get_tool_tip_position)

  def get_tip_velocity(self, data):
    global vel_x
    global vel_y
    global vel_z
    vel_x = data.x
    vel_y = data.y
    vel_z = data.z

  def get_breath_simulation_b_scan(self, data):
    global breath_simuation_b_scan_raw
    breath_simuation_b_scan_raw = data

  def get_tool_tip_position(self, data):
    global x
    global y
    global z
    global rx
    global ry
    global rz
    global rw
    x = data.translation.x
    y = data.translation.y
    z = data.translation.z
    rx = data.rotation.x
    ry = data.rotation.y
    rz = data.rotation.z
    rw = data.rotation.w

# for publishing the clicked goal position
def create_publisher():
  """
      Usage:
        publisher = create_publisher()
        publisher.publish(desired_position[0], desired_position[1], desired_position[2])
  """
  pub_tip_vel = rospy.Publisher('/eyerobot2/desiredTipVelocities', Vector3, queue_size = 3)
  pub_tip_vel_angular = rospy.Publisher('/eyerobot2/desiredTipVelocitiesAngular', Vector3, queue_size = 3)
  pub_rcm_point = rospy.Publisher('/rcm_point', Vector3, queue_size = 3)
  return pub_tip_vel, pub_tip_vel_angular, pub_rcm_point

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

##############################################
# MAIN CODE STARTS HERE
#############################################
if __name__ == "__main__":
    ic = image_converter()
    # for sending clicked goal position over ros node
    pub_tip_vel, pub_tip_vel_angular, pub_rcm_point = create_publisher()
    rospy.init_node('image_converter', anonymous=True)
    time.sleep(2)
    b_scan_breath_simulation_path = "../visualization/breath_simulation_{:10.2f}".format(time.time())
    if not os.path.exists(b_scan_breath_simulation_path):
        os.makedirs(b_scan_breath_simulation_path)
    # set RCM point
    rcm_point = None
    while rcm_point is None:
        rcm_point = np.array((x, y, z))
    print("rcm_point = ", rcm_point)
    input("Check if you have left the RCM point using Cannulation mode and press Enter to start navigation...")
    while (breath_simuation_b_scan_raw is None):
        continue
    count = 0
    breath_simuation_b_scan_cur = image_to_numpy(breath_simuation_b_scan_raw)
    time_stamp = time.time()
    cv2.imwrite(os.path.join(b_scan_breath_simulation_path, "b_scans_breath_simulation_{:10.2f}_{}.jpg".format(time_stamp, 1)), breath_simuation_b_scan_cur)
    b_scan_cur = cv2.imread(os.path.join(b_scan_breath_simulation_path, "b_scans_breath_simulation_{:10.2f}_{}.jpg".format(time_stamp, 1)))
    b_scan_cur = cv2.resize(b_scan_cur, (512, 512))
    b_scan_cur = cv2.cvtColor(b_scan_cur, cv2.COLOR_BGR2GRAY)

    kp_linear_vel = 1.5 # used to be 2
    kp_angular_vel = 1
    diff_angle_thresh = 0.1 * math.pi / 180
    feature_params = dict(maxCorners = 300, qualityLevel = 0.01, minDistance = 2, blockSize = 7)
    lk_params = dict(winSize = (10,10), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    target_point = np.array((x, y, z))
    current_quat = np.array((rx, ry, rz, rw))
    rcm_to_target_point = rcm_point - target_point
    dx = rcm_to_target_point[0]
    dy = rcm_to_target_point[1]
    dz = rcm_to_target_point[2]
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    theta_y = np.arctan2(dx, dz)
    theta_x = np.arcsin(-dy / magnitude)
    desired_euler_angle = np.array((theta_x, theta_y, 0))
    r_rcm_to_target = R.from_euler("xyz", desired_euler_angle)
    rotation_matrix_rcm_to_target = r_rcm_to_target.as_matrix()
    r_current = R.from_quat(current_quat)
    rotation_matrix_current = r_current.as_matrix()
    error_angular_rotation_matrix = np.matmul(rotation_matrix_rcm_to_target, np.transpose(rotation_matrix_current))
    # error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_target)
    r_error = R.from_matrix(error_angular_rotation_matrix)
    error_rotation_vector = r_error.as_rotvec()
    unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)

    linear_vel_gain = 0.08
    angular_vel_gain = 0.08

    while (np.linalg.norm(error_rotation_vector) >= 0.03):
        angular_vel = unit_error_rotation_vector * angular_vel_gain
        current_quat = np.array((rx, ry, rz, rw))
        r_current = R.from_quat(current_quat)
        rotation_matrix_current = r_current.as_matrix()
        error_angular_rotation_matrix = np.matmul(rotation_matrix_rcm_to_target, np.transpose(rotation_matrix_current))
        # error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_target)
        r_error = R.from_matrix(error_angular_rotation_matrix)
        error_rotation_vector = r_error.as_rotvec()
        unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
        print("Angular ERROR: ", np.linalg.norm(error_rotation_vector))
        # publish the velocities
        pub_tip_vel_angular.publish(angular_vel[0], angular_vel[1], angular_vel[2])

    pub_tip_vel.publish(0, 0, 0)
    pub_tip_vel_angular.publish(0, 0, 0)

    input("Press Enter to start breath simulation")
    while not rospy.is_shutdown():
        if breath_simuation_b_scan_raw is not None:
            # optical flow calculation
            p0 = cv2.goodFeaturesToTrack(b_scan_cur, mask = None, **feature_params)
            breath_simuation_b_scan_next = image_to_numpy(breath_simuation_b_scan_raw)
            cv2.imwrite(os.path.join(b_scan_breath_simulation_path, "b_scans_breath_simulation_{:10.2f}_{}.jpg".format(time_stamp, 2)), breath_simuation_b_scan_next)
            time.sleep(0.005)
            b_scan_next = cv2.imread(os.path.join(b_scan_breath_simulation_path, "b_scans_breath_simulation_{:10.2f}_{}.jpg".format(time_stamp, 2)))
            b_scan_next = cv2.resize(b_scan_next, (512, 512))
            b_scan_next = cv2.cvtColor(b_scan_next, cv2.COLOR_BGR2GRAY)
            p1, status, error = cv2.calcOpticalFlowPyrLK(b_scan_cur, b_scan_next, p0, None, **lk_params)

            good_new = p1[status == 1]
            good_old = p0[status == 1]
            moving_distance = np.median(good_new - good_old, axis = 0) * 3.379 / 1024
            print("moving distance = ", moving_distance)

            cur_z_position = z
            goal_position = np.array((x, y, z - moving_distance[1]))
            # moving the robot along the breath pattern
            if moving_distance[1] == 0:
               moving_distance[1] = 1
            linear_vel = abs(np.median(good_new - good_old, axis = 0)[1]) * 0.06 * moving_distance[1] / abs(moving_distance[1])
            send_linear_velocity = kp_linear_vel * linear_vel
            if (abs(cur_z_position - goal_position[2]) < 0.005):
                pub_tip_vel.publish(0, 0, 0)
            else:
                pub_tip_vel.publish(0, 0, -send_linear_velocity)

            # Track angular velocity
            current_quat = np.array((rx, ry, rz, rw))
            r_current = R.from_quat(current_quat)
            rotation_matrix_current = r_current.as_matrix()
            rcm_to_goal_position = np.array((rcm_point[0], rcm_point[1], rcm_point[2])) - goal_position
            dx = rcm_to_goal_position[0]
            dy = rcm_to_goal_position[1]
            dz = rcm_to_goal_position[2]
            h1 = np.sqrt(dz**2 + dy**2)
            h2 = np.sqrt(dz**2 + dx**2)
            magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
            theta_x = np.arcsin(-dy / magnitude)
            theta_y = np.arctan2(dx, dz)
            desired_euler_angle = np.array((theta_x, theta_y, 0))
            # convert euler angle to quaternion
            # r_temp = R.from_euler("XYZ", desired_euler_angle)
            r_temp = R.from_euler("xyz", desired_euler_angle)
            rotation_matrix_rcm_to_desired = r_temp.as_matrix()
            error_angular_rotation_matrix = np.matmul(rotation_matrix_rcm_to_desired, np.transpose(rotation_matrix_current))
            r_error = R.from_matrix(error_angular_rotation_matrix)
            error_rotation_vector = r_error.as_rotvec()
            diff_to_final_angle = np.linalg.norm(error_rotation_vector)
            if diff_to_final_angle < diff_angle_thresh:
              pub_tip_vel_angular.publish(0, 0, 0)
              continue
            # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
            unit_error_rotation_vector = error_rotation_vector
            angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)

            # Track angular vel w/o normalization and also using desired angular velocity
            send_angular_velocity = angular_velocity
            pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)

            cv2.imwrite(os.path.join(b_scan_breath_simulation_path, "b_scans_breath_simulation_{:10.2f}_{}.jpg".format(time_stamp, 1)), breath_simuation_b_scan_next)
            time.sleep(0.005)
            b_scan_cur = cv2.imread(os.path.join(b_scan_breath_simulation_path, "b_scans_breath_simulation_{:10.2f}_{}.jpg".format(time_stamp, 1)))
            b_scan_cur = cv2.resize(b_scan_cur, (512, 512))
            b_scan_cur = cv2.cvtColor(b_scan_cur, cv2.COLOR_BGR2GRAY)
    pub_tip_vel.publish(0, 0, 0)
    pub_tip_vel_angular.publish(0, 0, 0)
    time.sleep(0.8)
    pub_tip_vel.publish(0, 0, 0)
    pub_tip_vel_angular.publish(0, 0, 0)