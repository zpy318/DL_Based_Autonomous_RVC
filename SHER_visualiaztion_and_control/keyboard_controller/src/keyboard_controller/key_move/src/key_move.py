#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import cv2 
import os 
import time 
from datetime import datetime
from PIL import Image 
import shutil # to be able to delete entire directories
import csv 
import rospy 
from geometry_msgs.msg import Vector3, Transform
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray
import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from key_publisher_msgs.msg import key_publisher_msgs
import math

x = None
y = None
z = None
rx = None
ry = None
rz = None
rw = None
# # for camera recording
# ddp_results = None
# # for clicking goal points and publishing over ros
# indicator_x = None
# network_output = None
# rcm_point = None

move_left_flag = False
move_right_flag = False
move_up_flag = False
move_down_flag = False
move_upward_flag = False
move_downward_flag = False
move_forward_flag = False
move_backward_flag = False
move_angle_insertion_flag = False
move_angle_retraction_flag = False
large_gain_flag = False
small_gain_flag = False
small_push_flag = False

class move:
  def __init__(self):
    self.key_sub = rospy.Subscriber("/key_input", key_publisher_msgs, self.get_key_input)
    self.tool_tip_position_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.get_tool_tip_position)

  def get_key_input(self, data):
    global move_left_flag
    global move_right_flag
    global move_up_flag
    global move_down_flag
    global move_upward_flag
    global move_downward_flag
    global move_forward_flag
    global move_backward_flag
    global move_angle_insertion_flag
    global move_angle_retraction_flag
    global large_gain_flag
    global small_gain_flag
    global small_push_flag


    move_left_flag = data.move_left
    move_right_flag = data.move_right
    move_up_flag = data.move_up
    move_down_flag = data.move_down
    move_upward_flag = data.move_upward
    move_downward_flag = data.move_downward
    move_forward_flag = data.move_forward
    move_backward_flag = data.move_backward
    move_angle_insertion_flag = data.move_angle_insertion
    move_angle_retraction_flag = data.move_angle_retraction
    large_gain_flag = data.large_gain
    small_gain_flag = data.small_gain
    small_push_flag = data.small_push

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

# class ddp_tracking:

#   def __init__(self):
#     self.ddp_results_sub = rospy.Subscriber("/ddp_output", ddp_output_msgs, self.get_ddp_results)
#     self.indicator_sub = rospy.Subscriber("/CNN/DesiredTipPosition",  Vector3, self.get_indicator)
#     self.tool_tip_position_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.get_tool_tip_position)
#     self.network_output_sub = rospy.Subscriber("/network_output", Vector3, self.get_network_output)
#     self.rcm_point_sub = rospy.Subscriber("/ddp_input", ddp_input_msgs, self.get_rcm_point)

#   def get_ddp_results(self, data):
#     global ddp_results
#     ddp_results = data.ddp_traj
#     # usb_image = bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')

#   def get_indicator(self, data):
#     global indicator_x
#     indicator_x = data.x
  
#   def get_tool_tip_position(self, data):
#     global x
#     global y
#     global z
#     global rx
#     global ry
#     global rz
#     global rw

#     x = data.translation.x
#     y = data.translation.y
#     z = data.translation.z
#     rx = data.rotation.x
#     ry = data.rotation.y
#     rz = data.rotation.z
#     rw = data.rotation.w
  
#   def get_network_output(self, data):
#     global network_output
#     network_output = data
  
#   def get_rcm_point(self, data):
#     global rcm_point
#     rcm_point = data.rcm_point

# for publishing the clicked goal position
def create_publisher():
  """
      Usage:
        publisher = create_publisher()
        publisher.publish(desired_position[0], desired_position[1], desired_position[2])
  """
  pub_tip_vel = rospy.Publisher('/eyerobot2/desiredTipVelocities', Vector3, queue_size = 3)
  pub_tip_vel_angular = rospy.Publisher('/eyerobot2/desiredTipVelocitiesAngular', Vector3, queue_size = 3)
  return pub_tip_vel, pub_tip_vel_angular

if __name__ == '__main__':
  move()
  pub_tip_vel, pub_tip_vel_angular = create_publisher()
  rospy.init_node('move', anonymous=True)
  rate = rospy.Rate(100) # 100hz
  kp_linear_vel = 2 # used to be 1.5 01/26/2023
  kp_angular_vel = 3 # 1.0, 2, 0.005, 1.5, 0.015
  move_angle = 20 * math.pi / 180 # set the tool-tip bended angle default 20
  linear_vel = 0.1
  # top_k = 10
  # desired_index = 1
  diff_angle_thresh = 0.1 * math.pi / 180
  ### change it each time!
  rcm_point = np.array((15.077, -59.023, -9.501))
  target_point = np.array((x + 0.1, y + 0.1, z + 0.1))

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
  rotation_matrix_rcm_to_target = r_rcm_to_target.as_dcm()
  r_current = R.from_quat(current_quat)
  rotation_matrix_current = r_current.as_dcm()
  error_angular_rotation_matrix = np.matmul(rotation_matrix_rcm_to_target, np.transpose(rotation_matrix_current))
  # error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_target)
  r_error = R.from_dcm(error_angular_rotation_matrix)
  error_rotation_vector = r_error.as_rotvec()
  unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)

  linear_vel_gain = 0.08
  angular_vel_gain = 0.08

  while (np.linalg.norm(error_rotation_vector) >= 0.03):
      angular_vel = unit_error_rotation_vector * angular_vel_gain
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      error_angular_rotation_matrix = np.matmul(rotation_matrix_rcm_to_target, np.transpose(rotation_matrix_current))
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      print("Angular ERROR: ", np.linalg.norm(error_rotation_vector))
      # publish the velocities
      pub_tip_vel_angular.publish(angular_vel[0], angular_vel[1], angular_vel[2])

  pub_tip_vel.publish(0, 0, 0)
  pub_tip_vel_angular.publish(0, 0, 0)

  print("You can control the robot now!")
  
  while not rospy.is_shutdown():
    # if (indicator_x == 1 and ddp_results != None and network_output != None):
    if (large_gain_flag):
      kp_linear_vel = 8 # 0.5, 1, 0.01, 0.6, 0.03
      kp_angular_vel = 6.5 # 1.0, 2, 0.005, 1.5, 0.015
      linear_vel = 0.1
    elif (small_gain_flag):
      kp_linear_vel = 2 # usded to be 1 01/26/2023
      kp_angular_vel = 3 # 1.0, 2, 0.005, 1.5, 0.015
      linear_vel = 0.1
    if (move_left_flag):
      send_linear_velocity = kp_linear_vel * linear_vel
      pub_tip_vel.publish(0, send_linear_velocity, 0)
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y + send_linear_velocity * 0.3, z))
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_left_flag = False
      rate.sleep()
    elif (move_right_flag):
      send_linear_velocity = kp_linear_vel * linear_vel
      pub_tip_vel.publish(0, -send_linear_velocity, 0)
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y - send_linear_velocity * 0.3, z))
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_right_flag = False
      rate.sleep()
    elif (move_up_flag):
      send_linear_velocity = kp_linear_vel * linear_vel
      pub_tip_vel.publish(send_linear_velocity, 0, 0)
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x + send_linear_velocity * 0.3, y, z))
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_up_flag = False
      rate.sleep()
    elif (move_down_flag):
      send_linear_velocity = kp_linear_vel * linear_vel
      pub_tip_vel.publish(-send_linear_velocity, 0, 0)
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x - send_linear_velocity * 0.3, y, z))
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_down_flag = False
      rate.sleep()
    elif (move_upward_flag):
      send_linear_velocity = kp_linear_vel * linear_vel
      pub_tip_vel.publish(0, 0, send_linear_velocity)
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y, z + send_linear_velocity * 0.3))
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_upward_flag = False
      rate.sleep()
    elif (move_downward_flag):
      send_linear_velocity = kp_linear_vel * linear_vel
      pub_tip_vel.publish(0, 0, -send_linear_velocity)
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y, z - send_linear_velocity * 0.3))
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_downward_flag = False
      rate.sleep()
    elif (move_forward_flag):
      # send_linear_velocity = kp_linear_vel * linear_vel
      current_position = np.array((x, y, z))
      rcm_to_current_vector = current_position - rcm_point
      # for 15 degree bended tool
      # linear_vel_vector = np.array((rcm_to_current_vector[0], math.cos(15 * math.pi / 180) * rcm_to_current_vector[1] - math.sin(15 * math.pi / 180) * rcm_to_current_vector[2], \
      #                               math.sin(15 * math.pi / 180) * rcm_to_current_vector[1] + math.cos(15 * math.pi / 180) * rcm_to_current_vector[2]))
      linear_vel_vector = np.array((rcm_to_current_vector[0], rcm_to_current_vector[1], rcm_to_current_vector[2]))
      unit_linear_vel_vector = linear_vel_vector / np.linalg.norm(linear_vel_vector)
      send_linear_velocity = kp_linear_vel * linear_vel * unit_linear_vel_vector
      pub_tip_vel.publish(send_linear_velocity[0], send_linear_velocity[1], send_linear_velocity[2]) # need to change
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y, z)) + send_linear_velocity * 0.3
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_forward_flag = False
      rate.sleep()
    elif (move_backward_flag):
      current_position = np.array((x, y, z))
      rcm_to_current_vector = current_position - rcm_point
      # for 15 degree bended tool
      # linear_vel_vector = np.array((rcm_to_current_vector[0], math.cos(15 * math.pi / 180) * rcm_to_current_vector[1] - math.sin(15 * math.pi / 180) * rcm_to_current_vector[2], \
      #                               math.sin(15 * math.pi / 180) * rcm_to_current_vector[1] + math.cos(15 * math.pi / 180) * rcm_to_current_vector[2]))
      linear_vel_vector = np.array((rcm_to_current_vector[0], rcm_to_current_vector[1], rcm_to_current_vector[2]))
      unit_linear_vel_vector = linear_vel_vector / np.linalg.norm(linear_vel_vector)
      send_linear_velocity = kp_linear_vel * linear_vel * unit_linear_vel_vector
      pub_tip_vel.publish(-send_linear_velocity[0], -send_linear_velocity[1], -send_linear_velocity[2]) # need to change
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y, z)) - send_linear_velocity * 0.3
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_backward_flag = False
      rate.sleep()
    elif (move_angle_insertion_flag):
      # send_linear_velocity = kp_linear_vel * linear_vel
      current_position = np.array((x, y, z))
      rcm_to_current_vector = current_position - rcm_point
      xy_sqrt = math.sqrt(rcm_to_current_vector[0]**2 + rcm_to_current_vector[1]**2)
      # for certain degree bended tool, define the value above
      linear_vel_vector = np.array((rcm_to_current_vector[0] * ((math.cos(move_angle) * xy_sqrt - math.sin(move_angle) * rcm_to_current_vector[2]) / xy_sqrt), \
                                    rcm_to_current_vector[1] * ((math.cos(move_angle) * xy_sqrt - math.sin(move_angle) * rcm_to_current_vector[2]) / xy_sqrt), \
                                    math.sin(move_angle) * xy_sqrt + math.cos(move_angle) * rcm_to_current_vector[2]))
      unit_linear_vel_vector = linear_vel_vector / np.linalg.norm(linear_vel_vector)
      send_linear_velocity = kp_linear_vel * linear_vel * unit_linear_vel_vector
      pub_tip_vel.publish(send_linear_velocity[0], send_linear_velocity[1], send_linear_velocity[2]) # need to change
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y, z)) + send_linear_velocity * 0.3
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_angle_insertion_flag = False
      rate.sleep()
    elif (move_angle_retraction_flag):
      # send_linear_velocity = kp_linear_vel * linear_vel
      current_position = np.array((x, y, z))
      rcm_to_current_vector = current_position - rcm_point
      xy_sqrt = math.sqrt(rcm_to_current_vector[0]**2 + rcm_to_current_vector[1]**2)
      # for certain degree bended tool, define the value above
      linear_vel_vector = np.array((rcm_to_current_vector[0] * ((math.cos(move_angle) * xy_sqrt - math.sin(move_angle) * rcm_to_current_vector[2]) / xy_sqrt), \
                                    rcm_to_current_vector[1] * ((math.cos(move_angle) * xy_sqrt - math.sin(move_angle) * rcm_to_current_vector[2]) / xy_sqrt), \
                                    math.sin(move_angle) * xy_sqrt + math.cos(move_angle) * rcm_to_current_vector[2]))
      unit_linear_vel_vector = linear_vel_vector / np.linalg.norm(linear_vel_vector)
      send_linear_velocity = kp_linear_vel * linear_vel * unit_linear_vel_vector
      pub_tip_vel.publish(-send_linear_velocity[0], -send_linear_velocity[1], -send_linear_velocity[2]) # need to change
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y, z)) - send_linear_velocity * 0.3
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_angle_retraction_flag = False
      rate.sleep()
    elif (small_push_flag):
      kp_linear_vel = 8 # 0.5, 1, 0.01, 0.6, 0.03
      kp_angular_vel = 6.5 # 1.0, 2, 0.005, 1.5, 0.015
      linear_vel = 0.3
      # send_linear_velocity = kp_linear_vel * linear_vel
      current_position = np.array((x, y, z))
      rcm_to_current_vector = current_position - rcm_point
      xy_sqrt = math.sqrt(rcm_to_current_vector[0]**2 + rcm_to_current_vector[1]**2)
      # for certain degree bended tool, define the value above
      linear_vel_vector = np.array((rcm_to_current_vector[0] * ((math.cos(move_angle) * xy_sqrt - math.sin(move_angle) * rcm_to_current_vector[2]) / xy_sqrt), \
                                    rcm_to_current_vector[1] * ((math.cos(move_angle) * xy_sqrt - math.sin(move_angle) * rcm_to_current_vector[2]) / xy_sqrt), \
                                    math.sin(move_angle) * xy_sqrt + math.cos(move_angle) * rcm_to_current_vector[2]))
      unit_linear_vel_vector = linear_vel_vector / np.linalg.norm(linear_vel_vector)
      send_linear_velocity = kp_linear_vel * linear_vel * unit_linear_vel_vector
      pub_tip_vel.publish(send_linear_velocity[0], send_linear_velocity[1], send_linear_velocity[2]) # need to change
      # Track angular velocity
      current_quat = np.array((rx, ry, rz, rw))
      r_current = R.from_quat(current_quat)
      rotation_matrix_current = r_current.as_dcm()
      goal_position = np.array((x, y, z)) + send_linear_velocity * 0.3
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
      rotation_matrix_rcm_to_desired = r_temp.as_dcm()
      error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
      r_error = R.from_dcm(error_angular_rotation_matrix)
      error_rotation_vector = r_error.as_rotvec()
      diff_to_final_angle = np.linalg.norm(error_rotation_vector)
      if diff_to_final_angle < diff_angle_thresh:
        pub_tip_vel_angular.publish(0, 0, 0)
        continue
      # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
      unit_error_rotation_vector = error_rotation_vector
      angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)
      print("***********************angular_velocity", angular_velocity)

      # Track angular vel w/o normalization and also using desired angular velocity
      send_angular_velocity = angular_velocity
      pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
      move_angle_insertion_flag = False
      kp_linear_vel = 2 # usded to be 1 01/26/2023
      kp_angular_vel = 3 # 1.0, 2, 0.005, 1.5, 0.015
      linear_vel = 0.1
      rate.sleep()
    else:
      pub_tip_vel.publish(0, 0, 0)
      pub_tip_vel_angular.publish(0, 0, 0)
      rate.sleep()
