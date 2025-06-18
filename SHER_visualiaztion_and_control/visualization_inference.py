#!/usr/bin/env python3
from __future__ import print_function, division
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from torchvision import models
import warnings; warnings.simplefilter('ignore')
from geometry_msgs.msg import Vector3, Transform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import math
import time
from scipy.spatial.transform import Rotation as R
from image_conversion_without_using_ros import image_to_numpy
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from einops import rearrange
from torchvision.models import resnet18
import socket
import struct

x = None
y = None
z = None
rx = None
ry = None
rz = None
rw = None
camera_image = None
user_defined_target_point = None
b_scan_raw = None
volume_data = None

class LeicaEngine(object):

    def __init__(self, ip_address="192.168.1.75", port_num=2000, n_bscans=100, xd=2.5, yd=2.5, zd=3.379, scale=1):

        # x: n_Ascan in Bsacn dir
        # y: n_Bscans dir
        # z: Ascan dir
        # output: n_bscans*len_ascan*width

        self.max_bytes = 2 ** 16
        self.server_address = (ip_address, port_num)
        self.b_scan_reading = None
        self.n_bscans = n_bscans
        self.xd = xd
        self.yd = yd
        self.zd = zd
        self.scale = scale
        self.__connect__()
        self.active = True
        self.latest_complete_scans = None
        self.latest_spacing = None

    def get_bscan(self, idx):

        return self.latest_complete_scans[:, idx, :]

    def __get_b_scans_volume_continously__(self):

        while self.active:

            latest_volume, latest_spacing = self.__get_b_scans_volume__()

            self.latest_complete_scans = latest_volume
            self.latest_spacing = latest_spacing

    def __get_b_scans_volume__(self):
        global volume_data
        start = None

        buf = self.__get_buffer__()
        _, frame = self.__parse_data__(buf)
        latest_scans = np.zeros((self.n_bscans, frame.shape[0], frame.shape[1]))
        resized_shape = (np.array(latest_scans.shape)*self.scale).astype(int)
        latest_scans_resized_1 = np.zeros([self.n_bscans, resized_shape[1], resized_shape[2]])
        latest_scans_resized_2 = np.zeros(resized_shape)

        t = np.array(latest_scans_resized_2.shape)
        spacing = np.array([self.xd, self.zd, self.yd]) / t

        while True:

            buf = self.__get_buffer__()
            frame_number, frame = self.__parse_data__(buf)

            if start is None:
                start = frame_number

            latest_scans[frame_number, :, :] = frame
            latest_scans_resized_1[frame_number, :, :] = cv2.resize(frame, (resized_shape[2], resized_shape[1]))

            if frame_number == (start - 1) % self.n_bscans:
                break
        # print("current directory is: ", os.getcwd())
        volume_data = latest_scans

        for i in range(resized_shape[2]):
            latest_scans_resized_2[:, :, i] = cv2.resize(latest_scans_resized_1[:, :, i], (resized_shape[1], resized_shape[0]))

        latest_scans_resized_2 = np.transpose(latest_scans_resized_2, (2, 0, 1))
        latest_scans_resized_2 = np.flip(latest_scans_resized_2, 1)
        latest_scans_resized_2 = np.flip(latest_scans_resized_2, 2)
        # cv2.imwrite("tmp", latest_scans_resized)
        # print("latest_scans_resized_2 shape = ", latest_scans_resized_2.shape)
        spacing = spacing[[2, 0, 1]]

        return latest_scans_resized_2, spacing

    def __disconnect__(self):

        self.active = False
        self.sock.close()

    def __connect__(self):

        print(f"Connecting to {self.server_address[0]} and port {self.server_address[1]}")

        tries = 0
        connected = False
        while tries < 10 and not connected:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(self.server_address)
                connected = True
            except Exception as e:
                print(f'No connection. Waiting on server. {tries} Attempts.')
                tries += 1

        self.active = True

        if connected:
            print(f"Connection Successful")
        else:
            print("Connection Failed")

    def __get_buffer__(self):

        buf = None
        num_expected_bytes = 0

        while True:
            try:
                data = self.sock.recv(self.max_bytes)
            except Exception as e:
                print('Connection error. Trying to re-establish connection.')
                break

            if buf is None:
                if len(data) == 0:
                    break

                if len(data) < 10:
                    message = 'Waiting for new frame'
                    message_bytes = str.encode(message)
                    self.sock.sendall(message_bytes)
                    continue

                buf = data

                start_pos = 0
                end_pos = 4
                dataLabelSize = struct.unpack('I', buf[start_pos:end_pos])[0]
                dataLabel = struct.unpack('B' * int(dataLabelSize), buf[end_pos:end_pos + int(dataLabelSize)])
                dataLabel = ''.join([chr(L) for L in dataLabel])

                start_pos = end_pos + int(dataLabelSize)
                end_pos = start_pos + 4

                if dataLabel == 'EXPECTEDBYTES':
                    val_length = struct.unpack('I', buf[start_pos:end_pos])[0]
                    num_expected_bytes = struct.unpack('I', buf[end_pos:end_pos + val_length])[0]
                    start_pos = end_pos + 4
                    end_pos = start_pos + 4
            else:
                buf = buf + data

            if buf is not None and len(buf) >= num_expected_bytes:
                break

        message = 'Received frame'
        message_bytes = str.encode(message)
        self.sock.sendall(message_bytes)

        return buf

    def __parse_data__(self, buf):

        dataLabel = None
        start_pos = 0
        end_pos = 4

        while dataLabel != 'ENDFRAMEHEADER':
            dataLabelSize = struct.unpack('I', buf[start_pos:end_pos])[0]
            dataLabel = struct.unpack('B' * int(dataLabelSize), buf[end_pos:end_pos + int(dataLabelSize)])
            start_pos = end_pos + int(dataLabelSize)
            end_pos = start_pos + 4

            dataLabel = ''.join([chr(L) for L in dataLabel])

            if dataLabel == 'ENDFRAMEHEADER':
                data_start_pos = start_pos + 8
                break
            else:
                val_length = struct.unpack('I', buf[start_pos:end_pos])[0]
                if val_length <= 4:
                    val = struct.unpack('I', buf[end_pos:end_pos + 4])[0]
                    val_pos = end_pos
                    start_pos = end_pos + 4
                    end_pos = start_pos + 4
                else:
                    val = struct.unpack('d', buf[end_pos:end_pos + 8])[0]
                    val_pos = end_pos
                    start_pos = end_pos + 8
                    end_pos = start_pos + 4

                if dataLabel == 'FRAMENUMBER':
                    frame_number = val
                    frame_number_pos = val_pos
                if dataLabel == 'FRAMECOUNT':
                    frame_count = val
                if dataLabel == 'LINECOUNT':
                    line_count = val
                if dataLabel == 'LINELENGTH':
                    line_length = val
                if dataLabel == 'AIMFRAMES':
                    aim_frames = val

        frameData = np.zeros((line_length, line_count))

        frame_number = frame_number % frame_count

        for i in range(0, line_count):
            start = data_start_pos + i * line_length * 2
            frameData[:, i] = np.frombuffer(buf[start:start + line_length * 2], dtype='u2', count=line_length)

        frame = frameData / self.max_bytes

        return frame_number, frame

class camera_image_network(torch.nn.Module):
    """
    output size is 512 x 512
    """
    def __init__(self, orig_resnet, n_classes=9):
        super().__init__()
        self.orig_resnet = orig_resnet
        self.orig_resnet.avgpool = nn.AvgPool2d(kernel_size=(8,8), stride=(4,4))
        self.orig_resnet.fc = nn.Linear(in_features=3*3*512, out_features=1000)
        self.orig_resnet.out = nn.Linear(in_features=1000, out_features=n_classes)

    def forward(self, x):
        # encoder
        x1 = self.orig_resnet.conv1(x) # 64 x 256 x 256
        x2 = self.orig_resnet.bn1(x1) 
        x3 = self.orig_resnet.relu(x2)
        x4 = self.orig_resnet.maxpool(x3) # 64 x 128 x 128
        x5 = self.orig_resnet.layer1(x4) # 64 x 128 x 128
        x6 = self.orig_resnet.layer2(x5) # 128 x 64 x 64
        x7 = self.orig_resnet.layer3(x6) # 256 x 32 x 32
        x8 = self.orig_resnet.layer4(x7) # 512 x 16 x 16
        x9 = self.orig_resnet.avgpool(x8) # 512 x 3 x 3
        x10 = x9.reshape(x9.shape[0], -1) # N x (512 x 3 x 3)
        x11 = self.orig_resnet.relu(self.orig_resnet.fc(x10)) # N x 1000
        x12 = self.orig_resnet.out(x11) # N x 9
        return x12
  
class subscribe_image_and_goal:

    def __init__(self):
        self.camera_img_sub = rospy.Subscriber("/decklink/camera/image_raw", Image, self.get_camera_image)
        self.clicked_goal_sub = rospy.Subscriber("/click_goal", Vector3, self.get_target_point)
        self.tool_tip_position_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.get_tool_tip_position)
        self.b_scan_sub = rospy.Subscriber("/b_scan", Image, self.get_b_scan)

    def get_camera_image(self, data):
        global camera_image
        # bridge = CvBridge()
        # camera_image = bridge.imgmsg_to_cv2(data, "bgr8")
        camera_image = data
    
    def get_b_scan(self, data):
        global b_scan_raw
        b_scan_raw = data

    def get_target_point(self, data):
        global user_defined_target_point
        user_defined_target_point = data
        print("select code, print clicked goal: ", user_defined_target_point)

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

def create_publisher():
  """
      Usage:
        publisher = create_publisher()
        publisher.publish(desired_position[0], desired_position[1], desired_position[2])
  """
  pub_tip_vel = rospy.Publisher('/eyerobot2/desiredTipVelocities', Vector3, queue_size = 3)
  pub_tip_vel_angular = rospy.Publisher('/eyerobot2/desiredTipVelocitiesAngular', Vector3, queue_size = 3)
  return pub_tip_vel, pub_tip_vel_angular

subscribe_image_and_goal()
# Camera network
resnet_camera = models.resnet18(pretrained=False)
# choose resnet of your choice
n_input_channel_camera = 4
resnet_camera.conv1 = torch.nn.Conv2d(n_input_channel_camera, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device =", device)
resnet_enc_dec_concat_more = camera_image_network(resnet_camera) 
model_test_camera = resnet_enc_dec_concat_more.to(device)
    
# load previous model
checkpoint_camera = torch.load("../dataset_for_all/navigation_network/best_model/epoch_178val_checkpoint.pth.tar")
# checkpoint = torch.load("training_parameters/5_eyes_not_aggressive_augmentation/epoch_112val_checkpoint.pth.tar", map_location={'cuda:0':'cpu'})
model_test_camera.load_state_dict(checkpoint_camera['state_dict'])
model_test_camera = model_test_camera.to(device).eval()

model_test_contact = YOLO('../dataset_for_all/contact_network/best_model/best.pt')
model_test_puncture = YOLO('../dataset_for_all/puncture_network/best_model/best.pt')

sz = 20

pub_tip_vel, pub_tip_vel_angular = create_publisher()
rospy.init_node('move', anonymous=True)
rate = rospy.Rate(100) # 100hz
kp_linear_vel = 2 # used to be 1.5 01/26/2023
kp_angular_vel = 3 # 1.0, 2, 0.005, 1.5, 0.015
downward_linear_vel = 2
downward_angular_vel = 3
downward_flag = False
move_angle = 0 * math.pi / 180 # set the tool-tip bended angle default 20
forward_step_size = 0.3
linear_vel = 0.1
diff_angle_thresh = 0.1 * math.pi / 180
rcm_point = None
while rcm_point is None:
    rcm_point = np.array((x, y, z))
print("rcm_point = ", rcm_point)
input("Check if you have left the RCM point using Cannulation mode and press Enter to start navigation...")
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

a = LeicaEngine("192.168.1.75", 2000, 5, 6, 0.1, 3.379, 1)

time_keeper = []
print("Start navigation!")
start = time.time()
b_scan_contact_path = "../data/small_c_scans/b_scans_{:10.2f}_contact".format(time.time())
if not os.path.exists(b_scan_contact_path):
    os.makedirs(b_scan_contact_path)
b_scan_puncture_path = "../data/small_c_scans/b_scans_{:10.2f}_puncture".format(time.time())
if not os.path.exists(b_scan_puncture_path):
    os.makedirs(b_scan_puncture_path)
time_stamp = time.time()
while not rospy.is_shutdown():
    if b_scan_raw is not None:
        # a.__get_b_scans_volume__()
        argmax_output_contact = 0
        b_scan = image_to_numpy(b_scan_raw)
        cv2.imwrite(os.path.join(b_scan_contact_path, "b_scans_{:10.2f}_contact_{}.jpg".format(time_stamp, 0)), b_scan)
        b_scan = cv2.imread(os.path.join(b_scan_contact_path, "b_scans_{:10.2f}_contact_{}.jpg".format(time_stamp, 0)))
        b_scan = cv2.resize(b_scan, (512, 512))
        # b_scan = TF.to_tensor(b_scan) # C x H x W
        # b_scan = b_scan.unsqueeze(0).float().to(device) # N x C x H x W
        outputs = model_test_contact(b_scan)
        prob_contact = outputs[0].probs.data[0].item()
        prob_no_contact = outputs[0].probs.data[1].item()
        print("prob_contact = ", prob_contact)
        print("prob_no_contact = ", prob_no_contact)
        if prob_contact > 0.90:
            argmax_output_contact = 1
            # break
        if (argmax_output_contact == 0):
            if (user_defined_target_point is not None):
                # print("user_defined_target_point = ", user_defined_target_point)
                target = [user_defined_target_point.x, user_defined_target_point.y]
                input_image = image_to_numpy(camera_image)
                input_image = input_image[90:990, 510:1410, :]
                input_image = cv2.resize(input_image, (512, 512))
                target_image = np.zeros((512, 512))
                target_image[int(target[1])-sz:int(target[1])+sz+1, int(target[0])-sz:int(target[0])+sz+1] = 1 
                target_image = target_image[:,:,np.newaxis]*255 # 1080 x 1920 x 1
                image = np.concatenate((input_image, target_image), axis = 2) / 255.0
                image = TF.to_tensor(image) # C x H x W
                image = image.unsqueeze(0).float().to(device) # N x C x H x W

                # Test - the standard way
                outputs = model_test_camera(image)
                outputs = outputs.cpu().detach().numpy()
                outputs = outputs[0]
                argmax_output = np.argmax(outputs)
                print("Predicted result = ", argmax_output)
                horizontal_linear_vel_x = 0
                horizontal_linear_vel_y = 0
                
                if (argmax_output == 8):
                    downward_flag = True
                    kp_linear_vel = 0.6
                    kp_angular_vel = 0.9 # used to be 0.9
                elif downward_flag is False:
                    send_linear_velocity = kp_linear_vel * linear_vel
                    angle = (45 * argmax_output - 270) / 180 * np.pi
                    print("angle = ", angle)
                    pub_tip_vel.publish(send_linear_velocity * np.cos(angle), send_linear_velocity * np.sin(angle), 0)
                    # Track angular velocity
                    current_quat = np.array((rx, ry, rz, rw))
                    r_current = R.from_quat(current_quat)
                    rotation_matrix_current = r_current.as_matrix()
                    goal_position = np.array((x + send_linear_velocity * np.cos(angle) * 0.6, y + send_linear_velocity * np.sin(angle) * 0.6, z))
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
                    # error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
                    r_error = R.from_matrix(error_angular_rotation_matrix)
                    error_rotation_vector = r_error.as_rotvec()
                    diff_to_final_angle = np.linalg.norm(error_rotation_vector)
                    if diff_to_final_angle < diff_angle_thresh:
                        pub_tip_vel_angular.publish(0, 0, 0)
                        rate.sleep()
                        continue
                    # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
                    unit_error_rotation_vector = error_rotation_vector
                    angular_velocity = kp_angular_vel * unit_error_rotation_vector.reshape(1,3)

                    # Track angular vel w/o normalization and also using desired angular velocity
                    send_angular_velocity = angular_velocity
                    pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
                    # time.sleep(0.2)
                    rate.sleep()
                if downward_flag is True:
                    if (argmax_output != 8):
                        send_linear_velocity = kp_linear_vel * linear_vel
                        angle = (45 * argmax_output - 270) / 180 * np.pi
                        horizontal_linear_vel_x = send_linear_velocity * np.cos(angle)
                        horizontal_linear_vel_y = send_linear_velocity * np.sin(angle)
                    send_linear_velocity = downward_linear_vel * linear_vel
                    pub_tip_vel.publish(horizontal_linear_vel_x, horizontal_linear_vel_y, -send_linear_velocity)
                    # Track angular velocity
                    current_quat = np.array((rx, ry, rz, rw))
                    r_current = R.from_quat(current_quat)
                    rotation_matrix_current = r_current.as_matrix()
                    goal_position = np.array((x + horizontal_linear_vel_x * 0.3, y + horizontal_linear_vel_y * 0.3, z - send_linear_velocity * 0.6))
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
                    # error_angular_rotation_matrix = np.matmul(np.transpose(rotation_matrix_current), rotation_matrix_rcm_to_desired)
                    r_error = R.from_matrix(error_angular_rotation_matrix)
                    error_rotation_vector = r_error.as_rotvec()
                    diff_to_final_angle = np.linalg.norm(error_rotation_vector)
                    if diff_to_final_angle < diff_angle_thresh:
                        pub_tip_vel_angular.publish(0, 0, 0)
                        rate.sleep()
                        continue
                    # unit_error_rotation_vector = error_rotation_vector / np.linalg.norm(error_rotation_vector)
                    unit_error_rotation_vector = error_rotation_vector
                    angular_velocity = downward_angular_vel * unit_error_rotation_vector.reshape(1,3)

                    # Track angular vel w/o normalization and also using desired angular velocity
                    send_angular_velocity = angular_velocity
                    pub_tip_vel_angular.publish(send_angular_velocity[0,0], send_angular_velocity[0,1], 0)
                    rate.sleep()
        else:
            pub_tip_vel.publish(0, 0, 0)
            pub_tip_vel_angular.publish(0, 0, 0)
            break
    else:
        pub_tip_vel.publish(0, 0, 0)
        pub_tip_vel_angular.publish(0, 0, 0)
        # break
#############################################################################################
# make sure RCM constraint is satisfied
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

# pub_tip_vel.publish(0, 0, 0)
pub_tip_vel_angular.publish(0, 0, 0)

linear_vel_gain = 0.1
current_point = np.array((x, y, z))
while np.linalg.norm(current_point - target_point) > 0.01:
    # find appropriate linear vel
    current_point = np.array((x, y, z))
    diff_xyz = target_point - current_point
    diff_xyz_norm = diff_xyz / np.linalg.norm(diff_xyz)
    linear_vel = diff_xyz_norm * linear_vel_gain

    # publish the velocities
    pub_tip_vel.publish(linear_vel[0], linear_vel[1], linear_vel[2])
    time.sleep(0.002)

print("The RCM consraint is strictly saftisfied!")
pub_tip_vel.publish(0, 0, 0)
#############################################################################################

pub_tip_vel.publish(0, 0, 0)
pub_tip_vel_angular.publish(0, 0, 0)

start_puncture = time.time()
nav_duration = start_puncture - start
print("The navigation duration is: ", nav_duration)
time_keeper.append(nav_duration)
while not rospy.is_shutdown():
    print("Start puncture step.")
    input("Press Enter to start puncturing...")
    kp_linear_vel = 18 # 18
    kp_angular_vel = 6.5 # 6.5
    linear_vel = 0.3
    current_position = np.array((x, y, z))
    time.sleep(0.1)
    current_quat = np.array((rx, ry, rz, rw))
    r_current = R.from_quat(current_quat)
    rotation_matrix_current = r_current.as_matrix()
    moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, -1)))
    send_linear_velocity = moving_direction * kp_linear_vel * linear_vel
    keep_puncturing_start = time.time()
    while (time.time() - keep_puncturing_start < 0.1):
        pub_tip_vel.publish(send_linear_velocity[0], send_linear_velocity[1], send_linear_velocity[2])
    pub_tip_vel.publish(0, 0, 0)
    pub_tip_vel_angular.publish(0, 0, 0)
    time.sleep(0.3)

    target_distance = np.linalg.norm((current_position - np.array((x, y, z))) * 4 / 5)
    puncture_position = np.array((x, y, z))
    while(True):
        # find appropriate linear vel
        kp_linear_vel = 2 # used to be 1.5 01/26/2023
        kp_angular_vel = 3 # 1.0, 2, 0.005, 1.5, 0.015
        current_position = np.array((x, y, z))

        moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, -1)))
        send_linear_velocity = moving_direction * kp_linear_vel * linear_vel
        pub_tip_vel.publish(-send_linear_velocity[0], -send_linear_velocity[1], -send_linear_velocity[2])
        time.sleep(0.03)

        # difference to final goal
        diff_to_final_goal = np.linalg.norm(current_position - puncture_position) - target_distance
        print("target_distance = ", target_distance)
        print("np.linalg.norm(current_position - puncture_position) = ", np.linalg.norm(current_position - puncture_position))
        print("diff_to_final_goal", diff_to_final_goal)

        if diff_to_final_goal > 0.003:
            break
    pub_tip_vel.publish(0, 0, 0)
    pub_tip_vel_angular.publish(0, 0, 0)
    while b_scan_raw is None:
        print("No b_scan_raw image!!")
    if b_scan_raw is not None:
        # a.__get_b_scans_volume__()
        argmax_output_puncture = 0
        time_stamp = time.time()
        # puncture_count = 0
        for i in range(15):
            b_scan = image_to_numpy(b_scan_raw)
            cv2.imwrite(os.path.join(b_scan_puncture_path, "b_scans_{:10.2f}_puncture_{}.jpg".format(time_stamp, i)), b_scan)
            b_scan = cv2.imread(os.path.join(b_scan_puncture_path, "b_scans_{:10.2f}_puncture_{}.jpg".format(time_stamp, i)))
            b_scan = cv2.resize(b_scan, (512, 512))
            # b_scan = TF.to_tensor(b_scan) # C x H x W
            # b_scan = b_scan.unsqueeze(0).float().to(device) # N x C x H x W
            outputs = model_test_puncture(b_scan)
            if (len(outputs[0].boxes.data) == 0):
                continue
            if (outputs[0].boxes.cls[0].item() == 0):
                prob_puncture = 0
            else:
                prob_puncture = outputs[0].boxes.conf[0].item()
            print("prob_puncture = ", prob_puncture)
            if prob_puncture > 0.65:
                argmax_output_puncture = 1
        if argmax_output_puncture == 0:
            continue
        else:
            break

input("Press Enter to start infusion step...")
start_infusion = time.time()
punc_duration = start_infusion - start_puncture
print("The puncture duration is: ", punc_duration)
time_keeper.append(punc_duration)
input("Press Enter to tool retraction step...")
start_retraction = time.time()
infusion_duration = start_retraction - start_infusion
print("The infusion duration is: ", infusion_duration)
time_keeper.append(infusion_duration)
puncture_position = np.array((x, y, z))
while not rospy.is_shutdown():
    current_position = np.array((x, y, z))
    current_quat = np.array((rx, ry, rz, rw))
    r_current = R.from_quat(current_quat)
    rotation_matrix_current = r_current.as_matrix()
    moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, -1)))
    send_linear_velocity = moving_direction * kp_linear_vel * linear_vel
    pub_tip_vel.publish(-send_linear_velocity[0], -send_linear_velocity[1], -send_linear_velocity[2])
    diff_to_final_goal = np.linalg.norm(current_position - puncture_position)
    if diff_to_final_goal > 0.8:
        pub_tip_vel.publish(0, 0, 0)
        break
end = time.time()
retraction_duration = end - start_retraction
print("The retraction duration is: ", retraction_duration)
time_keeper.append(retraction_duration)
np.savetxt("../data/network_test/test{:02f}.csv".format(time.time()), time_keeper, delimiter=",")
print("We are done.")
pub_tip_vel.publish(0, 0, 0)
pub_tip_vel_angular.publish(0, 0, 0)