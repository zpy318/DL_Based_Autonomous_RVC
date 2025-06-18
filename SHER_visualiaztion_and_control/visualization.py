# For defining the target point on the images
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

# for camera image recording
camera_image = None
# for clicking goal points and publishing over ros
clicked_goal = None
clicked_goal_list = []
vel_x = None
vel_y = None
vel_z = None
b_scan_raw = None
volume_data = None
x = None
y = None
z = None
rx = None
ry = None
rz = None
rw = None

class image_converter:

  def __init__(self):
    self.camera_img_sub = rospy.Subscriber("/decklink/camera/image_raw", Image, self.get_camera_image)
    self.clicked_goal_sub = rospy.Subscriber("/click_goal", Vector3, self.get_clicked_goal)
    self.velocity_sub = rospy.Subscriber('/eyerobot2/desiredTipVelocities', Vector3, self.get_tip_velocity)
    # self.b_scan_sub = rospy.Subscriber("/breath_simulation_b_scan", Image, self.get_b_scan)
    self.b_scan_sub = rospy.Subscriber("/b_scan", Image, self.get_b_scan)
    self.tool_tip_position_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.get_tool_tip_position)

  def get_camera_image(self, data):
    global camera_image
    # bridge = CvBridge()
    # camera_image = bridge.imgmsg_to_cv2(data, "bgr8")
    camera_image = data

  def get_clicked_goal(self, data):
    global clicked_goal
    global clicked_goal_list
    clicked_goal = data
    print("select code, print clicked goal: ", clicked_goal)
    if len(clicked_goal_list) == 0:
      clicked_goal_list.append(clicked_goal)
    else:
      clicked_goal_list = []
      clicked_goal_list.append(clicked_goal)

  def get_tip_velocity(self, data):
    global vel_x
    global vel_y
    global vel_z
    vel_x = data.x
    vel_y = data.y
    vel_z = data.z

  def get_b_scan(self, data):
    global b_scan_raw
    # bridge = CvBridge()
    # b_scan_raw = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    b_scan_raw = data

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

# Path to save CSV file
#csv_file = '/home/peiyao/Downloads/clicked_positions.csv'

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

        # for i in range(resized_shape[2]):
        #     latest_scans_resized_2[:, :, i] = cv2.resize(latest_scans_resized_1[:, :, i], (resized_shape[1], resized_shape[0]))

        # latest_scans_resized_2 = np.transpose(latest_scans_resized_2, (2, 0, 1))
        # latest_scans_resized_2 = np.flip(latest_scans_resized_2, 1)
        # latest_scans_resized_2 = np.flip(latest_scans_resized_2, 2)
        # # cv2.imwrite("tmp", latest_scans_resized)
        # # print("latest_scans_resized_2 shape = ", latest_scans_resized_2.shape)
        # spacing = spacing[[2, 0, 1]]

        # return latest_scans_resized_2, spacing
        return 1

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
        # print("***********************frame number = ", frame_number)
        return frame_number, frame

class camera_image_network(torch.nn.Module):
    """
    output size is 512 x 512 x 4
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


# for publishing the clicked goal position
def create_publisher():
  """
      Usage:
        publisher = create_publisher()
        publisher.publish(desired_position[0], desired_position[1], desired_position[2])
  """
  pub = rospy.Publisher("/click_goal", Vector3, queue_size=10)
  return pub  

def mouseRGB(event,x,y,flags,param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = frame[y,x,0]
        colorsG = frame[y,x,1]
        colorsR = frame[y,x,2]
        colors = frame[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)
        publisher_send_clicked_goal.publish(x, y, 0.0)

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

ic = image_converter()
# for sending clicked goal position over ros node
publisher_send_clicked_goal = create_publisher()
rospy.init_node('image_converter', anonymous=True)
time.sleep(2)

# navigation_network
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

# b_scan contact network
model_test_contact = YOLO('../dataset_for_all/contact_network/best_model/best.pt')

# b_scan puncture network YOLO object detection
model_test_puncture = YOLO('../dataset_for_all/puncture_network/best_model/best.pt')

sleep_duration = 0.08
cv2.namedWindow('microscope_frame')
# cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('microscope_frame',mouseRGB)

center_x = 60
center_y = 60
# a = LeicaEngine("192.168.1.75", 2000, 5, 6, 0.1, 3.379, 1)
b_scan_visualization_path = "../visualization/b_scans_{:10.2f}".format(time.time())
if not os.path.exists(b_scan_visualization_path):
    os.makedirs(b_scan_visualization_path)
contact_threshold = 0.90
puncture_threshold = 0.65
time_stamp = time.time()
# remove latency
while (b_scan_raw is None):
   continue
b_scan = image_to_numpy(b_scan_raw)
cv2.imwrite(os.path.join(b_scan_visualization_path, "b_scans_{:10.2f}_{}.jpg".format(time_stamp, 1)), b_scan)
b_scan = cv2.imread(os.path.join(b_scan_visualization_path, "b_scans_{:10.2f}_{}.jpg".format(time_stamp, 1)))
b_scan = cv2.resize(b_scan, (512, 512))
outputs = model_test_contact(b_scan)
outputs = model_test_puncture(b_scan)
for i in range(4000):
  prob_contact = 0
  prob_puncture = 0
  while not rospy.is_shutdown():
    # buf = a.__get_buffer__()
    # frame_number, b_scan_direct = a.__parse_data__(buf)
    #   a.__get_b_scans_volume__()
    if b_scan_raw is not None:
        b_scan = image_to_numpy(b_scan_raw)
        cv2.imwrite(os.path.join(b_scan_visualization_path, "b_scans_{:10.2f}_{}.jpg".format(time_stamp, i)), b_scan)
        b_scan = cv2.imread(os.path.join(b_scan_visualization_path, "b_scans_{:10.2f}_{}.jpg".format(time_stamp, i)))
        b_scan = cv2.resize(b_scan, (512, 512))
        b_scan_frame = cv2.resize(b_scan, (909, 512))
        # b_scan = TF.to_tensor(b_scan) # C x H x W
        # b_scan = b_scan.unsqueeze(0).float().to(device) # N x C x H x W
        if (prob_contact <= contact_threshold):
            # b_scan = image_to_numpy(b_scan_raw)
            outputs = model_test_contact(b_scan)
            prob_contact = outputs[0].probs.data[0].item()
            print("contact probability = ", prob_contact)
            # prob_no_contact = outputs[0].probs.data[1].item()
        if (prob_contact > contact_threshold):
            outputs = model_test_puncture(b_scan)
            if (len(outputs[0].boxes.data) != 0):
                if (outputs[0].boxes.cls[0].item() == 0):
                    prob_puncture = 0
                else:
                    prob_puncture = outputs[0].boxes.conf[0].item()
            else:
               prob_puncture = 0
            print("puncture probability = ", prob_puncture)
        font = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1
        current_quat = np.array((rx, ry, rz, rw))
        r_current = R.from_quat(current_quat)
        rotation_matrix_current = r_current.as_matrix()
        vector_cur = np.matmul(rotation_matrix_current, np.array((0, 0, -1)))
        vector_plane = np.array((vector_cur[0], vector_cur[1], 0))
        insertion_angle = angle_between(vector_cur, vector_plane) / np.pi * 180
        angle_line_vector = np.array((-50, -50*np.tan(insertion_angle / 180 * np.pi)))
        angle_line_vector = angle_line_vector / np.linalg.norm(angle_line_vector) * 50
        b_scan_frame = cv2.line(b_scan_frame, (225, 50), (275, 50), color, thickness=2)
        b_scan_frame = cv2.line(b_scan_frame, (275, 50), (275 + int(angle_line_vector[0]), 50 + int(angle_line_vector[1])), color, thickness=2)
        b_scan_frame = cv2.putText(b_scan_frame, 'Insertion Angle:', (10, 50), font, fontScale, color, thickness, cv2.LINE_AA)
        b_scan_frame = cv2.putText(b_scan_frame, '{:4.2f}'.format(insertion_angle), (285, 50), font, fontScale, color, thickness, cv2.LINE_AA)
        b_scan_frame = cv2.putText(b_scan_frame, 'Contact Prob:', (10, 85), font, fontScale, color, thickness, cv2.LINE_AA)
        b_scan_frame = cv2.rectangle(b_scan_frame, (215, 85), (415, 65), color, thickness)
        b_scan_frame = cv2.line(b_scan_frame, (215 + int(200 * contact_threshold), 84), (215 + int(200 * contact_threshold), 66), color=(0, 255, 0), thickness=1)
        if (prob_contact <= contact_threshold):
            b_scan_frame = cv2.rectangle(b_scan_frame, (216, 84), (216 + int(198*prob_contact), 66), color=(255, 0, 0), thickness=-1)
        else:
            b_scan_frame = cv2.rectangle(b_scan_frame, (216, 84), (216 + int(198*prob_contact), 66), color=(0, 255, 0), thickness=-1)
        b_scan_frame = cv2.putText(b_scan_frame, 'Puncture Prob:', (10, 120), font, fontScale, color, thickness, cv2.LINE_AA)
        b_scan_frame = cv2.rectangle(b_scan_frame, (215, 120), (415, 100), color, thickness)
        b_scan_frame = cv2.line(b_scan_frame, (215 + int(200 * puncture_threshold), 119), (215 + int(200 * puncture_threshold), 101), color=(0, 255, 0), thickness=1)
        if (prob_contact > contact_threshold):
            if (prob_puncture <= puncture_threshold):
                b_scan_frame = cv2.rectangle(b_scan_frame, (216, 119), (216 + int(198*prob_puncture), 101), color=(255, 0, 0), thickness=-1)
            else:
                b_scan_frame = cv2.rectangle(b_scan_frame, (216, 119), (216 + int(198*prob_puncture), 101), color=(0, 255, 0), thickness=-1)
        if vel_x is not None:
            b_scan_frame = cv2.putText(b_scan_frame, 'Vel_z = ', (10, 155), font, fontScale, color, thickness, cv2.LINE_AA)
            b_scan_frame = cv2.putText(b_scan_frame, '{:.2f}'.format(vel_z), (125, 155), font, fontScale, color, thickness, cv2.LINE_AA)
        b_scan_frame = cv2.cvtColor(b_scan_frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('b_scan_frame', b_scan_frame)
    if camera_image is not None:
      frame = image_to_numpy(camera_image)
      frame = frame[90:990, 510:1410, :]
      frame = cv2.resize(frame, (512, 512))
    sz = 20
    sz_ = 8
    # plot the clicked goal positions
    if len(clicked_goal_list) == 1:
      xx = int(clicked_goal_list[0].x)
      yy = int(clicked_goal_list[0].y)
      
      target = [clicked_goal_list[0].x, clicked_goal_list[0].y]
      if (prob_contact < contact_threshold):
        input_image = image_to_numpy(camera_image)
        input_image = input_image[90:990, 510:1410, :]
        input_image = cv2.resize(input_image, (512, 512))
        target_image = np.zeros((512, 512))
        target_image[int(target[1])-sz:int(target[1])+sz+1, int(target[0])-sz:int(target[0])+sz+1] = 1 
        target_image = target_image[:,:,np.newaxis]*255 # 1080 x 1920 x 1
        image = np.concatenate((input_image, target_image), axis = 2) / 255.0
        image = TF.to_tensor(image) # C x H x W
        image = image.unsqueeze(0).float().to(device) # N x C x H x W
        outputs = model_test_camera(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs[0]
        argmax_output = np.argmax(outputs)
        print("Predicted result = ", argmax_output)
      else:
        argmax_output = 8
      frame = cv2.rectangle(frame, (xx-sz_, yy-sz_), 
                          (xx+sz_, yy+sz_), color = (255, 255,255), thickness = 1)
      frame = cv2.rectangle(frame, (xx-1, yy-1), 
                          (xx+1, yy+1), color = (255,255,255), thickness = 1)
      angle = (45 * argmax_output - 180) / 180 * np.pi
      vector_x = 50 * np.cos(angle)
      vector_y = 50 * np.sin(angle)
      color = (255, 255, 255)
      frame = cv2.circle(frame, (center_x, center_y), 50, color, thickness=2) 
      frame = cv2.line(frame, (int(center_x + 1 + 50 * np.cos(22.5 * np.pi / 180)), int(center_y + 1 + 50 * np.sin(22.5 * np.pi / 180))), (int(center_x + 50 * np.cos(-157.5 * np.pi / 180)), int(center_y + 50 * np.sin(-157.5 * np.pi / 180))), color, thickness=2)
      frame = cv2.line(frame, (int(center_x + 1 + 50 * np.cos(67.5 * np.pi / 180)), int(center_y + 1 + 50 * np.sin(67.5 * np.pi / 180))), (int(center_x + 50 * np.cos(-112.5 * np.pi / 180)), int(center_y + 50 * np.sin(-112.5 * np.pi / 180))), color, thickness=2)
      frame = cv2.line(frame, (int(center_x + 1 + 50 * np.cos(112.5 * np.pi / 180)), int(center_y + 1 + 50 * np.sin(112.5 * np.pi / 180))), (int(center_x + 50 * np.cos(-67.5 * np.pi / 180)), int(center_y + 50 * np.sin(-67.5 * np.pi / 180))), color, thickness=2)
      frame = cv2.line(frame, (int(center_x + 1 + 50 * np.cos(157.5 * np.pi / 180)), int(center_y + 1 + 50 * np.sin(157.5 * np.pi / 180))), (int(center_x + 50 * np.cos(-22.5 * np.pi / 180)), int(center_y + 50 * np.sin(-22.5 * np.pi / 180))), color, thickness=2)
      if (argmax_output < 8):
        frame = cv2.ellipse(frame, (center_x, center_y), (50, 50), 0, -argmax_output * 45 + 202.5, -argmax_output * 45 + 157.5, color=(255, 0, 0), thickness=-1)
        frame = cv2.arrowedLine(frame, (center_x, center_y), (center_x + int(vector_x), center_y - int(vector_y)), color=(0, 150, 0), thickness=2)
      else:
         frame = cv2.circle(frame, (center_x, center_y), 10, color=(255, 0, 0), thickness=-1) 
         frame = cv2.circle(frame, (center_x, center_y), 5, color=(0, 150, 0), thickness=-1) 
      font = cv2.FONT_HERSHEY_DUPLEX
      fontScale = 0.8
      color = (255, 255, 255)
      thickness = 1
      # Add xyz velocities to the image
      if vel_x is not None:
        frame = cv2.putText(frame, 'Vel_x = ', (130, 35), font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, '{:.2f}'.format(vel_x), (245, 35), font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, 'Vel_y = ', (130, 70), font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, '{:.2f}'.format(vel_y),(245, 70), font, fontScale, color, thickness, cv2.LINE_AA)
    # Display the resulting frame
    if camera_image is not None:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      cv2.imshow('microscope_frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      pass
  clicked_goal_list = []