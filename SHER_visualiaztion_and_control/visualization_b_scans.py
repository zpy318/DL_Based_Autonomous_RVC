#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from pytransform3d.trajectories import plot_trajectory
import time
import math
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray # for publishing and subscribing
from geometry_msgs.msg import Vector3, Transform
import socket
import struct
import cv2
import os
from cv_bridge import CvBridge

# pytorch dependencies
import torch
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from skimage import io, transform, util

class LeicaEngine(object):

    def __init__(self, ip_address="192.168.1.75", port_num=2000, n_bscans=500, xd=6.0, yd=6.0, zd=3.379, scale=1):

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

while not rospy.is_shutdown():
    folder = input("Enter name of current trial to collect b scans or interrupt to exit\n")

    oct = LeicaEngine("192.168.1.75")

    current_path = os.path.join("../b_scans", folder)
    print("current_path = ", current_path)
    if not os.path.exists(current_path):
        os.makedirs(current_path)

    time_record_start = time.perf_counter()
    ind = 0
    rospy.init_node("publish_b_scan", anonymous=True)
    pub_img = rospy.Publisher("/b_scan", Image, queue_size=10)
    br = CvBridge()
    try:
        while not rospy.is_shutdown():
            buf = oct.__get_buffer__()
            frame_number, frame = oct.__parse_data__(buf)
            pub_img.publish(br.cv2_to_imgmsg(frame*255))
            ind += 1
            time_record_end = time.perf_counter()
            print("current mean frequency = ",  (ind + 1) / (time_record_end - time_record_start))
    except KeyboardInterrupt:
        print('interrupted!')
    time_record_end = time.perf_counter()
    print("Time elapsed = ", time_record_end - time_record_start)
    print("Number of b_scans = ", ind)
    print("final mean frequency = ", (ind + 1) / (time_record_end - time_record_start))
