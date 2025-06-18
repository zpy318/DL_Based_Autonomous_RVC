#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import os
from cv_bridge import CvBridge
import numpy as np
import time
from image_conversion_without_using_ros import image_to_numpy

b_scan = None
camera_image = None

def get_b_scan(data):
    global b_scan
    b_scan = data

def get_camera_image(data):
    global camera_image
    bridge = CvBridge()
    camera_image = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node("subscribe_all_images", anonymous = True)
rospy.Subscriber("/b_scan", Image, get_b_scan)
rospy.Subscriber("/decklink/camera/image_raw", Image, get_camera_image)
count = 0
br = CvBridge()
b_scan_path = "../data/b_scans_{}".format(time.time())
if not os.path.exists(b_scan_path):
    os.makedirs(b_scan_path)
camera_image_path = "../data/camera_image_{}".format(time.time())
if not os.path.exists(camera_image_path):
    os.makedirs(camera_image_path)
time_record_start = time.time()
while not rospy.is_shutdown():
    if b_scan is not None and camera_image is not None:
        print("count = ", count)
        b_scan = image_to_numpy(b_scan)
        cv2.imwrite(os.path.join(b_scan_path, "b_scans_{}.jpg".format(count)), b_scan)
        cv2.imwrite(os.path.join(camera_image_path, "camera_image_{}.jpg".format(count)), camera_image)
        b_scan = None
        camera_image = None
        time_record_end = time.time()
        print("current mean frequency = ",  (count + 1) / (time_record_end - time_record_start))
        count += 1
