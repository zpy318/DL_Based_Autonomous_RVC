from pipython import GCSDevice, pitools
import time
import random
import numpy as np
import os
import pandas as pd
import rospy
from geometry_msgs.msg import Transform

# Q motion controller setup
CONTROLLERNAME = 'E-873'

x = None
y = None
z = None
rx = None
ry = None
rz = None
rw = None

class subscribe_frameee:

    def __init__(self):
        self.tool_tip_position_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.get_tool_tip_position)

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

if __name__ == "__main__":
    subscribe_frameee()
    rospy.init_node('frameee_rcoreder', anonymous=True)
    plot_recording = []
    with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.ConnectTCPIP(ipaddress='192.168.1.220')
        pitools.startup(pidevice, stages='Q-522.130', refmodes='FNL')
        dy = 0
        dz = 0
        xaxis = '1'
        yaxis = '2'
        zaxis = '3'
        xvalue = 0
        yvalue = 0
        pidevice.CCL('1', 'ADVANCED')
        pidevice.SPA('1', '0x1F000400', 25000) # Hz range 150Hz - 25000Hz
        pidevice.SPA('2', '0x1F000400', 25000) # Hz range 150Hz - 25000Hz
        pidevice.SPA('3', '0x1F000400', 10000) # Hz range 150Hz - 25000Hz
        pidevice.MOV({'1': 0, '2': 0, '3': -5.5})
        count = 1
        start = time.time()
        while not rospy.is_shutdown():
            cur_time = time.time()
            zvalue = -5.5 + 0.1 * np.sin(2 * np.pi * (cur_time - start) * 0.2)
            plot_recording.append([x, y, z, rx, ry, rz, rw, zvalue, time.time()])
            pidevice.MOV({'1': xvalue, '2': yvalue, '3': zvalue})
            count += 1
    header =  ["robot_x", "robot_y", "robot_z", "robot_rx", "robot_ry", "robot_rz", "robot_rw", "linear_stage_z", "timestanp"]
    plot_recording_data = pd.DataFrame(plot_recording)
    plot_recording_data.to_csv("/home/peiyao/Desktop/Demir/breathing_res2/100um-8.csv", index = False, header = header)
