#!/usr/bin/env python3
# license removed for brevity
import rospy
from key_publisher_msgs.msg import key_publisher_msgs
import keyboard

def key_publisher():
    pub = rospy.Publisher('key_input', key_publisher_msgs, queue_size=10)
    rospy.init_node('key_publisher', anonymous=True)
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        msg = key_publisher_msgs()
        if (keyboard.is_pressed('left arrow')):
            msg.move_left = True
        if (keyboard.is_pressed('right arrow')):
            msg.move_right = True
        if (keyboard.is_pressed('up arrow')):
            msg.move_up = True
        if (keyboard.is_pressed('down arrow')):
            msg.move_down = True
        if (keyboard.is_pressed('u')):
            msg.move_upward = True
        if (keyboard.is_pressed('d')):
            msg.move_downward = True
        # insertion along tool shaft
        if (keyboard.is_pressed('f')):
            msg.move_forward = True
        # retraction along tool shaft
        if (keyboard.is_pressed('b')):
            msg.move_backward = True
        # insertion along needle bend
        if (keyboard.is_pressed('i')):
            msg.move_angle_insertion = True
        # retraction along needle bend
        if (keyboard.is_pressed('r')):
            msg.move_angle_retraction = True
        if (keyboard.is_pressed('1')):
            msg.large_gain = True
        if (keyboard.is_pressed('2')):
            msg.small_gain = True
        if (keyboard.is_pressed('p')):
            msg.small_push = True
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    key_publisher()