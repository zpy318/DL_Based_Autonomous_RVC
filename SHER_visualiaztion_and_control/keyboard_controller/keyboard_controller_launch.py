#!/usr/bin/env python
import roslaunch
import subprocess
import stat
# from subprocess import Popen, PIPE
# subprocess.call("source devel/setup.bash", shell=True)
# os.system("roslaunch eye_camera_image_publisher eye_camera.launch")
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
launch = roslaunch.parent.ROSLaunchParent(uuid, ["./launch/keyboard_controller_launch.launch"])
launch.start()

# uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
# roslaunch.configure_logging(uuid)
# # launch_path1 = rospack.get_path('eye_camera_image_publisher') + '/eye_camera.launch'
# # launch_path2 = rospack.get_path('network_output') + '/network_output.launch'
# cli_args1 = ['eye_camera_image_publisher', 'eye_camera.launch']
# cli_args2 = ['ddp_input_publisher', 'ddp_input_publisher.launch']
# roslaunch_file1 = roslaunch.rlutil.resolve_launch_arguments(cli_args1)[0]
# roslaunch_file2 = roslaunch.rlutil.resolve_launch_arguments(cli_args2)[0]
# launch_files = [roslaunch_file1, roslaunch_file2]
# # launch_files = [roslaunch_file2]
# parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
# parent.start()

# commands = ["./silicone_eye_test/real_eye_collect_img_select_single_goal_point_thru_mouse_click.py", 
#             "./DDP/src/eye_camera_image_publisher/launch/eye_camera.launch"]
# procs = [ Popen(i, stdout=PIPE, stderr=PIPE) for i in commands ]
# for p in procs:
#    p.wait()
# subprocess.call("python ./silicone_eye_test/real_eye_collect_img_select_single_goal_point_thru_mouse_click.py", shell=True)
# subprocess.call("python ./intact_eye_test/real_eye_collect_img_predefined_20_goal_points_thru_mouse_click.py", shell=True)
# subprocess.call("python ./intact_eye_test/real_eye_collect_img_select_single_goal_point_thru_mouse_click.py", shell=True)
subprocess.call("source devel/setup.bash", shell=True)
# subprocess.call("roslaunch network_output network_output.launch", shell=True)