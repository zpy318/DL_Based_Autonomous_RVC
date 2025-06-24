# Deep Learning-Based Autonomous Retinal Vein Cannulation in ex vivo Porcine Eyes
# Introduction
This repository contains all necessary code and data for reproducing the training results and control algorithms for the paper titled "Deep Learning-Based Autonomous Retinal Vein Cannulation in ex vivo Porcine Eyes". Please download the dataset_for_all from the [OneDrive](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/pzhang24_jh_edu/EWAFqFn2MxxNixvxJ3GuBnoBuukORTDbnfWcCOmLmIoQ_Q?e=rbJPwk) to the root directory. 

# Installation
We recommend running this in a virtual environment:
```
# generate a virtual environment with name test_env and Python 3.8.16 installed
conda create -n test_env python=3.8.16
# activate the environment
conda activate test_env
# deactivate the environment
conda deactivate
# delete the virtual environment and all its packages
conda remove -n test_env --all
```
To install all necessary packages, please navigate to the cloned directory and run the following code in the terminal:
```
pip install -r requirements.txt
```

# Navigation Network
Train the navigation network using [navigation_network_train.ipynb](https://github.com/zpy318/DL_Based_Autonomous_RVC/blob/main/navigation_network/navigation_network_train.ipynb). Validate and test accuracy using [navigation_network_val_test.ipynb](https://github.com/zpy318/DL_Based_Autonomous_RVC/blob/main/navigation_network/navigation_network_val_test.ipynb).

# Contact Network
Train, validate, and test the contact network using [contact_network_train_val_test.ipynb](https://github.com/zpy318/DL_Based_Autonomous_RVC/blob/main/contact_network/contact_network_train_val_test.ipynb).

# Puncture Network
Train the puncture network using [puncture_network_train.ipynb](https://github.com/zpy318/DL_Based_Autonomous_RVC/blob/main/puncture_network/puncture_network_train.ipynb). Validate and test accuracy using [puncture_network_val_test.ipynb](https://github.com/zpy318/DL_Based_Autonomous_RVC/blob/main/puncture_network/puncture_network_val_test.ipynb).

# SHER Visualiaztion and Control
**The code in this folder was developed for Steady Hand Eye Robot (SHER) at Johns Hopkins University only.** If you need to use the code for your own robot system, please adjust the code accordingly. Following the below steps in the terminal to achieve autonomous retinal vein cannulation with fixed eyes and vertical sinusoidal motion and compensation.

* subscribe microscope image through HD SDI output and publish to /decklink/camera/image_raw ros topic:
```
cd ./gscam/
source devel/setup.bash
roslaunch gscam gscam_decklink.launch
```

* run the keyboard contorller code:
```
sudo -s
cd ./keyboard_controller/
source devel/setup.bash
# publish the key commonds
rosrun key_publisher key_publisher.py
```
* open a new terminal:
```
cd ./keyboard_controller/
source devel/setup.bash
rosrun key_move key_move.py 
```

## For Autonomous RVC with Fixed Eyes
```
# publish B-scans
python3 visualization_b_scans.py 
# visualization of the network output on microscope image and B-scan
python3 visualization.py
# control and inference
python3 visualization_inference.py
# data saving
python subscribe_all.py
```

## Autonomous RVC with Vertical Sinusoidal Motion and Compensation
```
# publish B-scans
python3 publish_b_scans_for_breath_simulation.py
# visualization of the network output on microscope image and B-scan
python3 visualization.py
# initialize  the XYZ linear stage
python xyz_stage_move_to_position.py
# simulate the breathing using the XYZ linear stage
python xyz_stage_rvc_z_axis_sinewave_motion_recording.py
# control and inference
python3 breath_simulation_inference.py
# data saving
python subscribe_all.py
```

## For Breathing Simulation Only - Show How It Works
```
python3 breath_simulation.py 
```
