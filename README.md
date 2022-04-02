# Human Detection System and Robot Safety System for Thorvald robots

This repository contains a ROS package that allows the Thorvald rotos to detect and safely interact with humans during logistic operations at polytunnels. The package includes a human detection system based on 2D LiDARs, RGB-D, and thermal cameras. The human information extracted from these sensors (humans position, motion, orientation, gesture) is used as input for a safety system that can perform safety actions such as 1) modify the robot's current goal (the goals are nodes in a topological map of the field) according to human commands (i.e. activated by gesture recognition),  2) controlling/modulating the robot velocities according to gesture recognition and/or the distance between the robot and human (when the robot is moving towards the human) 3) stop completely any operation when a human is detected within unsafe distance respect to the robot (in the case of UV-C treatment a safety distance is 7m, in the case of logistics it is 1.2m), 4) Activate audiovisual alerts in order to warn the human about the robot's current action and potential danger (critical during UV-C treatment). The components and overall system architecture are shown in the following scheme. A video with a demo can be found [here](https://www.youtube.com/watch?v=vIdlauwlmKo).

<img src=/images/Human_perception_and_safety_system_new.png width="900">


# How the Human Detetion works:
* The human detection system is based on information taken from two 2D LiDARs (the same used for robot localization), two RGBD cameras (realsense D455) one aligned to the robot's local x-axis (to detect a human in front of the robot), and the other one in the opposite direction (to detect a human in the back of the robot). Moreover, a thermal camera (FLIR Lepton 3.5) was mounted together with each RBGD camera using a 3D printed base in order to match the images extracted from both cameras.
* To start publishing human information, the detection system requires to be subscribed to topics published by 2D LiDARs nodes or RGBD cameras nodes, or both in the best case (for sensor fusion). When only LiDAR data is available, the human information contains only (x,y) position, a label with the motion estimation (if the human is static or not static), and the number of the area around the robot in which the human is being detected (there is a total of 10 areas around the robot, 5 correspond to the front of the robot and 5 to the back). When the RGBD data is available, then apart from the information above, the detection system delivers a human orientation label (if the human is facing or not the robot), a human posture label (if the human is performing a specific body gesture). If Thermal cameras are also publishing data (it is not mandatory), then, this information is used to robustify the human detection ( based only on RBD images) and remove false positives.  The thermal information is also valuable for UV-C treatment scenarios when even if a human skeleton is not detected, but If a certain percentage of the pixel on the thermal image is higher than a threshold, then a thermal detection label is activated to alert the safety system that the robot must pause operation.
* The name of the labels corresponding to the human body gestures that can be interpreted by the robot are: `"no_gesture","left_arm_up","left_hand_front","left_arm_sideways","left_forearm_sideways","right_arm_up","right_hand_front","right_arm_sideways","right_forearm_sideways","both_arms_up","both_hands_front","crouched_down","picking"`. The following figure shows examples of each gesture label.

<img src=/images/gesture_examples.png width="700">

* The name of the labels corresponding to the human motion are: `"not_defined","mostly_static", "moving"`.
* The name of the labels corresponding to the human orientation are: `"facing_the_robot", "giving_the_back", "left_side", "right_side"`.
* The following figure illustrates the distribution of the areas around the robot which are used for sensor fusion and safety purposes. This image illustrates a thorvald robot moving along polytunnels where the areas from 0 to 4 correspond to frontal areas (x is positive) and from 5 to 9 correspond to back areas (x is negative). The angle `a` is a configuration parameter set by the user and the parameter `w` is chosen according to the distance between crop rows. Human detected in green areas are not considered critical, but humans in red areas are particulary critical. The same areas distribution is used when a robot navigates outside polytunnels, i.e. giving priority to humans detected directly in front or back to the robot over the ones detected sideways or walking next to it.

<img src=/images/area_distribution.png width="400">

# How the Decision Making works:
* The decision-making controls the behavior of the safety system based on safety policies (determined during a Hazard Analysis stage) and information delivered by the human detection system and the Thorvald navigation system.
* The safety system must always be publishing messages, even if no safety action is required. If the safety topics stop being published for a specific period of time (e.g. if the safety system node stopped suddenly), the decision-making makes the current robot action stop and activates audiovisual alerts to warn the human know that the safety system is not running. When the safety system starts publishing again, the previous robot action is resumed.
* Similar to the safety system, the human detection system is always publishing messages, even if no human is detected. Thus, If human information is not being published for a while, the safety system makes the current robot action stop and activates audiovisual alerts to warn the human that the robot perception is not running. The robot can resume the previous action only when the human detection system is running again.
* If teleoperation mode is activated, then, any robot's autonomous action is stopped (as the standard behavior of Thorvald robots) and audiovisual alerts indicate to the human that the robot is being controlled by the joystick. When the autonomous mode is activated again, the previous action is not resumed and the robot keeps on pause, till the human gives a new command/goal.
* The audio alerts correspond to explicit voice messages recorded in two languages which are reproduced in a loop till a new safety action updates the message (there is a specific message for each safety action and robot operation). The visual alerts are performed by a colored beacon which changes the color according to the distance between the robot and the human detected and blinks in different ways in case the safety system is running properly or if a failure is detected.
* The information being published by the safety system include safety action label, voice message label, human command label, risk level label, operation mode label, action mode label, topological goal label, critical human detected index. 
* The name of the labels corresponding to the safety actions published by the safety system are: `"move_to_goal","approach_to_human","move_away_from_human","pause","wait_for_new_human_command","teleoperation","gesture_control","no_safety_action"`.
* The name of the labels corresponding to the audio/voice messages are: `"no_message","alert_UVC_danger","ask_for_next_action","ask_for_free_space","alert_approaching","alert_moving_away","alert_moving_to_goal","safety_system_error","human_perception_error","teleoperation_mode","gesture_control_mode"`.
* The name of the labels corresponding to human commands performed by body gesture are:
`"no_command","approach","move_away","stop","move_forwards","move_backwards","move_right","move_left"`.
* The name of the labels corresponding to the level of risk during the Human-Robot Interaction (HRI) are:
`"no_human","safety_hri","risky_hri","dangerous_hri"`.
* The operation mode labels can be only `"logistics"` or `"UV-C_treatment"`. Depending on which operation is selected, the safety system behavior will be different since the safety policies for each application are different.
* The action mode labels can be `"polytunnel"` or `"footpath"`. They are updated depending if the robot is navigating outside the polytunnel or along a row inside the polytunnel. If the action mode is `"polytunnel"` it means that the human gestures are limited to commands that make the robot stop, approach him/her, or make the robot move away. If the action mode is `"footpaths"` it means that the human gestures can also control the robot's motion in any direction outside the polytunnel, including moving sideways. The robot action activated by a gesture outside the polytunnel is valid only while the gesture is being detected. On the other hand, any robot action activated by a gesture inside the polytunnel is still valid after the human stop performing that gesture.
* The robot topological goal can be set as in the standard Thorlvald navigation, i.e. by using rviz interface (clicking on a node). However, when the safety system is being used, this goal can be modified in some situations. For instance, if the robot is navigating in `"polytunnel"` mode, and the human performs gestures to make the robot move away or approach him/her, then the robot's goal is updated in order to make the robot move in the proper direction according to the human command.
* The human critical index corresponds to the index of the element in the human detected list which represent the most critical human to be tracked. This is critical when more than one human is detected at the same time. The selection of the most critical human to be tracked depends on the area in which the humans are located, the distance between them and the robot, and if they are performing body gestures or not.

SAFETY POLICIES:

In order to minimize the risk of getting human injuries during HRIs, the following safety policies were considered for the decision-making:

* The distances "d_log" used to classify the risk of collisions during logistics operations are: 0<d_log<1.2m (`"dangerous_hri"`), 1.2m<d_log<3.6m (`"risky_hri"`), d_log>3.6m (`"safety_hri"`).
* The distances "d_uvc" used to classify the risk of getting human injuries during UV-C treatment are: 0<d_uvc<7m (`"dangerous_hri"`) , 7m<d_uvc<10m (`"risky_hri"`), d_uvc>10m (`"safety_hri"`).
* The robot must always stop any action in any operation (`"logistics"` or `"UV-C_treatment"`) if the risk label is `"dangerous_hri"`.
* If the robot is in `"logistics"` operation, inside or outside polytunnels, and the human is performing gestures to command the robot to approach to him/her, then the robot speed must be reduce proportionally to the distance between them and stop completaly when the risk becomes `"dangerous_hri"`.
* If the robot is in `"logistics"` operation, outside polytunnels (`"footpahts"`), and the human is performing gestures to command the robot to approach him/her, then before the robot starts moving towards the human, it must be reoriented and approach the human always facing it directly (this includes moving forwards or backward towards the human).
* If the robot is in `"logistics"` operation, inside or outside polytunnels, and moving to a current goal but a human is detected occluding the path, then the robot safety action becomes `"pause"` and stays like this till the human is no longer detected or when is no longer in `"risky_hri"`. However, the `"pause"` action is only valid if the human is occluding the robot's path, i.e. a `"pause"` is not activated if the human is in `"risky_hri"` but is located on the back of the robot when the robot is moving forwards, or when the human is in front of the robot and the robot is moving backward. Moreover, If one or more humans are detected but they are not located at areas 2 or 7 (red color areas), they do not activate `"pause"` action since humans walking in another row or at the side of the robot are not considered critical.
* If the robot is in `"logistics"` operation, inside or outside polytunnels, and a human detected is performing a body gesture which correspond to a command, such command is only valid if the human orientation label is `"facing_the_robot"`. 
* If the robot is in `"logistics"` operation, inside polytunnels, and is performing actions to move towards a human, the robot must stop immediately if the human motion label turns into `"moving"` or if the human orientation label is not `"facing_the_robot"`.

# How to use de MeSAPro code:
PREREQUISITES:

The mesapro package requires the following packages/dependencies in order to be used in the Thorvald robots. Make sure that all packages are cloned into the directory `~/<workspace name>/src` where `<workspace name>` is your workspace name (the default name used in the Thorvald robots is `rasberry_ws`).

1. Install ROS Melodic following the steps shown [here](http://wiki.ros.org/melodic/Installation/Ubuntu).
2. Clone the [LCAS/RASberry](https://github.com/LCAS/RASberry) repository and install all the dependencies. This repository contains the necessary packages to interface with the Thorvald robots and to make them navigate autonomously. Note that the RASberry repository is private, so make sure you have access to it. A detailed installation guideline can be found [here](https://github.com/LCAS/RASberry/wiki/RASberry-Setup).  
3. Clone the [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose.git) repository. This repository contains the so-called Openpose framework used to extract human skeleton features based on RBD images. Make sure to download and install the OpenPose prerequisites for your particular operating system (e.g. cuda, cuDNN, OpenCV, Caffe, Python). Follow the instructions shown [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md).
4. Clone the [LCAS/people-detection](https://github.com/LCAS/people-detection.git) repository. This repository contatins a LiDAR based leg detector. This repository is also private, so make sure you have access to it and follow the installation instructions shown in the README section.
5. Clone the flir_module_driver repository. This repository contains the drivers to interface with FLIR lepton cameras. 
6. Clone the [IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros) repository. This repository contains the necessary packages for using Intel RealSense cameras (D400 series SR300 camera and T265 Tracking Module) with ROS. Follow the installation instructions shown in the README section.
7. Clone the [ds4_driver](https://github.com/naoki-mizuno/ds4_driver) repository. This repository contains the drivers to interface with a Ps4 joystick (this is only used for simulation purposes). To install it correctly, make sure you follow the instructions shown in the README section.
8. Clone the [ros_numpy](https://github.com/eric-wieser/ros_numpy.git) repository. This repository contains a tools for converting ROS messages to and from numpy arrays. 
9. Clone the [LCAS/topological_navigation](https://github.com/LCAS/topological_navigation) repository. This repository contains the basic packages used for the topological navigation of Thorvald robots.
10. Install [rosserial_arduino](http://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup) to allows your Arduino to be a full fledged ROS node which can directly publish and subscribe to ROS messages. Follow the installation instructions shown in the README section.
11. Finally, clone the mesapro repository, and make sure to build and source the workspace.


HOW TO USE IT:
* There are 3 important config files (into the `/mesapro/tmule` folder) that must be launched in order to run everything shown on the system architecture scheme. If implementing it on the Thorvald-014 (the one used during the whole MeSAPro project), the `rasberry-hri_navigation.yaml` is launched on the NUC (computer without GPU, used as master), the `rasberry-hri_safety_perception.yaml` is launched on the ZOTAC (computer with GPU), and the `rasberry-hri_monitoring.yaml` can be launched in any laptop in order to visualize and monitor the robot localization, human detections and safety actions.
* The config files into `tmule` folder have several parameters that can be modified in case some features of the human detection or safety system are not required for certain tests. For instance, the leg detection can be disabled to test only camera detections, the thermal information can be disable, audio or visual alerts can be disabled if neccesary, etc. Moreover, the directories of important packages/files can be modified using these configuration parameters, e.g. the bag files directory, the workspace directory, the anaconda directory, the OpenPose directory, etc.
* To test the safety system in simulation (GAZEBO), you can launch the config file `rasberry-hri_sim.yaml`. In this simulation, the human perception system is replaced by a node that is publishing the human information of two simulated people commanded by a joystick.
* To test the human detection system (based only on camera data) using bag files, you can launch the config file `rasberry-hri_camera_detector.yaml`.
* To test the human detection system (based only on LiDAR data) using bag files, you can launch the config file `rasberry-hri_leg_detector.yaml`.
* To use the human gesture recognition feature for first time, it is necessary to uncompress the file which contains the trained model. This file is located in the `config` folder.

To launch any config file into the `/mesapro/tmule` folder, it is necessary to first install Tmule-TMux Launch Engine with `pip install tmule` (source code [here](https://github.com/marc-hanheide/TMuLE)), and execute the following commands in terminal:
```
roscd mesapro/tmule
tmule -c <config_file_name>.yaml launch
```
To terminate the execution of a specific tmule session:
```
tmule -c <config_file_name>.yaml terminate
```
To monitor the state of every panel launched for the current active tmule sessions:
```
tmux a
```

# Notes: 
* The creation of this package was motivated by the MeSAPro project which aims to ensure the autonomy of agricultural robots in scenarios that involve human-robot interactions. The decision-making and safety policies of the safety system were designed to be implemented mainly during logistics operations at polytunnels (especially the gesture control features), however, some of the safety features (audiovisual alerts and safety stops) are still relevant during UV-C treatment operations.

