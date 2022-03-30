# Human Detection System and Robot Safety System for Thorvald robots

This repository contains a ROS package that allows the Thorvald rbotos to detect and safely interact with humans during logistic operations at polytunnels. The package includes a human detection system based on 2D LiDARs, RGB-D, and thermal cameras. The human information extracted from these sensors (humans position, motion, orientation, gesture) is used as input for a safety system that can perform safety actions such as 1) modify the robot's current goal (the goals are nodes in a topological map of the field) according to human commands (i.e. activated by gesture recognition),  2) controlling/modulating the robot velocities according to gesture recognition and/or the distance between the robot and human (when the robot is moving towards the human) 3) stop completely any operation when a human is detected within unsafe distance respect to the robot (in the case of UV-C treatment a safety distance is 7m, in the case of logistics it is 1.2m), 4) Activate audiovisual alerts in order to warn the human about the robot's current action and potential danger (critical during UV-C treatment). The components and overall system architecture are shown in the following scheme (if you click on the image it will direct you to a demo video).

[![Overal_system](/Human_perception_and_safety_system_new.png)](https://www.youtube.com/watch?v=vIdlauwlmKo)

HOW THE HUMAN DETECTION WORKS:

* To get human information, the detection system requires to be subscribed to topics published by 2D LiDARs nodes or RGBD cameras nodes, or both in the best case (for sensor fusion). When only LiDAR data is available, the human information contains only (x,y) position, a label with the motion estimation (if the human is static or not static), and the number of the area around the robot in which the human is being detected (there is a total of 10 areas around the robot, 5 correspond to the front of the robot and 5 to the back). When the RGBD data is available, then apart from the information above, the detection system delivers a human orientation label (if the human is facing or not the robot), a human posture label (if the human is performing a specific body gesture). If Thermal cameras are also publishing data (it is not mandatory), then, this information is used to robustify the human detection ( based only on RBD images) and remove false positives.  The thermal information is also valuable for UV-C treatment scenarios when even if a human skeleton is not detected, but If a certain percentage of the pixel on the thermal image is higher than a threshold, then a thermal detection label is activated to alert the safety system that the robot must pause operation.
* The name of the labels corresponding to the human body gestures that can be interpreted by the robot are: "no_gesture","left_arm_up","left_hand_front","left_arm_sideways","left_forearm_sideways","right_arm_up","right_hand_front","right_arm_sideways","right_forearm_sideways","both_arms_up","both_hands_front","crouched_down","picking".
* The name of the labels corresponding to the human motion are: "not_defined","mostly_static", "moving".
* The name of the labels corresponding to the human orientation are: "facing_the_robot", "giving_the_back", "left_side", "right_side".
* The following figures illustrate the distribution of the areas around the robot (used for sensor fusion and safety purposes) and show samples of the body gestures mentioned above.

HOW THE DECISION MAKING WORKS:

HOW TO USE THE MESAPRO PACKAGE:

* To run the human detection and safety system on the Thorvald robots. You have to launch the config files into the folder "tmule". There are 3 config files that must be launched in order to run everything shown on the system architecture scheme. If implementing it on the Thorvald-014 (the one used during the whole MeSAPro project), the rasberry-hri_navigation.yaml is launched on the NUC (computer without GPU, used as master), the rasberry-hri_safety_perception.yaml is launched on the ZOTAC (computer with GPU), and the rasberry-hri_monitoring.yaml can be launched in any laptop in order to visualize and monitor the robot localization, human detections and safety actions.

* To test the safety system in simulation (GAZEBO), you can launch the config file rasberry-hri_sim.yaml .
* To test the human detection system (based only on camera data) using bag files, you can launch the config file rasberry-hri_camera_detector.yaml .
* To test the human detection system (based only on LiDAR data) using bag files, you can launch the config file rasberry-hri_leg_detector.yaml .

NOTES: 
* The creation of this package was motivated by the MeSAPro project which aims to ensure autonomy of agricultural robots in scenarios that involve human-robot interactions. The decision-making and safety policies of the current version of the safety system are designed to be implemented mainly during logistics operations at polytunnels, but can also be used during UV-C treatment operations.
* To use the mesapro package, it is required to also have all the packages into the LCAS/RASberry repository which includes among others, the topological navigation package which is used to make the Thorvald robots navigate in an autonomous way at polytunnels (Note: The RASberry repository is private, so make sure you have access to it).
* To launch any config file into the tmule folder, it is necesary to install "tmule".
