# mesapro
Human Detection System and Robot Safety System for Thorvald robots

This repository contains a  ROS package for human-aware navigation of agricultural robots. The package includes a human detection system based on 2D LiDAR + RGB-D information mounted on a Thorvald robot. The human information is an input for a safety system which: 1) modifies the robot route according to safety policies and human commands (activated by gesture recognition), and 2) controls/modulates the robot speed in order to keep a safety distance between the robot and the human detected. The overall system arquitecture is shown as follows:
