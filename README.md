# Human Detection System and Robot Safety System for Thorvald robots

This repository contains a ROS package for human-aware navigation of agricultural robots. The package includes a human detection system based on 2D LiDAR + RGB-D information mounted on a Thorvald robot. The human information is an input for a safety system which: 1) modifies the robot route according to safety policies and human commands (activated by gesture recognition), and 2) controls/modulates the robot speed in order to keep a safety distance between the robot and the human detected. The overall system arquitecture is shown as follows:

![Overall_system](/Human_perception_and_safety_system.png)

The creation of this package was motivated by the MeSAPro project which aims to ensure autonomy of agricultural robots in scenarios which involves human-robot interactions. The decision making and safety policies of the current version of the safety system were implemented only in transportation robots during harvesting at polytunnels. However, it is planned to create alternative versions for other robot tasks covered by the MeSAPro project such as UV-C treatment and automatic data collection.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/U68gQpjClaIE/0.jpg)](https://youtu.be/U68gQpjClaI)
