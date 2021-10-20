#! /usr/bin/python

#required packages
import rospy #tf
import geometry_msgs.msg
from geometry_msgs.msg import PoseArray
import message_filters #to sync the messages
import sys
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import os
import yaml
from mesapro.msg import human_msg, hri_msg, robot_msg
##########################################################################################

#Importing global parameters from .yaml file
src_direct=os.getcwd()
config_direct=src_direct[0:len(src_direct)-9]+"config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
direct_param=list(dict.items(parsed_yaml_file["directories_config"]))
ar_param=list(dict.items(parsed_yaml_file["action_recog_config"]))
hs_param=list(dict.items(parsed_yaml_file["human_safety_config"]))
#Setup ROS publiser
pub_robot = rospy.Publisher('robot_info', robot_msg)
rob_msg = robot_msg()
pub_human = rospy.Publisher('human_info', human_msg)
msg = human_msg()
pub_hz=0.01 #publising rate in seconds
#General purposes variables
main_counter=0
new_data=[0] #flag to know if the human_msg or robot_msg has been received
demo=1 #which demo is been simulated?
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        self.position_x=0
        self.position_y=0
        self.posture=0 #from camera [posture_label,posture_probability]
        self.orientation=0 # it can be "front" or "back" if the human is facing the robot or not , from camera
        self.distance=0  # distance between the robot and the average of the skeleton joints distances taken from the depth image, from camera
        self.sensor=0 # it can be 0 if the data is from camera and Lidar, 1 if the data is  only from the LiDAR or 2 if the data is only from de camera
        self.sensor_t=[0,0]
        self.critical_index=0 #index of the closest human to the robot
        self.motion=0 #from lidar + camara

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.position=np.zeros([3,1]) #[x,y,theta]
        self.inputs=np.zeros([2,1]) #w,v control signals
        self.operation=1 #["UV-C_treatment","moving_to_picker_location", "wait_for_command_to_approach", "approaching_to_picker","wait_for_command_to_move_away", "moving_away_from_picker"]

class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=0    
        self.human_command=0
       
def safety_callback(safety_info):
    if new_data[0]==0:
        hri.status=safety_info.hri_status
        hri.audio_message=safety_info.audio_message
        hri.safety_action=safety_info.safety_action
        hri.human_command=safety_info.human_command
        new_data[0]=1
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node       
    hri=hri_class()  
    robot=robot_class()
    if demo==1:
        rospy.init_node('demo_perception',anonymous=True)
        # Setup and call subscription
        rospy.Subscriber('human_safety_info',hri_msg,safety_callback)  
        #Rate setup
        rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
        while not rospy.is_shutdown():	      
            main_counter=main_counter+1   
            ############################################################################################
            if new_data[0]==1:
                if robot.operation==1: #if robot is moving to picker location 
                    if hri.status==2: #if distance <=3.6m
                        robot.operation=2 #wait for command to approach
                    if hri.human_command==1: #if human command is "approach"
                        robot.operation=3 
                    if hri.human_command==2: #if human command is "move_away"
                        robot.operation=5
                    if hri.human_command==3: #if human command is "stop"
                        robot.operation=2 #wait for command to approach
                
                elif robot.operation==2: #if robot is waiting for a human command to approach
                    if hri.human_command==1: #if human command is "approach"
                        robot.operation=3 
                    if hri.human_command==2: #if human command is "move_away"
                        robot.operation=5
                elif robot.operation==3: #if robot is approaching to picker (already identified)
                    if hri.status==3: #if distance <=1.2m
                        robot.operation=4 #wait for command to move away
                    if hri.human_command==2: #if human command is "move_away"
                        robot.operation=5
                    if hri.human_command==3: #if human command is "stop"
                        robot.operation=4 #wait for command to move away
                
                elif robot.operation==4: #if robot is waiting for a human command to move away
                    if hri.human_command==2: #if human command is "move_away"
                        robot.operation=5
                elif robot.operation==5: #if robot is moving away from the picker (it can be after collecting tray or because of human command)
                    if hri.human_command==1: #if human command is "approach"
                        robot.operation=3 
                    if hri.human_command==3: #if human command is "stop"
                        robot.operation=2 #wait for command to approach
                else: #uv-c treatment
                    robot.operation=0
                
                    
                    
                new_data[0]=0
        ############################################################################################
            #Publish Robot new operation
            rob_msg.operation=robot.operation
            pub_robot.publish(rob_msg)        
            rate.sleep() #to keep fixed the publishing loop rate
    else:
        human=human_class()
        rospy.init_node('demo_actuation',anonymous=True)
        # Setup and call subscription
        rospy.Subscriber('human_safety_info',human_msg,safety_callback)  
        #Rate setup
        rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
        while not rospy.is_shutdown():	      
            main_counter=main_counter+1   
            ############################################################################################
            if new_data[0]==1:
                    
                new_data[0]=0
        ############################################################################################
            #Publish Robot new operation
            rob_msg.operation=robot.operation
            pub_robot.publish(rob_msg)        
            #Publish Robot new operation
            msg.posture = human.posture_track[0]
            msg.motion = human.motion_track[0]
            msg.position_x = human.position_x[0]
            msg.position_y = human.position_y[1]
            msg.distance = human.distance_track[0]
            msg.orientation = human.orientation_track[0]
            msg.sensor = human.sensor[0]
            msg.sensor_t0=human.sensor_t[0]
            msg.sensor_t1=human.sensor_t[1]
            msg.critical_index = human.critical_index
            pub_human.publish(msg)        
            
            rate.sleep() #to keep fixed the publishing loop rate