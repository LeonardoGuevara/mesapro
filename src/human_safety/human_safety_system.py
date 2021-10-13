#! /usr/bin/python

#required packages
import rospy #tf
import geometry_msgs.msg
from geometry_msgs.msg import PoseArray
import message_filters #to sync the messages
import sys
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from numpy import linalg as LA
import os
import yaml
from mesapro.msg import human_msg
from mesapro.msg import hri_msg
##########################################################################################

#Importing global parameters from .yaml file
src_direct=os.getcwd()
config_direct=src_direct[0:len(src_direct)-20]+"config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
direct_param=list(dict.items(parsed_yaml_file["directories_config"]))
ar_param=list(dict.items(parsed_yaml_file["action_recog_config"]))
hs_param=list(dict.items(parsed_yaml_file["human_safety_config"]))
#Setup ROS publiser
pub = rospy.Publisher('human_safety_info', hri_msg)
msg = hri_msg()
pub_hz=0.01 #publising rate in seconds
#Extraction of Human variables
action_labels=ar_param[3][1]
hri_status_label=hs_param[0][1]
audio_message_label=hs_param[1][1] 
safety_stop_label=hs_param[2][1]
#General purposes variables
main_counter=0
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.position=np.zeros([3,1]) #[x,y,theta]
        self.control=np.zeros([2,1]) #[w,v]
        self.operation=0 #0 means UVC treatment, 1 means approaching to a picker in logistics
        self.call_a_robot_goal=np.array([5,0,0])
        self.topo_goal=np.array([5,0])
        
class human_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.motion=[0] 
        self.position_x=[0] 
        self.position_y=[0] 
        self.distance=[0]
        self.sensor=[0]
        self.critical_index=0
        self.goal_index=0
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_stop=0      

def hri_callback(human_info):
    human.motion=human_info.motion   
    human.position_x=human_info.position_x
    human.position_y=human_info.position_y
    human.distance=human_info.distance
    human.sensor=human_info.sensor
    human.critical_index=human_info.critical_index
    ################################################################################################################                  
    ##hri_status############
    if len(human.sensor) and (human.motion*human.position_x*human.position_y*human.distance*human.sensor*human.critical_index==0): #None human detected
        hri.status=0
        hri.audio_message=0
        hri.safety_stop=0
    else: #if at least one human was detected
        if human.sensor!=2: #if data is from lidar or lidar+camera
            distance=sqrt((human.position_x[critical_index])**2+(human.position_y[critical_index])**2)
        else: #if data is from camera
            distance=human.distance[critical_index]    
        if robot.operation==0: #UV-C treatment
            if distance>9: #if human is above 7m from the robot
                hri.status=1
            elif distance>7 and distance<9:
                hri.status=2
            else:
                hri.status=3              
        if robot.operation==1: #logistics
            if distance
   
   
    ################################################################################################################                  
    ##audio_message##########

    ################################################################################################################                  
    ##safety_stop###########


###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node       
    hri=hri_class()  
    human=human_class()
    robot=robot_class()
    rospy.init_node('human_safety_system',anonymous=True)
    # Setup and call subscription
    #human_sub = message_filters.Subscriber('human_info', human_msg)
    #robot_sub = message_filters.Subscriber('camera/camera1/aligned_depth_to_color/image_raw', Image)
    #ts = message_filters.ApproximateTimeSynchronizer([human_sub, robot_sub], 1, 0.01)
    #ts.registerCallback(hri_callback)
    rospy.Subscriber('human_info',human_msg,hri_callback)  
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	      
        main_counter=main_counter+1   
            
        #Publish        
        msg.hri_status = hri.status
        msg.audio_message = hri.audio_message
        msg.safety_stop = hri.safety_stop
        pub.publish(msg)
       
        rate.sleep() #to keep fixed the publishing loop rate
