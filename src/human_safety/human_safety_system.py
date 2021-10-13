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
##########################################################################################

#Importing global parameters from .yaml file
src_direct=os.getcwd()
config_direct=src_direct[0:len(src_direct)-20]+"config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
direct_param=list(dict.items(parsed_yaml_file["directories_config"]))
ar_param=list(dict.items(parsed_yaml_file["action_recog_config"]))
#Setup ROS publiser
pub = rospy.Publisher('human_safety_info', human_msg)
msg = human_msg()
pub_hz=0.01 #publising rate in seconds
#Feature extraction variables
action_labels=ar_param[3][1]
#General purposes variables
main_counter=0
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.position=np.zeros([3,1]) #[x,y,theta]
        self.control=np.zeros([2,1]) #[w,v]

class human_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.action=[0] 
        self.position_x=[0] 
        self.position_y=[0] 
        self.distance=[0]
        self.sensor=[0]
        self.critical_index=0

class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.aware=0
        self.audio_message=0
        self.safety_stop=0      

def hri_callback(ros_image, ros_depth):
   


################################################################################################################                  
def hri_status(human):
    
################################################################################################################                  
def aware_status(human):

################################################################################################################                  
def audio_message(human):

################################################################################################################                  
def safety_stop(human):


###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node       
    hri=hri_class()  
    human=human_class()
    robot=robot_class()
    rospy.init_node('human_safety_system',anonymous=True)
    # Setup and call subscription
    human_sub = message_filters.Subscriber('human_info', human_msg)
    robot_sub = message_filters.Subscriber('camera/camera1/aligned_depth_to_color/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([human_sub, robot_sub], 1, 0.01)
    ts.registerCallback(hri_callback)
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	      
        main_counter=main_counter+1
       
            
        #Publish        
        msg.action = list(human.action_track[:,0])
        msg.position_x = list(human.position_track[:,0])
        msg.position_y = list(human.position_track[:,1])
        msg.distance = list(human.distance_track[:,0])
        msg.sensor = list(human.data_source[:,0])
        msg.critical_index = human.critical_index 
        pub.publish(msg)
        
        rate.sleep() #to keep fixed the publishing loop rate
