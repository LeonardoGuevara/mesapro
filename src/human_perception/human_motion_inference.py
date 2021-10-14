#! /usr/bin/python

#required packages
import rospy #tf
from math import * #to avoid prefix math.
import numpy as np #to use matrix
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
pub = rospy.Publisher('human_info', human_msg)
msg = human_msg()
pub_hz=0.01 #publising rate in seconds
#Feature extraction variables
posture_labels=ar_param[2][1]
motion_labels=ar_param[3][1]
orientation_labels=ar_param[4][1]
#General purposes variables
main_counter=0
n_points=400 #number of points to be used to construct the human trajectory to be analized, e.g. 400 means 4sec if dt=0.01
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        #Variables to store temporally the info of human detectection at each time instant
        self.n_human=1 # considering up to 1 human to track initially
        self.motion=0 #from lidar + camara
        
        self.distance=np.zeros([self.n_human,1])
        self.position=np.zeros([self.n_human,2])
        self.posture=np.zeros([self.n_human,1])
        self.orientation=np.zeros([self.n_human,1])
        self.sensor=np.zeros([self.n_human,1])
        
        self.distance_buffer=np.zeros([self.n_human,n_points,2]) #record of distances from the critical humans, initially only 1 human is recorded
        self.position_x_buffer=np.zeros([self.n_human,n_points,2])
        self.position_y_buffer=np.zeros([self.n_human,n_points,2])  
        self.posture_buffer=np.zeros([self.n_human,1])
        self.orientation_buffer=np.zeros([self.n_human,1])
        self.sensor_buffer=np.zeros([self.n_human,n_points,1]) # it can be 0 if the data is from camera and Lidar, 1 if the data is  only from the LiDAR or 2 if the data is only from de camera
        self.counter=np.zeros([self.n_human,1]) #counter vector to determine for how many cycles a human tracked was not longer detected, counter>0 means is not longer detected, counter<=0 means is being detected
        
        
def human_callback(human_info):
    #print("LEGS",legs.poses)
    print(len(human_info.sensor))
    if len(human_info.sensor)!=0: #only if there is a new human_info.msg data
        index=human_info.critical_index
        print(np.array(human_info.distance))
        human.distance=np.array(human_info.distance)[index]
        human.position_x=np.array(human_info.position_x)[index]
        human.position_y=np.array(human_info.position_y)[index]
        human.posture=np.array(human_info.posture)[index]
        human.orientation=np.array(human_info.orientation)[index]
        human.sensor=np.array(human_info.sensor)[index]
       


###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node       
    human=human_class()  
    rospy.init_node('human_motion_inference',anonymous=True)
    # Setup and call subscription
    #image_sub = message_filters.Subscriber('camera/camera1/color/image_raw', Image)
    #depth_sub = message_filters.Subscriber('camera/camera1/aligned_depth_to_color/image_raw', Image)
    #ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.01)
    #ts.registerCallback(camera_callback)
    rospy.Subscriber('human_info',human_msg,human_callback)  
    #rospy.spin()
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	      
        main_counter=main_counter+1       
       
        human.motion=human.motion+1
        #Publish     
        msg.motion = human.motion
        pub.publish(msg) 
        rate.sleep() #to keep fixed the publishing loop rate

