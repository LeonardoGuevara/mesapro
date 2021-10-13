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
        self.operation=0 #0 means UVC treatment, 1 means approaching to a picker, 2 moving to the picker location, 3 moving away from the picker, 4 wait for human command to approach, 5 wait for human command to move away 
        self.operation_new=0
        self.call_a_robot_goal=np.array([5,0,0])
        self.topo_goal=np.array([5,0])
        
class human_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.posture=[0]
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
    human.posture=human_info.posture
    human.motion=human_info.motion   
    human.position_x=human_info.position_x
    human.position_y=human_info.position_y
    human.distance=human_info.distance
    human.sensor=human_info.sensor
    human.critical_index=human_info.critical_index
    
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
        ############################################################################################
        ##hri_status############
        if len(human.sensor) and (human.posture*human.motion*human.position_x*human.position_y*human.distance*human.sensor*human.critical_index==0): #None human detected
            hri.status=0
            hri.audio_message=0
            hri.safety_stop=0
        else: #if at least one human was detected
            if human.sensor!=2: #if data is from lidar or lidar+camera
                distance=sqrt((human.position_x[critical_index])**2+(human.position_y[critical_index])**2)
            else: #if data is from camera
                distance=human.distance[critical_index]    
            ###UV-C treatment#######################################
            if robot.operation==0: 
                if distance>9: #if human is above 7m from the robot
                    hri.status=1
                    hri.audio_message=7
                    hri.safety_stop=0
                elif distance>7 and distance<=9: # if human is between 7-9m
                    hri.status=2
                    hri.audio_message=7
                    hri.safety_stop=0
                else: #if human is within 7m
                    hri.status=3
                    hri.audio_message=7
                    hri.safety_stop=2
            ##LOGISTICS###############################################
            if robot.operation==1: 
                if distance>7: #if human is above 7m
                    hri.status=1
                    hri.audio_message=0
                    hri.safety_stop=0
                elif distance>3.6 and distance<=7: #if human is between 3.6-7m
                    #NORMAL CASE
                    hri.status=1
                    hri.safety_stop=0
                    hri.audio_message=0
                    #SPECIAL CASES
                    #In case the picker wants identifying him/herself before the robot ask for it
                    if robot.operation==2 #if robot is moving to the picker location
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==1: #picker is identifying itself as the picker who call the robot
                            robot.operation_new=1 #make the robot approach to the picker from this point
                            hri.audio_message=6 #alert to make the picker aware of the robot approaching to him/her
                    #In case the picker wants the robot to stop before the robot approach more to him/her
                    if robot.operation==2 #if robot is moving to the picker location
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==2:  #picker is ordering the robot to stop (using both hands)
                            robot.operation_new=4 #make the robot wait till the picker perform the order to approach
                    #in case the picker make the robot stop, and now requires the robot service
                    if robot.operation==4: #if robot is waiting for human command to approach
                         if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==7:  #picker is ordering the robot to approach to him/her (using right arm)
                            robot.operation_new=1 #make the robot wait till the picker perform the order to approach
                    #In case the picker make the robot move way from him/she
                    if robot.operation==2 or robot.operation==4: #if robot is waiting for human command to approach
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==8: #picker is ordering the robot to move away (using right hand)
                            robot.operation_new=3 #to make the robot move away from the picker
                elif distance>1.2 and distance<=3.6: #if human is between 1.2-3.6m
                    #NORMAL CASE
                    hri.status=2
                    hri.safety_stop=1
                    hri.audio_message=0
                    if robot.operation==4: #if robot is waiting for human command to approach
                        hri.audio_message=1 #asking the picker if he/she called the robot
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==1: #picker is identifying itself as the picker who call the robot (using both arms)
                            robot.operation_new=1 #to make the robot approach to the picker
                        if human.sensor[human.critical_index]==1 and human.motion[human.critical_index]==0: #picker is mostly static (for at least 3 sec)
                            robot.operation_new=1 #to make the robot approach to the picker
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==8: #picker is replying that he/she is not who call the robot (using right hand)
                            robot.operation_new=3 #to make the robot move away from the picker
                    elif robot.operation==1: #if robot is approaching to the picker
                        hri.audio_message=6 #alert to make the picker aware of the robot approaching to him/her
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==2: #picker is ordering the robot to stop (using both hands)
                            robot.operation_new=5 #to make the robot wait till the picker perform the order to move away
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==8: #picker is ordering the robot to move away (using right hand)
                            robot.operation_new=3 #to make the robot move away from the picker
                    elif robot.operation==5: #if robot is waiting for the human command to move away
                        if human.sensor[human.critical_index]!=1 and (human.posture[human.critical_index]==8 | human.posture[human.critical_index]==6): #picker is ordering the robot to move away (using right hand) or picker is picking again
                            robot.operation_new=3 #to make the robot move away from the picker
                        if human.sensor[human.critical_index]==1 and human.motion[human.critical_index]==3: #robot moves away when picker is moving away slowly 
                            robot.operation_new=3 #to make the robot move away from the picker
                else: #if human is within 1.2m
                    hri.status=3
                    hri.audio_message=0
                    hri.safety_stop=2
                    #SPECIAL CASES
                    if robot.operation==1: #if robot is approaching to the picker
                        robot.operation_new=5 #to make the robot wait till the picker perform the order to move away
                    if robot.operation==2: #if robot is moving to the picker location
                        robot.operation_new=5 #to make the robot wait till the picker perform the order to move away
                    if human.sensor[human.critical_index]!=1 and (human.posture[human.critical_index]==8 | human.posture[human.critical_index]==6): #picker is ordering the robot to move away (using right hand) or picker is picking again
                        robot.operation_new=3 #to make the robot move away from the picker 
                        hri.safety_stop=1
                #0 means UVC treatment, 1 means approaching to a picker, 2 moving to the picker location, 3 moving away from the picker, 4 wait for human command to approach, 5 wait for human command to move away 
        
        #robot.operation=robot.operation_new


        
        
        ############################################################################################
        #Publish        
        msg.hri_status = hri.status
        msg.audio_message = hri.audio_message
        msg.safety_stop = hri.safety_stop
        pub.publish(msg)
       
        rate.sleep() #to keep fixed the publishing loop rate
