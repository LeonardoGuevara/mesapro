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
config_direct=src_direct[0:len(src_direct)-16]+"config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
direct_param=list(dict.items(parsed_yaml_file["directories_config"]))
ar_param=list(dict.items(parsed_yaml_file["action_recog_config"]))
hs_param=list(dict.items(parsed_yaml_file["human_safety_config"]))
#Setup ROS publiser
pub_safety = rospy.Publisher('human_safety_info', hri_msg)
safety_msg = hri_msg()
pub_hz=0.01 #publising rate in seconds
#Extraction of Human variables
motion_labels=ar_param[3][1]
hri_status_label=hs_param[0][1]
audio_message_label=hs_param[1][1] 
safety_action_label=hs_param[2][1]
human_command_label=hs_param[3][1]
#General purposes variables
main_counter=0
new_data=[0,0] #flag to know if the human_msg or robot_msg has been received
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.position=np.zeros([3,1]) #[x,y,theta]
        self.operation=2 #["UV-C_treatment","moving_to_picker_location", "wait_for_command_to_approach", "approaching_to_picker","wait_for_command_to_move_away", "moving_away_from_picker"]
        
class human_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.posture=[0]
        self.motion=[0] 
        self.position_x=[0] 
        self.position_y=[0] 
        self.distance=[0]
        self.sensor=[0]
        self.sensor_t0=[0]
        self.sensor_t1=[0]
        self.critical_index=0
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=0     
        self.human_command=0

def human_callback(human_info):
    if len(human_info.sensor)!=0 and new_data[0]==0: #only if there is a new human_info data
        human.posture=human_info.posture
        human.motion=human_info.motion   
        human.position_x=human_info.position_x
        human.position_y=human_info.position_y
        human.distance=human_info.distance
        human.sensor=human_info.sensor
        human.sensor_t0=human_info.sensor_t0
        human.sensor_t1=human_info.sensor_t1
        human.critical_index=human_info.critical_index
        new_data[0]=1

def robot_callback(robot_info):
    if new_data[1]==0: #only if there is a new human_info data
        robot.position=np.array([robot_info.position_x,robot_info.position_y,robot_info.orientation])
        robot.operation=robot_info.operation
        robot.operation_new=robot_info.operation
        new_data[1]=1
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
    rospy.Subscriber('human_info',human_msg,human_callback)  
    rospy.Subscriber('robot_info',robot_msg,robot_callback)  
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	      
        main_counter=main_counter+1   
        ############################################################################################
        if new_data[0]==1 or new_data[1]==1:
            ##hri_status############
            if len(human.sensor)==1 and (human.posture[0]+human.position_x[0]+human.position_y[0]+human.distance[0])==0: #None human detected
                #print("HOLAAAA")
                #print(human.distance)
                hri.status=0
                hri.audio_message=0
                hri.safety_action=0
                hri.human_command=0 
            else: #if at least one human was detected
                if human.sensor[human.critical_index]==1: #if data is from lidar 
                    distance=sqrt((human.position_x[human.critical_index])**2+(human.position_y[human.critical_index])**2)
                elif human.sensor[human.critical_index]==2: #if data is from camera
                    distance=human.distance[human.critical_index]    
                else: #data from lidar + camera
                    if human.sensor_t0[human.critical_index]>=human.sensor_t1[human.critical_index]: #if data from lidar is newer than from camera
                        distance=sqrt((human.position_x[human.critical_index])**2+(human.position_y[human.critical_index])**2)
                    else: #if data from camera is newer than from lidar
                        distance=human.distance[human.critical_index] 
                ###UV-C treatment#######################################
                print("Distance",distance)
                if robot.operation==0:
                    hri.human_command=0 #no human command expected during uv-c treatment
                    if distance>9: #if human is above 7m from the robot
                        hri.status=1
                        hri.audio_message=6
                        hri.safety_action=0
                    elif distance>7 and distance<=9: # if human is between 7-9m
                        hri.status=2
                        hri.audio_message=6
                        hri.safety_action=0
                    else: #if human is within 7m
                        hri.status=3
                        hri.audio_message=6
                        hri.safety_action=2
                ##LOGISTICS###############################################
                if robot.operation>=1: 
                    if distance>7: #if human is above 7m
                        if robot.operation!=3: #if robot is not performing approaching maneuvers
                            hri.human_command=0
                            hri.status=1
                            hri.audio_message=0
                            hri.safety_action=0
                        else: # if robot is performing approaching maneuvers, make it move slow and activate alert
                            hri.human_command=0
                            hri.status=1
                            hri.safety_action=1 #make the robot reduce speed 
                            hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                        #CASES WHEN HUMAN PERFORM A BODY GESTURE
                        #In case the picker wants the robot to approch to him/her before the robot ask for permission
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==1: #picker is ordering the robot to approach (using both arms)
                            hri.human_command=1 #make the robot approach to the picker from this point i.e robot.operation=1
                            hri.safety_action=1 #make the robot reduce speed
                            hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                        #In case the picker wants the robot to stop before the robot approach more to him/her
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==8:  #picker is ordering the robot to stop (using right hand)
                            robot.human_command=3 #make the robot stop 
                            hri.safety_action=3 #waiting new command          
                            hri.audio_message=1 #message to ask the picker for a new order to approach/move away
                        #In case the picker wants the robot to move away before the robot approach more to him/her
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                            robot.human_command=2 #make the robot move away 
                            hri.safety_action=0 #normal operation             
                            hri.audio_message=6 #alert of presence
                    elif distance>3.6 and distance<=7: #if human is between 3.6-7m
                        hri.status=1
                        #In case human is not performing any gesture
                        if (human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==0) or (human.sensor[human.critical_index]==1): 
                            hri.human_command=0 #no command
                            if robot.operation!=3: #if robot is not performing approaching maneuvers
                                hri.safety_action=0 #normal operation
                                hri.audio_message=5 #alert of presence
                            else: # if robot is performing approaching maneuvers, make it move slow and activate alert
                                hri.safety_action=1 #make the robot reduce speed 
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                        #CASES WHEN HUMAN PERFORM A BODY GESTURE
                        #In case the picker wants the robot to approch to him/her before the robot ask for permission
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==1: #picker is ordering the robot to approach (using both arms)
                            hri.human_command=1 #make the robot approach to the picker from this point i.e robot.operation=1
                            hri.safety_action=1 #make the robot reduce speed
                            hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                        #In case the picker wants the robot to stop before the robot approach more to him/her
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==8:  #picker is ordering the robot to stop (using right hand)
                            robot.human_command=3 #make the robot stop 
                            hri.safety_action=3 #waiting new command          
                            hri.audio_message=1 #message to ask the picker for a new order to approach/move away
                        #In case the picker wants the robot to move away before the robot approach more to him/her
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                            robot.human_command=2 #make the robot move away 
                            hri.safety_action=0 #normal operation             
                            hri.audio_message=6 #alert of presence
                                                  
                    elif distance>1.2 and distance<=3.6: #if human is between 1.2-3.6m
                        #NORMAL CASES
                        hri.status=2
                        #To make the robot stop and ask for permission to approach
                        if robot.operation==1 or robot.operation==2:
                            robot.human_command=0 #no command 
                            hri.safety_action=3 #waiting new command 
                            hri.audio_message=1 #message to ask the picker for a new order to approach/move away
                        #To make the robot approach with slow speed and activate alert
                        if robot.operation==3:
                            if human.motion[human.critical_index]==1: # if the human is mostly static
                                robot.human_command=0 #no command 
                                hri.safety_action=1 #make the robot reduce speed 
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                            else: #if human is not static
                                robot.human_command=3 #make the robot stop 
                                hri.safety_action=3 #waiting new command          
                                hri.audio_message=1 #message to ask the picker for a new order to approach/move away
                        #To make the robot ask for permision to move away
                        if robot.operation==4:
                            robot.human_command=0 #no command 
                            hri.safety_action=3 #waiting new command 
                            hri.audio_message=1 #message to ask the picker for a new order to approach/move away
                        #CASES WHEN HUMAN PERFORM A BODY GESTURE
                         #In case the picker wants the robot to approch to him/her 
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==1: #picker is ordering the robot to approach (using both arms)
                            hri.human_command=1 #make the robot approach to the picker from this point i.e robot.operation=1
                            hri.safety_action=1 #make the robot reduce speed
                            hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                        #In case the picker wants the robot to stop 
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==8:  #picker is ordering the robot to stop (using right hand)
                            robot.human_command=3 #make the robot stop 
                            hri.safety_action=3 #waiting new command               
                            hri.audio_message=1 #message to ask the picker for a new order to approach/move away
                        #In case the picker wants the robot to move away 
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                            robot.human_command=2 #make the robot move away 
                            hri.safety_action=0 #normal operation             
                            hri.audio_message=6 #alert of presence
                        #SPECIAL CASE WHEN THE ROBOT WAS WAITING TOO LONG AN ORDER TO MOVE AWAY 
                        if robot.operation==4 and  (human.sensor[human.critical_index]==1 or (human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]!=2)):  #picker is not ordering the robot to move away (using both hands)
                            if human.motion[human.critical_index]>=4: # if the human is moving away 
                                robot.human_command=2 #make the robot move away 
                                hri.safety_action=0 #normal operation             
                                hri.audio_message=6 #alert of presence
                        
                        
                    else: #if human is within 1.2m
                        hri.status=3 #dangerous hri
                        robot.human_command=0 #no command
                        hri.audio_message=0 
                        hri.safety_action=2 #safety stop
                        #In case the picker wants the robot to move away 
                        if human.sensor[human.critical_index]!=1 and human.posture[human.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                            robot.human_command=2 #make the robot move away 
                            hri.safety_action=1 #reduced speed        
                            hri.audio_message=6 #alert of presence
                        
            ########################################################################3
            #print("CURRENT OPERATION",robot.operation)
            #if robot.operation!=robot.operation_new:
            #    print("NEW OPERATION",robot.operation_new) 
            #    robot.operation=robot.operation_new
            ############################################################################            
            if new_data[0]==1:
                new_data[0]=0
            if new_data[1]==1:
                new_data[1]=0
        ############################################################################################
        #Publish SAFETY SYSTEM MESSAGES        
        safety_msg.hri_status = hri.status
        safety_msg.audio_message = hri.audio_message
        safety_msg.safety_action = hri.safety_action
        safety_msg.human_command = hri.human_command
        pub_safety.publish(safety_msg)

        
        rate.sleep() #to keep fixed the publishing loop rate
