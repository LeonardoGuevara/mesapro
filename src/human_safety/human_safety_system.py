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
#Human selection
posture_threshold=0.6 #minimum probability from the posture recognition
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
        self.posture_prob=[0]
        self.motion=[0] 
        self.position_x=[0] 
        self.position_y=[0] 
        self.distance=[0]
        self.orientation=[0]
        self.area=[0]
        self.sensor=[0]
        self.sensor_t0=[0]
        self.sensor_t1=[0]
        self.sensor_c0=[0]
        self.sensor_c1=[0]
        self.position_x_goal=[0] #from the call-a-robot code
        self.position_y_goal=[0] #from the call-a-robot code
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=0     
        self.human_command=0
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        self.risk=0 #it is 1 if the critical human is in front of the robot (same row), or 0 if the critical human is detected in another row.
        #self.goal_index=0 #index of the human whom location is considered as goal

def human_callback(human_info):
    #print("NEW HUMAN DATA")
    if len(human_info.sensor)!=0 and new_data[0]==0: #only if there is a new human_info data
        human.posture=human_info.posture
        human.posture_prob=human_info.posture_prob
        human.motion=human_info.motion   
        human.position_x=human_info.position_x
        human.position_y=human_info.position_y
        human.distance=human_info.distance
        human.orientation=human_info.orientation
        human.area=human_info.area
        human.sensor=human_info.sensor
        human.sensor_t0=human_info.sensor_t0
        human.sensor_t1=human_info.sensor_t1
        human.sensor_c0=human_info.sensor_c0
        human.sensor_c1=human_info.sensor_c1       
        new_data[0]=1
        
def robot_callback(robot_info):
    if new_data[1]==0: #only if there is a new human_info data
        robot.position=np.array([robot_info.position_x,robot_info.position_y,robot_info.orientation])
        robot.operation=robot_info.operation
        new_data[1]=1
        
def critical_human_selection():
    sensor=human.sensor
    sensor_c0=human.sensor_c0
    sensor_c1=human.sensor_c1
    distance=human.distance
    position_x=human.position_x
    position_y=human.position_y
    posture_prob=human.posture_prob
    posture=human.posture
    operation=robot.operation
    area=human.area
    closest_distance=1000 #initial value
    closest_index=0
    n_human=len(sensor)
    #CLOSEST HUMAN TRACKED
    for k in range(0,n_human):
        #if sensor_c1[k]<0 and sensor[k]==2:# and posture_prob[k]>posture_threshold:# if data was taken from camera
        if distance[k]<=closest_distance:
            closest_index=k
            closest_distance=distance[k]
        #if sensor[k]!=2:# if data was taken from lidar or lidar+camera
        #    if sensor_c0[k]<=sensor_c1[k]: #lidar data was tracked for longer
        #        distance_lidar=sqrt((position_x[k])**2+(position_y[k])**2)
        #        if distance_lidar<=closest_distance:
        #            closest_index=k
        #            closest_distance=distance_lidar  
        #    else:
        #        if distance[k]<=closest_distance:
        #            closest_index=k
        #            closest_distance=distance[k]
    print("closest_distance", closest_distance)
    critical_index=closest_index
    #RISK INFERENCE
    if operation>=1: #if robot operation is logistics
        if area[closest_index]>=1 and area[closest_index]<=3: #if the closest human is in front of the robot (i.e. same row)
            risk=1 #there is a risk
            #if (posture_prob[closest_index]<=posture_threshold and posture[closest_index]!=0 and area[closest_index]!=2): #additional condition only if the data was taken from camera 
            #    risk=0 #to avoid false positives
        else:
            risk=0 #there is not risk
    else: #for uv-c treatment
        risk=1 #there is always a risk
    
    return critical_index,risk


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
            #Critical human selection
            [hri.critical_index,hri.risk]=critical_human_selection()
            if hri.risk==1: #if there is risk of producing human injuries, human in the same row (logistics), human presence (uv-c treatment)
                ##hri_status############
                if len(human.sensor)==1 and (human.posture[0]+human.position_x[0]+human.position_y[0]+human.distance[0])==0: #None human detected
                    hri.status=0 #no human
                    hri.audio_message=0 #no message
                    hri.safety_action=0 #normal operation
                    hri.human_command=0  #no command
                else: #if at least one human was detected
                    distance=human.distance[hri.critical_index]    
                    print("DISTANCE",distance)
                    ###UV-C treatment#######################################
                    if robot.operation==0: #if robot is performing UV-C treatment
                        hri.human_command=0 #no human command expected during uv-c treatment
                        if distance>9: #if human is above 7m from the robot
                            hri.status=2 #safety HRI
                            hri.audio_message=1 #UVC danger message
                            hri.safety_action=2 # stop UV-C
                        else: #if human is within 9m
                            hri.status=3 #risky HRI
                            hri.audio_message=1 #UVC danger message
                            hri.safety_action=2 # stop UV-C
                   
                    ##LOGISTICS###############################################
                    if robot.operation>=3: 
                        if distance>7: #if human is above 7m
                            if robot.operation!=6: #if robot is not performing approaching maneuvers
                                hri.human_command=0 #no command
                                hri.status=1 # safety HRI
                                hri.audio_message=0 # no message
                                hri.safety_action=0 #normal operation
                            else: # if robot is performing approaching maneuvers, make it move slow and activate alert
                                hri.human_command=0 #no command
                                hri.status=1 # safety HRI
                                hri.safety_action=1 #make the robot reduce speed 
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                            #CASES WHEN HUMAN PERFORM A BODY GESTURE
                            #In case the picker wants the robot to approch to him/her before the robot ask for permission
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==1: #picker is ordering the robot to approach (using both arms)
                                hri.human_command=1 #make the robot approach to the picker from this point i.e robot.operation=1
                                hri.safety_action=1 #make the robot reduce speed
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                            #In case the picker wants the robot to stop before the robot approach more to him/her
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==8:  #picker is ordering the robot to stop (using right hand)
                                hri.human_command=3 #make the robot stop 
                                hri.safety_action=2 #waiting new command          
                                hri.audio_message=2 #message to ask the picker for a new order to approach/move away
                            #In case the picker wants the robot to move away before the robot approach more to him/her
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                                hri.human_command=2 #make the robot move away 
                                hri.safety_action=0 #normal operation             
                                hri.audio_message=5 #alert of presence
                        elif distance>3.6 and distance<=7: #if human is between 3.6-7m
                            #NORMAL CASES
                            hri.status=1
                            hri.human_command=0 #no command 
                            #To alert of its presence while moving
                            if robot.operation==3 or robot.operation==7:
                                hri.safety_action=0 #normal operation
                                hri.audio_message=5 #alert of presence
                            ##To make the robot stop and ask for free space
                            #if robot.operation==4 :
                            #    hri.safety_action=2 #waiting new command 
                            #    hri.audio_message=3 #message to ask the picker for free space to continue moving
                            
                            #To make the robot stop and ask for new order
                            if robot.operation==5:
                                hri.safety_action=2 #waiting new command 
                                hri.audio_message=2 #message to ask the picker for a new order to approach/move away
                            
                            #To make the robot approach with slow speed and activate alert
                            if robot.operation==6:
                                hri.safety_action=1 #make the robot reduce speed 
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                            
                            #CASES WHEN HUMAN PERFORM A BODY GESTURE
                            #In case the picker wants the robot to approch to him/her before the robot ask for permission
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==1: #picker is ordering the robot to approach (using both arms)
                                hri.human_command=1 #make the robot approach to the picker from this point i.e robot.operation=1
                                hri.safety_action=1 #make the robot reduce speed
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                            #In case the picker wants the robot to stop before the robot approach more to him/her
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==8:  #picker is ordering the robot to stop (using right hand)
                                hri.human_command=3 #make the robot stop 
                                hri.safety_action=2 #waiting new command          
                                hri.audio_message=2 #message to ask the picker for a new order to approach/move away
                            #In case the picker wants the robot to move away before the robot approach more to him/her
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                                hri.human_command=2 #make the robot move away 
                                hri.safety_action=0 #normal operation             
                                hri.audio_message=5 #alert of presence
                                                      
                        elif distance>1.2 and distance<=3.6: #if human is between 1.2-3.6m
                            #NORMAL CASES
                            hri.status=2
                            hri.human_command=0 #no command 
                            #To make the robot stop and ask for new order
                            if robot.operation==3 or robot.operation==5:
                                hri.safety_action=2 #waiting new command 
                                hri.audio_message=2 #message to ask the picker for a new order to approach/move away
                            
                            #To make the robot stop and ask for free space
                            if robot.operation==4 :
                                hri.safety_action=2 #waiting new command 
                                hri.audio_message=3 #message to ask the picker for free space to continue moving
                            
                            #To make the robot approach with slow speed and activate alert
                            if robot.operation==6:
                                #if human.motion[hri.critical_index]==1: # if the human is mostly static
                                    #print("HUMANO QUIETO")
                                hri.safety_action=1 #make the robot reduce speed 
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                            #To make the robot move away while alerting of its presence
                            if robot.operation==7:
                                hri.safety_action=0 #normal operation             
                                hri.audio_message=5 #alert of presence
                            #CASES WHEN HUMAN PERFORM A BODY GESTURE
                             #In case the picker wants the robot to approch to him/her 
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==1: #picker is ordering the robot to approach (using both arms)
                            #    if human.motion[hri.critical_index]==1: # if the human is mostly static
                                #print("HUMANO PIDIENDO APROX")
                                hri.human_command=1 #make the robot approach to the picker from this point i.e robot.operation=1
                                hri.safety_action=1 #make the robot reduce speed
                                hri.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                            #    else: #if human is not static
                            #        print("HUMANO MOVIENDOAE")
                            #        hri.human_command=3 #make the robot stop 
                            #        hri.safety_action=3 #waiting new command          
                            #        hri.audio_message=1 #message to ask the picker for a new order to approach/move away
                            #In case the picker wants the robot to stop 
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==8:  #picker is ordering the robot to stop (using right hand)
                                hri.human_command=3 #make the robot stop 
                                hri.safety_action=2 #waiting new command               
                                hri.audio_message=2 #message to ask the picker for a new order to approach/move away
                            #In case the picker wants the robot to move away 
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                                hri.human_command=2 #make the robot move away 
                                hri.safety_action=0 #normal operation             
                                hri.audio_message=5 #alert of presence
                            #SPECIAL CASE WHEN THE ROBOT WAS WAITING TOO LONG AN ORDER TO MOVE AWAY 
                            #if robot.operation==4 and  (human.sensor[hri.critical_index]==1 or (human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]!=2)):  #picker is not ordering the robot to move away (using both hands)
                            #    if human.motion[hri.critical_index]>=4: # if the human is moving away 
                            #        hri.human_command=2 #make the robot move away 
                            #        hri.safety_action=0 #normal operation             
                            #        hri.audio_message=6 #alert of presence
                            #In case the human is not static, or is not facing the robot
                            if (robot.operation!=7) and (human.motion[hri.critical_index]!=1 or human.orientation[hri.critical_index]==1): 
                                #print("HUMANO MOVIENDOsE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                                hri.human_command=3 #make the robot stop 
                                hri.safety_action=2 #waiting new command          
                                hri.audio_message=2 #message to ask the picker for a new order to approach/move away                       
                            
                        else: #if human is within 1.2m
                            #print("CRITICAL CASE")
                            hri.status=3 #dangerous hri
                            if robot.operation!=7:# only if robot is not moving away already                                
                                hri.human_command=0 #no command
                                hri.audio_message=0 #no message 
                                hri.safety_action=2 #safety stop
                            #In case the picker wants the robot to move away 
                            if human.sensor[hri.critical_index]!=1 and human.posture[hri.critical_index]==2:  #picker is ordering the robot to move away (using both hands)
                                #print("MOVE AWAY")
                                hri.human_command=2 #make the robot move away 
                                hri.safety_action=0 #normal operation   
                                hri.audio_message=5 #alert of presence
                            #print("Safety_action",hri.safety_action)
                            #print("Human_command",hri.human_command)
            else: #The critical human detected is not in the same row (Logistics)
                hri.status=0 #no human
                hri.audio_message=0 #no message
                hri.safety_action=0 #normal operation
                hri.human_command=0  #no command
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
            
        print("status",hri.status)
        print("safety_action",hri.safety_action)
        print("human_command",hri.human_command)
        print("robot_operation",robot.operation)
        print("risk",hri.risk)
        ############################################################################################
        #Publish SAFETY SYSTEM MESSAGES        
        safety_msg.hri_status = hri.status
        safety_msg.audio_message = hri.audio_message
        safety_msg.safety_action = hri.safety_action
        safety_msg.human_command = hri.human_command
        safety_msg.critical_index = hri.critical_index
        #safety_msg.goal_index = hri.goal_index
        pub_safety.publish(safety_msg)
       
        rate.sleep() #to keep fixed the publishing loop rate
