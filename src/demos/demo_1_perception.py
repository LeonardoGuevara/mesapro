#! /usr/bin/python

#required packages
import rospy #tf
import message_filters #to sync the messages
#from sklearn.ensemble import RandomForestClassifier
from sensor_msgs.msg import Image
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from cv_bridge import CvBridge, CvBridgeError
import os
import cv2
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
r_param=list(dict.items(parsed_yaml_file["robot_config"]))

#Initializating cv_bridge
bridge = CvBridge()
posture_labels=ar_param[2][1]
motion_labels=ar_param[3][1]
orientation_labels=ar_param[4][1]
hri_status_label=hs_param[0][1]
audio_message_label=hs_param[1][1] 
safety_action_label=hs_param[2][1]
human_command_label=hs_param[3][1]
operation_label=r_param[0][1]
new_data=[0,0,0] #data from [human_info, safety_info,robot_info]
main_counter=0
pub_hz=0.01
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        self.position_x=0
        self.position_y=0
        self.posture=0 #from camera [posture_label,posture_probability]
        self.centroid_x=0 #x,y (pixels) of the human centroid, from camera
        self.centroid_y=0 #x,y (pixels) of the human centroid, from camera
        self.orientation=0 # it can be "front" or "back" if the human is facing the robot or not , from camera
        self.distance=0  # distance between the robot and the average of the skeleton joints distances taken from the depth image, from camera
        self.sensor=0 # it can be 0 if the data is from camera and Lidar, 1 if the data is  only from the LiDAR or 2 if the data is only from de camera
        self.critical_index=0 #index of the closest human to the robot
        self.motion=0 #from lidar + camara
        self.image=np.zeros((800,400,3), np.uint8) #initial value

class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=0  
        self.human_command=0

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.position=np.zeros([3,1]) #[x,y,theta]
        self.operation=2 #["UV-C_treatment","moving_to_picker_location", "wait_for_command_to_approach", "approaching_to_picker","wait_for_command_to_move_away", "moving_away_from_picker"] 
        self.operation_new=self.operation

        
def camera_callback(ros_image):
    #print("DATA FROM CAMERA")
    try:
        human.image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        #image_width = depth_array.shape[1]
        #print("WIDHT",image_width)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    #new_data[0]=1
    if new_data[0]==1 or new_data[1]==1 or new_data[2]==1: #only if there is a new human_info data or robot_data or hri_data
        if human.sensor==0:
            sensor="camera+lidar"
        if human.sensor==1:
            sensor="lidar"
        if human.sensor==2:
            sensor="camera"
            
        print(human.critical_index)
        center_coordinates = (int(human.centroid_x), int(human.centroid_y)) 
        color_image = cv2.circle(human.image, center_coordinates, 5, (255, 0, 0), 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #Print HUMAN PERCEPTION INFO
        color_image = cv2.putText(color_image,"HUMAN_PERCEPTION",(500, 50) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,posture_labels[int(human.posture)],(500, 80) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,orientation_labels[int(human.orientation)],(500, 110) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,motion_labels[int(human.motion)],(500, 140) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"sensor: "+sensor,(500, 170) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if sensor=="camera":
            color_image = cv2.putText(color_image,"distance: "+str(round(human.distance,2)),(500, 200) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if sensor=="camera+lidar":
            color_image = cv2.putText(color_image,"distance: "+str(round(human.distance,2)),(500, 230) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"x: "+str(round(human.position_x,2)),(500, 260), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"y: "+str(round(human.position_y,2)),(500, 290) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #Print SAFETY SYMTEM INFO    
        color_image = cv2.putText(color_image,"SAFETY_SYSTEM",(50, 250) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"status: "+hri_status_label[int(hri.status)],(50, 280) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"audio: "+audio_message_label[int(hri.audio_message)],(50, 310) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"human command: "+human_command_label[int(hri.human_command)],(50, 340) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        #Print ROBOT OPERATION INFO
        color_image = cv2.putText(color_image,"ROBOT_OPERATION",(50, 370) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"operation: "+operation_label[int(robot.operation)],(50, 400) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"safety action: "+safety_action_label[int(hri.safety_action)],(50, 430) , font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        cv2.imshow("Human tracking",color_image)
        cv2.waitKey(5)
        if new_data[0]==1:
            new_data[0]=0
        if new_data[1]==1:
            new_data[1]=0
        if new_data[2]==1:
            new_data[2]=0

def human_callback(human_info):
    if len(human_info.sensor)!=0 and new_data[0]==0:# and new_data[0]==1: #only if there is a new human_info data
        human.critical_index=human_info.critical_index
        human.posture=human_info.posture[human.critical_index]
        human.motion=human_info.motion[human.critical_index]   
        human.position_x=human_info.position_x[human.critical_index]
        human.position_y=human_info.position_y[human.critical_index]
        human.centroid_x=human_info.centroid_x[human.critical_index]
        human.centroid_y=human_info.centroid_y[human.critical_index]
        human.distance=human_info.distance[human.critical_index]
        human.sensor=human_info.sensor[human.critical_index]
        new_data[0]=1#        
            
def safety_callback(safety_info):
    #print("NEW SAFETY DATA")
    if new_data[1]==0:
        hri.status=safety_info.hri_status
        hri.audio_message=safety_info.audio_message
        hri.safety_action=safety_info.safety_action
        hri.human_command=safety_info.human_command
        new_data[1]=1
    
def robot_callback(robot_info):
    #print("NEW SAFETY DATA")
    if new_data[1]==0:
        robot.position=np.array([robot_info.position_x,robot_info.position_y,robot_info.orientation])
        robot.operation=robot_info.operation
        new_data[2]=1
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node       
    human=human_class()  
    hri=hri_class()
    robot=robot_class()
    rospy.init_node('human_perception_system',anonymous=True)
    # Setup and call subscription
    #image_sub = message_filters.Subscriber('camera/camera1/color/image_raw', Image)
    #depth_sub = message_filters.Subscriber('camera/camera1/aligned_depth_to_color/image_raw', Image)
    #ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.01)
    #ts.registerCallback(camera_callback)
    rospy.Subscriber('camera/camera1/color/image_raw', Image,camera_callback)  
    rospy.Subscriber('human_info',human_msg,human_callback)
    rospy.Subscriber('human_safety_info',hri_msg,safety_callback)
    rospy.Subscriber('robot_info',robot_msg,robot_callback)
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        main_counter=main_counter+1
        #if new_data!=[0,0]:
            
        print(main_counter)    
        print("Distance",round(human.distance,2))
        rate.sleep() #to keep fixed the publishing loop rate
        
        