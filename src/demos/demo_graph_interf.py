#! /usr/bin/python

#required packages
import rospy #tf
import message_filters #to sync the messages
import geometry_msgs.msg
from geometry_msgs.msg import Pose
#from tf.transformations import euler_from_quaternion
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
action_label=r_param[0][1]
main_counter=0
pub_hz=0.01
demo=1 #demo 1: perception, demo 2: topological navigation
no_detection=True  

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
        self.motion=0 #from lidar + camara
        self.area=0 
        self.image=np.zeros((800,400,3), np.uint8) #initial value
    
    def human_callback(self,human_info):
        global no_detection
        #print("NEW HUMAN DATA")
        self.posture=human_info.posture[hri.critical_index]
        self.motion=human_info.motion[hri.critical_index]   
        self.position_x=human_info.position_x[hri.critical_index]
        self.position_y=human_info.position_y[hri.critical_index]
        self.centroid_x=human_info.centroid_x[hri.critical_index]
        self.centroid_y=human_info.centroid_y[hri.critical_index]
        self.distance=human_info.distance[hri.critical_index]
        self.sensor=human_info.sensor[hri.critical_index]
        self.orientation=human_info.orientation[hri.critical_index]
        self.area=human_info.area[hri.critical_index]
        print(len(human_info.sensor))
        print(human_info.posture[0])
        print(human_info.position_x[0])
        print(human_info.position_y[0])
        print(human_info.distance[0])
        if (human.centroid_x+human.centroid_y+human_info.motion[0]+human_info.posture[0]+human_info.position_x[0]+human_info.position_y[0]+human_info.distance[0])==0: #None human detected
            no_detection=True  
        else:
            no_detection=False
        print(no_detection)
    def camera_callback(self,ros_image):
        try:
            self.image = bridge.imgmsg_to_cv2(ros_image, "bgr8")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        color_image=self.image
        demo_outputs(color_image)

class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=5 #no safety action  
        self.human_command=0
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
                 
    def safety_callback(self,safety_info):
        self.status=safety_info.hri_status
        self.audio_message=safety_info.audio_message
        self.safety_action=safety_info.safety_action
        self.human_command=safety_info.human_command
        self.critical_index=safety_info.critical_index   
        

class robot_class:
    def __init__(self): #It is done only the first iteration
        self.pos_x=0 #[x,y,theta]
        self.pos_y=0
        self.action=4 #static 
        self.speed=0
        self.current_node="Unknown"
        self.goal_node="Unknown"
        
    def robot_callback_info(self,robot_info):       
        robot.action=robot_info.action
        robot.current_node=robot_info.current_node
        robot.goal_node=robot_info.goal_node


    def robot_callback_pos(self,pose):
        self.pos_x=pose.position.x
        self.pos_y=pose.position.y
               
    def robot_callback_vel(self,msg):
        self.speed=msg.linear.x

def closest_human(data):
    dist=data.distance
    closest_dist=1000 #initial value
    closest_index=0
    n_human=len(dist)
    #CLOSEST HUMAN TRACKED
    for k in range(0,n_human):
        if dist[k]<=closest_dist:
            closest_index=k
            closest_dist=dist[k]
    return closest_index

def demo_outputs(color_image):
    
    if human.sensor==0:
        sensor="camera+lidar"
    if human.sensor==1:
        sensor="lidar"
    if human.sensor==2:
        sensor="camera"
    
    if human.area>=1 and human.area<=3:
        area="frontal_detection"
    elif human.area>=6 and human.area<=8:
        area="back_detection"
    else:
        area="side_detection"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #Print HUMAN PERCEPTION INFO
    if no_detection:# or hri.status==0: #None human detected
        color_image = cv2.putText(color_image,"***NO HUMAN DETECTION***",(50, 30) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    #elif area=="side_detection":
    #    color_image = cv2.putText(color_image,"***HUMAN DETECTED - NO RISK ***",(50, 30) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        if sensor=="lidar":
            color_image = cv2.putText(color_image,"***HUMAN PERCEPTION***",(50, 30) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"sensor:      "+sensor,(50, 60) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"motion:      "+motion_labels[int(human.motion)],(50, 90) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 120) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"x:           "+str(round(human.position_x,2))+"m",(50, 150), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"y:           "+str(round(human.position_y,2))+"m",(50, 180) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"area:        "+str(human.area),(50, 210) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        else:
            color_image = cv2.putText(color_image,"***HUMAN PERCEPTION***",(50, 30) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"sensor:      "+sensor,(50, 60) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"orientation:  "+orientation_labels[int(human.orientation)],(50, 90) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"motion:      "+motion_labels[int(human.motion)],(50, 120) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"posture:     "+posture_labels[int(human.posture)],(50, 150) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
    
            if sensor=="camera":
                color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 180) , font, 0.7,(0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"area:           "+str(human.area),(50, 210) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if sensor=="camera+lidar":
                color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 180) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"x:           "+str(round(human.position_x,2))+"m",(50, 210), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"y:           "+str(round(human.position_y,2))+"m",(50, 240) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"area:        "+str(human.area),(50, 270) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if demo==2:
                #Print SAFETY SYMTEM INFO    
                color_image = cv2.putText(color_image,"***SAFETY SYSTEM***",(50, 300) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"hri status:           "+hri_status_label[int(hri.status)],(50, 330) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"audio message:     "+audio_message_label[int(hri.audio_message)],(50, 360) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"human command:   "+human_command_label[int(hri.human_command)],(50, 390) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"safety action:        "+safety_action_label[int(hri.safety_action)],(50, 420) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                               
                #Print ROBOT ACTION INFO
                color_image = cv2.putText(color_image,"***ROBOT ACTION***",(50, 450) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"action:   "+action_label[int(robot.action)],(50, 480) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"x:           "+str(round(robot.pos_x,2))+"m",(50, 510), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"y:           "+str(round(robot.pos_y,2))+"m",(50, 540) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"speed:       "+str(round(robot.speed,2))+"m/s",(50, 570) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"current node:   "+robot.current_node,(50, 600) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"goal node:      "+robot.goal_node,(50, 630) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                center_coordinates = (int(human.centroid_x), int(human.centroid_y)) 
                color_image = cv2.circle(human.image, center_coordinates, 5, (255, 0, 0), 20) 
    
    cv2.imshow("System outputs",color_image)
    cv2.waitKey(5)
    
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node       
    human=human_class()  
    hri=hri_class()
    robot=robot_class()
    rospy.init_node('demo_visualization',anonymous=True)
    # Setup and call subscription
    rospy.Subscriber('human_info',human_msg,human.human_callback)
    if demo==1:
        rospy.Subscriber('camera/camera1/color/image_raw', Image,human.camera_callback)  
    if demo==2:
        rospy.Subscriber('human_safety_info',hri_msg,hri.safety_callback)
        rospy.Subscriber('robot_info',robot_msg,robot.robot_callback_info)
        rospy.Subscriber('/robot_pose', Pose, robot.robot_callback_pos) 
        rospy.Subscriber('/nav_vel',geometry_msgs.msg.Twist,robot.robot_callback_vel)    
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        main_counter=main_counter+1
        if demo==2:  
            color_image = np.zeros((650,650,3), np.uint8) 
            demo_outputs(color_image)
        print(main_counter)    
        print("Distance",round(human.distance,2))
        rate.sleep() #to keep fixed the publishing loop rate
        
        
