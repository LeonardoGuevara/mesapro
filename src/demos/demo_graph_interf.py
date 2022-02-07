#! /usr/bin/python3

#required packages
import rospy #tf
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import numpy as np #to use matrix
from cv_bridge import CvBridge, CvBridgeError
import cv2
import yaml
from mesapro.msg import human_msg, hri_msg, robot_msg
##########################################################################################

#Importing global parameters from .yaml file
config_direct="/home/leo/rasberry_ws/src/mesapro/config/"
#config_direct=rospy.get_param("/hri_visualization/config_direct") #you have to change /hri_visualization/ if the node is not named like this
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
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
visual_mode = rospy.get_param("/hri_visualization/visual_mode") #you have to change /hri_visualization/ if the node is not named like this
#visual_mode=1 means only perception, visual_mode=2 means simulated perception + simulated topo nav, visual_mode=3 means real robot
no_detection=True  
image_width=840
#Areas
max_dist=8 #in meters, the probabilities are fixed at prob_init from this distance 
delta_prob_1_4=0.17 #percentage of variation of areas from initial probability at max_dist to final probability at 0 m.
delta_prob_2_3=0.12
offset=0.02 #in case camera is not perfectly aligned, can be positive or negative
prob_2_init=0.45 #in pixels percent of area 2
prob_3_init=(0.5-prob_2_init)+0.5+offset
prob_1_init=prob_2_init-delta_prob_2_3
prob_4_init=prob_3_init+delta_prob_2_3
prob_0_init=0 #initial area pixel percentage
prob_5_init=1 #last area pixel percentage
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        self.position_x=0
        self.position_y=0
        self.posture=0 #from camera [posture_label,posture_probability]
        self.centroids_x=[0]
        self.centroids_y=[0]
        self.orientation=0 # it can be "front" or "back" if the human is facing the robot or not , from camera
        self.distance=0  # distance between the robot and the average of the skeleton joints distances taken from the depth image, from camera
        self.sensor=0 # it can be 0 if the data is from camera and Lidar, 1 if the data is  only from the LiDAR or 2 if the data is only from de camera
        self.motion=0 #from lidar + camara
        self.area=0 
        self.image=np.zeros((650,650,3), np.uint8) #initial value
        self.n_human=0
    
    def human_callback(self,human_info):
        self.n_human=human_info.n_human
        if hri.critical_index<=self.n_human-1 and self.n_human!=0:
            self.posture=human_info.posture[hri.critical_index]
            self.motion=human_info.motion[hri.critical_index]   
            self.position_x=human_info.position_x[hri.critical_index]
            self.position_y=human_info.position_y[hri.critical_index]
            self.centroids_x=human_info.centroid_x
            self.centroids_y=human_info.centroid_y
            self.distance=human_info.distance[hri.critical_index]
            self.sensor=human_info.sensor[hri.critical_index]
            self.orientation=human_info.orientation[hri.critical_index]
            self.area=human_info.area[hri.critical_index]
            
    def camera_callback(self,ros_image):
        global image_width
        try:
            self.image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
            size = np.array(self.image)
            image_width = size.shape[1]
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        color_image=self.image
        visual_outputs(color_image)

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
    
    def area_inference_camera(self):
        #Update area according to the distance
        dist=human.distance
        if dist>=max_dist:
            areas_percent=[prob_0_init,prob_1_init,prob_2_init,prob_3_init,prob_4_init,prob_5_init]
        else:
            prob_2=(delta_prob_2_3/max_dist)*dist+prob_2_init-delta_prob_2_3
            prob_3=(0.5-prob_2)+0.5+offset
            prob_1=(delta_prob_1_4/max_dist)*dist+prob_1_init-delta_prob_1_4
            prob_4=(0.5-prob_1)+0.5+offset
            areas_percent=[prob_0_init,prob_1,prob_2,prob_3,prob_4,prob_5_init]
        return areas_percent

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


def visual_outputs(color_image):
    if visual_mode!=2:
        black_image=np.zeros((color_image.shape[0],650,3), np.uint8)
        color_image=np.append(black_image,color_image,axis=1) 
        extra=black_image.shape[1]
    if human.sensor==0:
        sensor="camera+lidar"
    if human.sensor==1:
        sensor="lidar"
    if human.sensor==2:
        sensor="camera"
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_lines=[prob_0_init,prob_1_init,prob_2_init,prob_3_init,prob_4_init,prob_5_init] #by default
    if visual_mode>=2:
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
    #Print HUMAN PERCEPTION INFO
    if human.n_human==0: #None human detected
        color_image = cv2.putText(color_image,"***NO HUMAN DETECTION***",(50, 30) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        if sensor=="lidar":# and human.centroids_x[hri.critical_index]+human.centroids_y[hri.critical_index]!=0:
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
            color_image = cv2.putText(color_image,"motion:      "+motion_labels[int(human.motion)],(50, 90) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"orientation:  "+orientation_labels[int(human.orientation)],(50, 120) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"posture:     "+posture_labels[int(human.posture)],(50, 150) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
    
            if sensor=="camera":
                color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 180) , font, 0.7,(0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"area:           "+str(human.area),(50, 210) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if sensor=="camera+lidar":
                color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 180) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"x:           "+str(round(human.position_x,2))+"m",(50, 210), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"y:           "+str(round(human.position_y,2))+"m",(50, 240) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                color_image = cv2.putText(color_image,"area:        "+str(human.area),(50, 270) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if visual_mode!=2 : #visual_mode==1 or visual_mode==3
                centroids_x=human.centroids_x
                centroids_y=human.centroids_y
                for k in range(0,len(centroids_x)):    
                    if centroids_x[k]+centroids_y[k]!=0:
                        center_coordinates = (int(centroids_x[k])+extra, int(centroids_y[k])) 
                        #center_coordinates=(0,480)
                        if k==hri.critical_index:
                            color_image = cv2.circle(color_image, center_coordinates, 5, (0, 0, 255), 20) #RED
                        else:
                            color_image = cv2.circle(color_image, center_coordinates, 5, (255, 0, 0), 20) #BLUE
                x_lines=hri.area_inference_camera()
                #x_lines=[0,0.3,0.4,0.6,0.7,1]
    if visual_mode!=2:
        color_image=cv2.line(color_image, (int(x_lines[1]*image_width)+extra, 0), (int(x_lines[1]*image_width)+extra, 600), (0, 255, 0), thickness=1)
        color_image=cv2.line(color_image, (int(x_lines[2]*image_width)+extra, 0), (int(x_lines[2]*image_width)+extra, 600), (0, 255, 0), thickness=1)
        color_image=cv2.line(color_image, (int(x_lines[3]*image_width)+extra, 0), (int(x_lines[3]*image_width)+extra, 600), (0, 255, 0), thickness=1)
        color_image=cv2.line(color_image, (int(x_lines[4]*image_width)+extra, 0), (int(x_lines[4]*image_width)+extra, 600), (0, 255, 0), thickness=1)
 
    cv2.imshow("System outputs",color_image)
    cv2.waitKey(5)
    
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node       
    human=human_class()  
    hri=hri_class()
    robot=robot_class()
    rospy.init_node('hri_visualization',anonymous=True)
    # Setup and call subscription
    rospy.Subscriber('human_info',human_msg,human.human_callback)
    rospy.Subscriber('human_safety_info',hri_msg,hri.safety_callback)
    if visual_mode==1 or visual_mode==3:
        rospy.Subscriber('camera/camera1/color/image_raw', Image,human.camera_callback)  
    if visual_mode>=2:
        rospy.Subscriber('robot_info',robot_msg,robot.robot_callback_info)
        rospy.Subscriber('/robot_pose', Pose, robot.robot_callback_pos) 
        rospy.Subscriber('/nav_vel',geometry_msgs.msg.Twist,robot.robot_callback_vel)    
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        main_counter=main_counter+1
        if visual_mode==2:  
            color_image = np.zeros((650,650,3), np.uint8) 
            visual_outputs(color_image)
        print(main_counter)  
        print("MODE",visual_mode)
        print("Distance",round(human.distance,2))
        #print("CRITICAL_INDEX",hri.critical_index)
        print("NO DETECTION",no_detection)
        rate.sleep() #to keep fixed the publishing loop rate
        
        
