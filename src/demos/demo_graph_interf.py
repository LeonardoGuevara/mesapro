#! /usr/bin/python3

#required packages
import rospy
import message_filters #to sync the messages
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import numpy as np #to use matrix
import cv2
import yaml
import threading # Needed for Timer
from mesapro.msg import human_msg, hri_msg, robot_msg
import ros_numpy
##########################################################################################
#GENERAL PURPUSES VARIABLES
pub_hz=0.01 #main loop frequency
#Importing global parameters from .yaml file
config_direct=rospy.get_param("/hri_visualization/config_direct")
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
##############################################################################################################################
#IMPORTING LABELS NAMES
posture_labels=parsed_yaml_file.get("action_recog_config").get("posture_labels") # labels of the gestures used to train the gesture recognition model
motion_labels=parsed_yaml_file.get("action_recog_config").get("motion_labels") # labels of the possible motion actions
orientation_labels=parsed_yaml_file.get("action_recog_config").get("orientation_labels") # labels of the possible human orientations
hri_status_label=parsed_yaml_file.get("human_safety_config").get("hri_status") # labels of the possible HRI status
audio_message_label=parsed_yaml_file.get("human_safety_config").get("audio_message") # labels of the possible safety audio messages
safety_action_label=parsed_yaml_file.get("human_safety_config").get("safety_action") # labels of the possible safety actions
human_command_label=parsed_yaml_file.get("human_safety_config").get("human_command") # labels of the possible human commands based on gesture recognition
action_label=parsed_yaml_file.get("robot_config").get("action") # labels of the possible robot actions
#########################################################################################################################
#PLOT HRI PARAMETERS
black_image_size=[680,650] #size of the black background where the parameters are shown
visual_mode = rospy.get_param("/hri_visualization/visual_mode",1) #"1" for testing only camera perception, "2" for gazebo simulation, "3" for real implementation 
################################################################################################################################
#ROS PUBLISHER SET UP
pub_img = rospy.Publisher('han_param_visual', Image,queue_size=1)
msg_img = Image()  
###################################################################################################################################

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
        self.image=np.zeros((black_image_size[0],black_image_size[1],3), np.uint8) #initial value
        self.image_size=[black_image_size[0],black_image_size[1]] #size of a single image, initial values
        self.n_human=0
        self.thermal_detection="inactive"
        
    def human_callback(self,human_info):
        self.n_human=human_info.n_human
        if human_info.thermal_detection==True:    
            self.thermal_detection="active"
        else:
            self.thermal_detection="inactive"
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
            
    def image_callback(self,rgb):
        ##################################################################################33
        #Front cameras info extraction
        #Color image
        color_image = ros_numpy.numpify(rgb)
        #color_image = color_image1[...,[2,1,0]].copy() #from bgr to rgb
        scaling=2 #to recover the original image size
        color_image=cv2.resize(color_image,(int(color_image.shape[1]*scaling),int(color_image.shape[0]*scaling))) #resizing it to fit the screen
        self.image=color_image
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=5 #no safety action  
        self.human_command=0
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        self.time_without_msg=parsed_yaml_file.get("human_safety_config").get("time_without_msg") # Maximum time without receiving safety messages
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
                 
    def safety_callback(self,safety_info):
        self.status=safety_info.hri_status
        self.audio_message=safety_info.audio_message
        self.safety_action=safety_info.safety_action
        self.human_command=safety_info.human_command
        self.critical_index=safety_info.critical_index   
        print("Safety message received")
        self.timer_safety.cancel()
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
    
    def safety_timeout(self):
        print("No safety message received in a long time")
        self.audio_message=7 #to alert that safety system is not publishing 
        
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
    if visual_mode!=2: #only when images from camera are goint to be visualized
        #black_image=np.zeros((black_image_size[0],black_image_size[1],3), np.uint8)
        black_image=np.ones((black_image_size[0],black_image_size[1],3), np.uint8)*255
        proportion=black_image.shape[0]/color_image.shape[0] #proportion used to resize the image to match the high of the black_image
        color_image=cv2.resize(color_image,(int(color_image.shape[1]*proportion),black_image.shape[0])) #to resize it proportionally
        color_image=np.append(black_image,color_image,axis=1) 
        extra=black_image.shape[1]
    if human.sensor==0:
        sensor="camera+lidar"
    if human.sensor==1:
        sensor="lidar"
    if human.sensor==2:
        sensor="camera"
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    if visual_mode>=2:
        #Print SAFETY SYMTEM INFO    
        color_image = cv2.putText(color_image,"**** HUMAN-AWARE NAVIGATION INFO ****",(50, 270) , font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        if hri.audio_message!=7:
            color_image = cv2.putText(color_image,"HRI risk level:       "+hri_status_label[int(hri.status)],(50, 300) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"audio message:     "+audio_message_label[int(hri.audio_message)],(50, 330) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"human command:   "+human_command_label[int(hri.human_command)],(50, 360) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"safety action:        "+safety_action_label[int(hri.safety_action)],(50, 360) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
        #Print ROBOT ACTION INFO
        #color_image = cv2.putText(color_image,"***ROBOT INFO***",(50, 390) , font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"robot action:  "+action_label[int(robot.action)],(50, 390) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"robot pos. x:   "+str(round(robot.pos_x,2))+"m",(50, 420), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"robot pos. y:   "+str(round(robot.pos_y,2))+"m",(50, 450) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"robot speed:    "+str(round(robot.speed,2))+"m/s",(50, 480) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"current node:    "+robot.current_node,(50, 510) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"goal node:       "+robot.goal_node,(50, 540) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    #Print HUMAN PERCEPTION INFO
    if human.n_human==0: #None human detected
        if human.thermal_detection=="active":
            color_image = cv2.putText(color_image,"********* HUMAN SENSING INFO *********",(50, 30) , font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"thermal:     "+human.thermal_detection,(50, 60) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            color_image = cv2.putText(color_image,"********* HUMAN SENSING INFO *********",(50, 30) , font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"no human detection:      ",(50, 60) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        if sensor=="lidar":# and human.centroids_x[hri.critical_index]+human.centroids_y[hri.critical_index]!=0:
            color_image = cv2.putText(color_image,"********* HUMAN SENSING INFO *********",(50, 30) , font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"sensor:      "+sensor,(50, 60) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"motion:      "+motion_labels[int(human.motion)],(50, 60) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 90) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"position x:  "+str(round(human.position_x,2))+"m",(50, 120), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"position y:  "+str(round(human.position_y,2))+"m",(50, 150) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"area:        "+str(human.area),(50, 210) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
        else:
            color_image = cv2.putText(color_image,"********* HUMAN SENSING INFO *********",(50, 30) , font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"sensor:      "+sensor,(50, 60) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"thermal:     "+str(human.thermal_detection),(50, 90) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"motion:      "+motion_labels[int(human.motion)],(50, 60) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"orientation:  "+orientation_labels[int(human.orientation)],(50, 90) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"gesture:     "+posture_labels[int(human.posture)],(50, 120) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 150) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"position x:  "+str(round(human.position_x,2))+"m",(50, 180), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"position y:  "+str(round(human.position_y,2))+"m",(50, 210) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"area:        "+str(human.area),(50, 300) , font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            if visual_mode!=2 : #visual_mode==1 or visual_mode==3
                centroids_x=human.centroids_x
                centroids_y=human.centroids_y
                for k in range(0,len(centroids_x)):    
                    if centroids_x[k]+centroids_y[k]!=0:
                        center_coordinates = (int(centroids_x[k]*proportion)+extra, int(centroids_y[k]*proportion)) 
                        #center_coordinates=(0,480)
                        if k==hri.critical_index:
                            color_image = cv2.circle(color_image, center_coordinates, 5, (0, 0, 255), 20) #RED
                        else:
                            color_image = cv2.circle(color_image, center_coordinates, 5, (255, 0, 0), 20) #BLUE
    if visual_mode!=2: #To resize the window with images and text
        scaling=0.75
        color_image=cv2.resize(color_image,(int(color_image.shape[1]*scaling),int(color_image.shape[0]*scaling))) #resizing it to fit the screen
        
    #cv2.imshow("System outputs",color_image)
    #cv2.waitKey(5)
    # From bgr to rgb
    color_image = color_image[...,[2,1,0]].copy()
    #Publishing IMAGE
    msg_img.header.stamp = rospy.Time.now()
    msg_img.height = color_image.shape[0]
    msg_img.width = color_image.shape[1]
    msg_img.encoding = "rgb8"
    msg_img.is_bigendian = False
    msg_img.step = 3 * color_image.shape[1]
    msg_img.data = np.array(color_image).tobytes()
    pub_img.publish(msg_img)
    
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
    rospy.Subscriber('openpose_output', Image,human.image_callback, queue_size=5) #new topic name before thermal info             
    if visual_mode>=2:
        rospy.Subscriber('robot_info',robot_msg,robot.robot_callback_info)
        rospy.Subscriber('/robot_pose', Pose, robot.robot_callback_pos) 
        rospy.Subscriber('/nav_vel',geometry_msgs.msg.Twist,robot.robot_callback_vel)    
    #Rate setup
    rate = rospy.Rate(1/pub_hz)  # main loop frecuency in Hz
    while not rospy.is_shutdown():	
        if visual_mode==2:  
            color_image = np.ones((black_image_size[0],black_image_size[1],3), np.uint8)*255 
            #color_image = np.zeros((black_image_size[0],black_image_size[1],3), np.uint8) 
        else: #visual_mode=1 or 3
            color_image=human.image
        visual_outputs(color_image)
        print("MODE",visual_mode)
        rate.sleep() #to keep fixed the publishing loop rate
        
        
