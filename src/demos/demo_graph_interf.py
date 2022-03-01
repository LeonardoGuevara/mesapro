#! /usr/bin/python3

#required packages
import rospy
import message_filters #to sync the messages
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import numpy as np #to use matrix
from cv_bridge import CvBridge
import cv2
import yaml
import threading # Needed for Timer
from mesapro.msg import human_msg, hri_msg, robot_msg
import ros_numpy
##########################################################################################

#Importing global parameters from .yaml file
#config_direct="/home/leo/rasberry_ws/src/mesapro/config/"
config_direct=rospy.get_param("/hri_visualization/config_direct") #you have to change /hri_visualization/ if the node is not named like this
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
ar_param=list(dict.items(parsed_yaml_file["action_recog_config"]))
hs_param=list(dict.items(parsed_yaml_file["human_safety_config"]))
r_param=list(dict.items(parsed_yaml_file["robot_config"]))
#CAMERAS INFO
##############################################################################################################################
#VALUES TAKEN FROM human_detector_rgbd_thermal.py
#################################################################################################################################
thermal_info=rospy.get_param("/hri_visualization/thermal_info") #you have to change /hri_visualization/ if the node is not named like this
#thermal_info=False
image_rotation=rospy.get_param("/hri_visualization/image_rotation") #you have to change /hri_visualization/ if the node is not named like this
#image_rotation=0 #it can be 0,90, 180, 270 measured clockwise     
if image_rotation==270:
    resize_param=[120,130,285,380] #[y_init_up,x_init_left,n_pixels_x,n_pixels_y] assuming portrait mode with image_rotation=270, keeping original aspect ratio 3:4,i.e 285/380 = 120/160 = 3/4
elif image_rotation==90: #IT IS NOT WELL TUNNED YET
    resize_param=[120,105,285,380] 
else: #image_rotation==0 #IT IS NOT WELL TUNNED YET
    resize_param=[120,130,380,285] 
#################################################################################################################################
#IMPORTING LABELS NAMES
posture_labels=ar_param[2][1]
motion_labels=ar_param[3][1]
orientation_labels=ar_param[4][1]
hri_status_label=hs_param[0][1]
audio_message_label=hs_param[1][1] 
safety_action_label=hs_param[2][1]
human_command_label=hs_param[3][1]
action_label=r_param[0][1]
#########################################################################################################################
#PLOT HRI PARAMETERS
black_image_size=[680,650] #size of the black background where the parameters are shown
visual_mode = rospy.get_param("/hri_visualization/visual_mode") #you have to change /hri_visualization/ if the node is not named like this
#visual_mode=1 # visual_mode=1 means only perception, visual_mode=2 means simulated perception + simulated topo nav, visual_mode=3 means real robot
n_cameras=rospy.get_param("/hri_visualization/n_cameras") #you have to change /hri_visualization/ if the node is not named like this
#n_cameras=1 # 1 means that the back camera is emulated by reproducing the front camera image

#GENERAL PURPUSES VARIABLES
main_counter=0
pub_hz=0.01
#Initializating cv_bridge
bridge = CvBridge()


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
            
    def rgb_thermal_1_callback(self,rgb_front, therm_front):
        ##################################################################################33
        #Front cameras info extraction
        #therm_image_front = bridge.imgmsg_to_cv2(therm_front, "mono8") #Gray scale image
        therm_image_front = ros_numpy.numpify(therm_front)
        if image_rotation==90:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_front=therm_image_front
        img_t_rot_front=cv2.resize(img_t_rot_front,(resize_param[2],resize_param[3])) #to match the rgbd aspect ratio
        
        #color_image_front = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front = ros_numpy.numpify(rgb_front)
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        img_rgb_rz_front=np.zeros((img_t_rot_front.shape[0],img_t_rot_front.shape[1],3),np.uint8) #to match the thermal field of view
        img_rgb_rz_front=img_rgb_rot_front[resize_param[0]:resize_param[0]+img_t_rot_front.shape[0],resize_param[1]:resize_param[1]+img_t_rot_front.shape[1],:]   
        
        self.image_size = img_rgb_rz_front.shape
        ##################################################################################
        #Back cameras emulation
        therm_image_back=therm_image_front
        if image_rotation==90:
            img_t_rot_back=cv2.rotate(therm_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_back=cv2.rotate(therm_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_back=therm_image_back
        img_t_rot_back=cv2.resize(img_t_rot_back,(resize_param[2],resize_param[3]))        
        
        color_image_back=color_image_front
        if image_rotation==90:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_back=color_image_back            
        img_rgb_rz_back=np.zeros((img_t_rot_back.shape[0],img_t_rot_back.shape[1],3),np.uint8)
        img_rgb_rz_back=img_rgb_rot_back[resize_param[0]:resize_param[0]+img_t_rot_back.shape[0],resize_param[1]:resize_param[1]+img_t_rot_back.shape[1],:]
        
        ##############################################################################################
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rz_front,img_rgb_rz_back,axis=1) 
        therm_array=np.append(img_t_rot_front,img_t_rot_back,axis=1)
        intensity_image=cv2.cvtColor(therm_array,cv2.COLOR_GRAY2RGB)
        color_image = cv2.addWeighted(color_image,0.7,intensity_image,0.7,0)      
        self.image=color_image
        
    def rgb_1_callback(self,rgb_front):
        ##################################################################################33
        #Front camera info extraction
        #color_image_front = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front = ros_numpy.numpify(rgb_front)
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        self.image_size = img_rgb_rot_front.shape
        ##################################################################################
        #Back cameras emulation
        color_image_back=color_image_front
        if image_rotation==90:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_back=color_image_back            

        ##############################################################################################        
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rot_front,img_rgb_rot_back,axis=1) 
        #######################################################################################
        self.image=color_image
        
    def rgb_thermal_2_callback(self,rgb_front, therm_front,rgb_back, therm_back):
        ##################################################################################33
        #Front cameras info extraction
        #therm_image_front = bridge.imgmsg_to_cv2(therm_front, "mono8") #Gray scale image
        therm_image_front = ros_numpy.numpify(therm_front)
        if image_rotation==90:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_front=therm_image_front
        img_t_rot_front=cv2.resize(img_t_rot_front,(resize_param[2],resize_param[3])) #to match the rgbd aspect ratio
        
        #color_image_front = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front = ros_numpy.numpify(rgb_front)
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        img_rgb_rz_front=np.zeros((img_t_rot_front.shape[0],img_t_rot_front.shape[1],3),np.uint8) #to match the thermal field of view
        img_rgb_rz_front=img_rgb_rot_front[resize_param[0]:resize_param[0]+img_t_rot_front.shape[0],resize_param[1]:resize_param[1]+img_t_rot_front.shape[1],:]   
        
        self.image_size = img_rgb_rz_front.shape
        ##################################################################################
        #Back cameras info extraction
        #therm_image_back = bridge.imgmsg_to_cv2(therm_back, "bgr8")
        therm_image_back = ros_numpy.numpify(therm_back)
        if image_rotation==90:
            img_t_rot_back=cv2.rotate(therm_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_back=cv2.rotate(therm_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_back=therm_image_back
        img_t_rot_back=cv2.resize(img_t_rot_back,(resize_param[2],resize_param[3]))        
        
        #color_image_back = bridge.imgmsg_to_cv2(rgb_back, "bgr8")
        color_image_back = ros_numpy.numpify(rgb_back)
        if image_rotation==90:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_back=color_image_back            
        img_rgb_rz_back=np.zeros((img_t_rot_back.shape[0],img_t_rot_back.shape[1],3),np.uint8)
        img_rgb_rz_back=img_rgb_rot_back[resize_param[0]:resize_param[0]+img_t_rot_back.shape[0],resize_param[1]:resize_param[1]+img_t_rot_back.shape[1],:]
        
        ##############################################################################################
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rz_front,img_rgb_rz_back,axis=1) 
        therm_array=np.append(img_t_rot_front,img_t_rot_back,axis=1)
        intensity_image=cv2.cvtColor(therm_array,cv2.COLOR_GRAY2RGB)
        color_image = cv2.addWeighted(color_image,0.7,intensity_image,0.7,0)      
        self.image=color_image
        
    def rgb_2_callback(self,rgb_front,rgb_back):
        ##################################################################################33
        #Front camera info extraction
        #color_image_front = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front = ros_numpy.numpify(rgb_front)
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        self.image_size = img_rgb_rot_front.shape
        ##################################################################################
        #Back camera info extraction
        #color_image_back = bridge.imgmsg_to_cv2(rgb_back, "bgr8")
        color_image_back = ros_numpy.numpify(rgb_back)
        if image_rotation==90:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_back=color_image_back            

        ##############################################################################################        
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rot_front,img_rgb_rot_back,axis=1) 
        #######################################################################################
        self.image=color_image
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=5 #no safety action  
        self.human_command=0
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        self.time_without_msg=5                 # Maximum time without receiving safety messages
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
        black_image=np.zeros((black_image_size[0],black_image_size[1],3), np.uint8)
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
        color_image = cv2.putText(color_image,"***SAFETY SYSTEM***",(50, 330) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        if hri.audio_message!=7:
            color_image = cv2.putText(color_image,"hri status:           "+hri_status_label[int(hri.status)],(50, 360) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"audio message:     "+audio_message_label[int(hri.audio_message)],(50, 390) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"human command:   "+human_command_label[int(hri.human_command)],(50, 420) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"safety action:        "+safety_action_label[int(hri.safety_action)],(50, 450) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        #Print ROBOT ACTION INFO
        color_image = cv2.putText(color_image,"***ROBOT ACTION***",(50, 480) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"action:   "+action_label[int(robot.action)],(50, 510) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"x:           "+str(round(robot.pos_x,2))+"m",(50, 540), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"y:           "+str(round(robot.pos_y,2))+"m",(50, 570) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"speed:       "+str(round(robot.speed,2))+"m/s",(50, 600) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"current node:   "+robot.current_node,(50, 630) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        color_image = cv2.putText(color_image,"goal node:      "+robot.goal_node,(50, 660) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    #Print HUMAN PERCEPTION INFO
    if human.n_human==0: #None human detected
        if human.thermal_detection=="active":
            color_image = cv2.putText(color_image,"***HUMAN PERCEPTION***",(50, 30) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"thermal:     "+human.thermal_detection,(50, 60) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            color_image = cv2.putText(color_image,"***HUMAN PERCEPTION***",(50, 30) , font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #color_image = cv2.putText(color_image,"no human detection:      ",(50, 60) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
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
            color_image = cv2.putText(color_image,"thermal:     "+str(human.thermal_detection),(50, 90) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"motion:      "+motion_labels[int(human.motion)],(50, 120) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"orientation:  "+orientation_labels[int(human.orientation)],(50, 150) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"posture:     "+posture_labels[int(human.posture)],(50, 180) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"distance:    "+str(round(human.distance,2))+"m",(50, 210) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"x:           "+str(round(human.position_x,2))+"m",(50, 240), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"y:           "+str(round(human.position_y,2))+"m",(50, 270) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            color_image = cv2.putText(color_image,"area:        "+str(human.area),(50, 300) , font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
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
        scaling=0.6
        color_image=cv2.resize(color_image,(int(color_image.shape[1]*scaling),int(color_image.shape[0]*scaling))) #resizing it to fit the screen
        
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
    if n_cameras==1:
        if visual_mode==1 or visual_mode==3:    
            if thermal_info==True:
                thermal_front_sub=message_filters.Subscriber('/flir_module_driver/thermal/image_raw', Image) #old topic name only for a single camera
                image_front_sub = message_filters.Subscriber('/camera1/color/image_raw', Image)    #old topic name only for a single camera
                ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, thermal_front_sub], 1, 1)
                ts.registerCallback(human.rgb_thermal_1_callback)
            else:
                rospy.Subscriber('camera/camera1/color/image_raw', Image,human.rgb_1_callback) #really old topic name before thermal info
    else: #n_cameras==2
        if visual_mode==1 or visual_mode==3:    
            if thermal_info==True:
                thermal_front_sub=message_filters.Subscriber('/flir_module_driver1/thermal/image_raw', Image) #new topic names
                image_front_sub = message_filters.Subscriber('/camera1/color/image_raw', Image)    #new topic names
                thermal_back_sub=message_filters.Subscriber('/flir_module_driver2/thermal/image_raw', Image) #new topic names
                image_back_sub = message_filters.Subscriber('/camera2/color/image_raw', Image)    #new topic names
                ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, thermal_front_sub,image_back_sub, thermal_back_sub], 1, 1)
                ts.registerCallback(human.rgb_thermal_2_callback)
            else:
                image_front_sub = message_filters.Subscriber('/camera1/color/image_raw', Image)    #new topic names
                image_back_sub = message_filters.Subscriber('/camera2/color/image_raw', Image)    #new topic names
                ts = message_filters.ApproximateTimeSynchronizer([image_front_sub,image_back_sub], 1, 1)
                ts.registerCallback(human.rgb_2_callback)
                
    if visual_mode>=2:
        rospy.Subscriber('robot_info',robot_msg,robot.robot_callback_info)
        rospy.Subscriber('/robot_pose', Pose, robot.robot_callback_pos) 
        rospy.Subscriber('/nav_vel',geometry_msgs.msg.Twist,robot.robot_callback_vel)    
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        main_counter=main_counter+1
        if visual_mode==2:  
            color_image = np.zeros((black_image_size[0],black_image_size[1],3), np.uint8) 
        else: #visual_mode=1 or 3
            color_image=human.image
        visual_outputs(color_image)
        print(main_counter)  
        print("MODE",visual_mode)
        rate.sleep() #to keep fixed the publishing loop rate
        
        
