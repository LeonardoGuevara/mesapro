#! /usr/bin/python3

#required packages
import rospy #tf
import message_filters #to sync the messages
#from sklearn.ensemble import RandomForestClassifier
from sensor_msgs.msg import Image
import sys
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from numpy import linalg as LA
from cv_bridge import CvBridge
import joblib
#import pickle
import time
import cv2
import yaml
from mesapro.msg import human_detector_msg
##########################################################################################
#Feature extraction variables
n_joints=19
n_features=36
joints_min=7 #minimum number of joints to consider a detection
performance="normal" #OpenPose performance, initial condition
time_threshold=3 #3 seconds as the minimum time between two consecute changes of openpose performance
time_change=0 #initial counter value
avoid_area=0.05 #percentage of the center of the merged image (front+back images) that is not considered as valid when detecting skeletons
search_area=0.3 #percentage of the image that is going to be search to find the pixel with the max temperature (centered on the skeleton joint with highest temp)
##Importing RF model for posture recognition
#posture_classifier_model=rospy.get_param("/hri_camera_detector/posture_classifier_model") #you have to change /hri_camera_detector/ if the node is not named like this
posture_classifier_model="/home/thorvald/rasberry_ws/src/mesapro/config/classifier_model_3D_v2.joblib"
model_rf = joblib.load(posture_classifier_model)   
##OPENPOSE INITIALIZATION 
#openpose_python=rospy.get_param("/hri_camera_detector/openpose_python") #you have to change /hri_camera_detector/ if the node is not named like this
#openpose_models=rospy.get_param("/hri_camera_detector/openpose_models") #you have to change /hri_camera_detector/ if the node is not named like this
openpose_python='/home/thorvald/rasberry_ws/src/openpose/build/python'
openpose_models="/home/thorvald/rasberry_ws/src/openpose/models"
try:
    sys.path.append(openpose_python);
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
params = dict()
params["model_folder"] = openpose_models
if performance=="normal":
    params["net_resolution"] = "-1x256" #the detection performance and the GPU usage depends on this parameter, has to be numbers multiple of 16, the default is "-1x368", and the fastest performance is "-1x160"
else:
    params["net_resolution"] = "-1x352" #High performance
#params["maximize_positives"] = True
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
#Initializating cv_bridge
bridge = CvBridge()
#Setup ROS publiser
pub = rospy.Publisher('human_info_camera', human_detector_msg,queue_size=1) #small queue means priority to new data
msg = human_detector_msg()
#THERMAL INFORMATION
#thermal_info=rospy.get_param("/hri_camera_detector/thermal_info") #you have to change /hri_camera_detector/ if the node is not named like this
thermal_info=True
#Parameters to resize rgbd images from 640x480 size to 160x120 and reoriented them 
temp_thresh=70 #threshold to determine if the temperature if a pixel is considered as higher as human temperature
detection_thresh=0.1 #percentage of pixels in the thermal image which have to satisfy the temp_thresh in order to rise the thermal_detection flag
#image_rotation=rospy.get_param("/hri_camera_detector/image_rotation") #you have to change /hri_camera_detector/ if the node is not named like this
image_rotation=270 #it can be 0,90,270 measured clockwise        
if image_rotation==270:
    resize_param=[120,130,285,380] #[y_init_up,x_init_left,n_pixels_x,n_pixels_y,theta_degrees] assuming portrait mode with image_rotation=270, keeping original aspect ratio 3:4,i.e 291/388 = 120/160 = 3/4
elif image_rotation==90: #IT IS NOT WELL TUNNED YET
    resize_param=[120,105,291,388] #[y_init_up,x_init_left,x_pixels,y_pixels] assuming portrait mode with image_rotation=90
else: #image_rotation==0 
    resize_param=[105,120,388,291] #[y_init_up,x_init_left,x_pixels,y_pixels] assuming portrait mode with image_rotation=0
#VISUALIZATION VARIABLES
#n_cameras=rospy.get_param("/hri_camera_detector/n_cameras") #you have to change /hri_camera_detector/ if the node is not named like this
n_cameras=1 # 1 means that the back camera is emulated by reproducing the front camera image
#openpose_visual=rospy.get_param("/hri_camera_detector/openpose_visual") #you have to change /hri_camera_detector/ if the node is not named like this
openpose_visual=True #to show or not a window with the human detection delivered by openpose
#RGBD CAMERA INTRINSIC,DISTORTION PARAMETERS
#posture_classifier_model=rospy.get_param("/hri_camera_detector/posture_classifier_model") #you have to change /hri_camera_detector/ if the node is not named like this
posture_classifier_model="/home/thorvald/rasberry_ws/src/mesapro/config/classifier_model_3D_v2.joblib"
#config_direct=rospy.get_param("/hri_camera_detector/config_direct") #you have to change /hri_camera_detector/ if the node is not named like this
config_direct="/home/thorvald/rasberry_ws/src/mesapro/config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
camera_param=list(dict.items(parsed_yaml_file["camera_config"]))
intr_param=camera_param[0][1]
dist_param=camera_param[1][1]
#intr_param=[384.7431945800781, 326.4798278808594, 384.34613037109375, 244.670166015625] #[fx cx fy cy]
#dist_param=[-0.056454725563526154, 0.06772931665182114, -0.0011188144562765956, 0.0003955118008889258, -0.022021731361746788] #[k1 k2 t1 t2 k3]
mtx =  np.array([[intr_param[0], 0, intr_param[1]],
                 [0, intr_param[2], intr_param[3]],
                 [0, 0, 1]])
dist=np.array(dist_param)
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        #Variables to store temporally the info of human detectection at each time instant
        self.n_human=1 # considering up to 1 human to track initially
        self.posture=np.zeros([self.n_human,2]) #from camera [posture_label,posture_probability]
        self.centroid=np.zeros([self.n_human,2]) #x,y (pixels) of the human centroid, from camera
        self.features=np.zeros([self.n_human,n_features]) #distances and angles of each skeleton, from camera
        self.orientation=np.zeros([self.n_human,1]) # it can be "front" or "back" if the human is facing the robot or not , from camera
        self.distance=np.zeros([self.n_human,1])  # distance between the robot and the average of the skeleton joints distances taken from the depth image, from camera
        self.image_size=[480,640] #initial condition, assuming portrait mode
        self.camera_id=np.zeros([self.n_human,1]) #to know which camera detected the human
        self.color_image=np.zeros((848,400,3), np.uint8) #rgb image, initial value
        self.therm_array=np.zeros((848,400,1), np.uint8) #intensity map, initial value
        self.depth_array=np.zeros((848,400,1), np.uint8) #depth map, initial value
        self.image_show=np.zeros((848,400,3), np.uint8) #image used for visualization, initial value
        self.intensity=np.zeros([self.n_human,1]) #from thermal camera
        self.thermal_detection=False #assuming no thermal detection as initial value
        self.centroid_3d=np.zeros([self.n_human,2]) #x,y (3d world) of the human centroids
    
    def rgbd_thermal_1_callback(self,rgb_front, depth_front, therm_front):
        ##################################################################################33
        #Front cameras info extraction
        therm_image_front = bridge.imgmsg_to_cv2(therm_front, "mono8") #Gray scale image
        if image_rotation==90:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_front=therm_image_front
        img_t_rot_front=cv2.resize(img_t_rot_front,(resize_param[2],resize_param[3])) #resize to match the rgbd field of view
        
        mat = get_M(0*(np.pi/180), 0*(np.pi/180), 0, 0, 0, 0,resize_param[2],resize_param[3])
        img_t_rot_front= cv2.warpPerspective(img_t_rot_front, mat, (resize_param[2],resize_param[3]))

        
        color_image = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front=cv2.undistort(color_image, mtx, dist) #undistort image 
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        img_rgb_rz_front=np.zeros((img_t_rot_front.shape[0],img_t_rot_front.shape[1],3),np.uint8) # crop to match the thermal field of view
        img_rgb_rz_front=img_rgb_rot_front[resize_param[0]:resize_param[0]+img_t_rot_front.shape[0],resize_param[1]:resize_param[1]+img_t_rot_front.shape[1],:]   
        
        depth_image = bridge.imgmsg_to_cv2(depth_front, "passthrough")
        depth_image_front=cv2.undistort(depth_image, mtx, dist) #undistort image 
        depth_array_front = np.array(depth_image_front, dtype=np.float32)/1000
        if image_rotation==90:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_front=depth_array_front            
        img_d_rz_front=np.zeros((img_t_rot_front.shape[0],img_t_rot_front.shape[1]),np.uint8) #crop to match the thermal field of view
        img_d_rz_front=img_d_rot_front[resize_param[0]:resize_param[0]+img_t_rot_front.shape[0],resize_param[1]:resize_param[1]+img_t_rot_front.shape[1]]     
        
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
        
        #mat = get_M(0, 0, 0, 0, 0, 0,resize_param[2],resize_param[3])
        #img_t_rot_back= cv2.warpPerspective(img_t_rot_back, mat, (resize_param[2],resize_param[3]))

        
        color_image_back=color_image_front
        if image_rotation==90:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_back=color_image_back            
        img_rgb_rz_back=np.zeros((img_t_rot_back.shape[0],img_t_rot_back.shape[1],3),np.uint8)
        img_rgb_rz_back=img_rgb_rot_back[resize_param[0]:resize_param[0]+img_t_rot_back.shape[0],resize_param[1]:resize_param[1]+img_t_rot_back.shape[1],:]
        
        depth_array_back=depth_array_front
        if image_rotation==90:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_back=depth_array_back            
        img_d_rz_back=np.zeros((img_t_rot_back.shape[0],img_t_rot_back.shape[1]),np.uint8)
        img_d_rz_back=img_d_rot_back[resize_param[0]:resize_param[0]+img_t_rot_back.shape[0],resize_param[1]:resize_param[1]+img_t_rot_back.shape[1]]
               
        ##############################################################################################
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rz_front,img_rgb_rz_back,axis=1) 
        depth_array=np.append(img_d_rz_front,img_d_rz_back,axis=1) 
        therm_array=np.append(img_t_rot_front,img_t_rot_back,axis=1)
        
        self.color_image=color_image
        self.depth_array=depth_array
        self.therm_array=therm_array
        #######################################################################################
        
        
    def rgbd_1_callback(self,rgb_front, depth_front):
        ##################################################################################33
        #Front camera info extraction
        color_image = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front=cv2.undistort(color_image, mtx, dist) #undistort image 
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        depth_image = bridge.imgmsg_to_cv2(depth_front, "passthrough")
        depth_image_front=cv2.undistort(depth_image, mtx, dist) #undistort image 
        
        depth_array_front = np.array(depth_image_front, dtype=np.float32)/1000
        if image_rotation==90:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_front=depth_array_front            
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

        depth_array_back=depth_array_front
        if image_rotation==90:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_back=depth_array_back   
               
        ##############################################################################################        
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rot_front,img_rgb_rot_back,axis=1) 
        depth_array=np.append(img_d_rot_front,img_d_rot_back,axis=1) 
        therm_array=np.zeros((color_image.shape[0],color_image.shape[1]),np.uint8)

        self.color_image=color_image
        self.depth_array=depth_array
        self.therm_array=therm_array
        #######################################################################################

    
    def rgbd_thermal_2_callback(self,rgb_front, depth_front, therm_front,rgb_back, depth_back, therm_back):
        ##################################################################################33
        #Front cameras info extraction
        therm_image_front = bridge.imgmsg_to_cv2(therm_front, "mono8") #Gray scale image
        if image_rotation==90:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_front=therm_image_front
        img_t_rot_front=cv2.resize(img_t_rot_front,(resize_param[2],resize_param[3])) #resize to match the rgbd field of view
        
        color_image = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front=cv2.undistort(color_image, mtx, dist) #undistort image 
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        img_rgb_rz_front=np.zeros((img_t_rot_front.shape[0],img_t_rot_front.shape[1],3),np.uint8) #crop to match the thermal field of view
        img_rgb_rz_front=img_rgb_rot_front[resize_param[0]:resize_param[0]+img_t_rot_front.shape[0],resize_param[1]:resize_param[1]+img_t_rot_front.shape[1],:]   
        
        depth_image = bridge.imgmsg_to_cv2(depth_front, "passthrough")
        depth_image_front=cv2.undistort(depth_image, mtx, dist) #undistort image 
        
        depth_array_front = np.array(depth_image_front, dtype=np.float32)/1000
        if image_rotation==90:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_front=depth_array_front            
        img_d_rz_front=np.zeros((img_t_rot_front.shape[0],img_t_rot_front.shape[1]),np.uint8) #crop to match the thermal field of view
        img_d_rz_front=img_d_rot_front[resize_param[0]:resize_param[0]+img_t_rot_front.shape[0],resize_param[1]:resize_param[1]+img_t_rot_front.shape[1]]     
        
        self.image_size = img_rgb_rz_front.shape
        ##################################################################################
        #Back cameras info extraction
        therm_image_back = bridge.imgmsg_to_cv2(therm_back, "bgr8")
        if image_rotation==90:
            img_t_rot_back=cv2.rotate(therm_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_back=cv2.rotate(therm_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_back=therm_image_back
        img_t_rot_back=cv2.resize(img_t_rot_back,(resize_param[2],resize_param[3]))        
        
        color_image = bridge.imgmsg_to_cv2(rgb_back, "bgr8")
        color_image_back=cv2.undistort(color_image, mtx, dist) #undistort image 
        
        if image_rotation==90:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_back=color_image_back            
        img_rgb_rz_back=np.zeros((img_t_rot_back.shape[0],img_t_rot_back.shape[1],3),np.uint8)
        img_rgb_rz_back=img_rgb_rot_back[resize_param[0]:resize_param[0]+img_t_rot_back.shape[0],resize_param[1]:resize_param[1]+img_t_rot_back.shape[1],:]
        
        depth_image = bridge.imgmsg_to_cv2(depth_back, "passthrough")
        depth_image_back=cv2.undistort(depth_image, mtx, dist) #undistort image 
        
        depth_array_back = np.array(depth_image_back, dtype=np.float32)/1000
        if image_rotation==90:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_back=depth_array_back            
        img_d_rz_back=np.zeros((img_t_rot_back.shape[0],img_t_rot_back.shape[1]),np.uint8)
        img_d_rz_back=img_d_rot_back[resize_param[0]:resize_param[0]+img_t_rot_back.shape[0],resize_param[1]:resize_param[1]+img_t_rot_back.shape[1]]
               
        ##############################################################################################
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rz_front,img_rgb_rz_back,axis=1) 
        depth_array=np.append(img_d_rz_front,img_d_rz_back,axis=1) 
        therm_array=np.append(img_t_rot_front,img_t_rot_back,axis=1)
        
        self.color_image=color_image
        self.depth_array=depth_array
        self.therm_array=therm_array
        #######################################################################################
        
        
    def rgbd_2_callback(self,rgb_front, depth_front,rgb_back, depth_back):
        ##################################################################################33
        #Front camera info extraction
        color_image = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        color_image_front=cv2.undistort(color_image, mtx, dist) #undistort image 
        
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        depth_image = bridge.imgmsg_to_cv2(depth_front, "passthrough")
        depth_image_front=cv2.undistort(depth_image, mtx, dist) #undistort image 
        
        depth_array_front = np.array(depth_image_front, dtype=np.float32)/1000
        if image_rotation==90:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_front=cv2.rotate(depth_array_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_front=depth_array_front            
        
        self.image_size = img_rgb_rot_front.shape
        ##################################################################################
        #Back camera info extraction
        color_image = bridge.imgmsg_to_cv2(rgb_back, "bgr8")
        color_image_back=cv2.undistort(color_image, mtx, dist) #undistort image 
        
        if image_rotation==90:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_back=cv2.rotate(color_image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_back=color_image_back            

        depth_image = bridge.imgmsg_to_cv2(depth_back, "passthrough")
        depth_image_back=cv2.undistort(depth_image, mtx, dist) #undistort image 
        
        depth_array_back = np.array(depth_image_back, dtype=np.float32)/1000
        if image_rotation==90:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_d_rot_back=cv2.rotate(depth_array_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_d_rot_back=depth_array_back   
                
        ##############################################################################################        
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(img_rgb_rot_front,img_rgb_rot_back,axis=1) 
        depth_array=np.append(img_d_rot_front,img_d_rot_back,axis=1) 
        therm_array=np.zeros((color_image.shape[0],color_image.shape[1]),np.uint8)
        self.color_image=color_image
        self.depth_array=depth_array
        self.therm_array=therm_array
        #######################################################################################
    
    def processing(self,color_image,depth_array,therm_array):
        global time_change, performance
        if time.time()-time_change>time_threshold: #only admit update if time_threshold is satisfied
            performance_past=performance
            if self.n_human>0:
                if min(self.distance)>7:
                    performance="high"
                else:
                    performance="normal"
            else:
                performance="normal"
            
            if performance=="high" and performance_past!=performance:
                params = dict()
                params["model_folder"] = openpose_models
                params["net_resolution"] = "-1x352" #the detection performance and the GPU usage depends on this parameter, has to be numbers multiple of 16, the default is "-1x368", and the fastest performance is "-1x160"
                #params["maximize_positives"] = True
                opWrapper.stop()
                opWrapper.configure(params)
                opWrapper.start()
                time_change=time.time()
                #datum = op.Datum()
            elif performance=="normal" and performance_past!=performance:
                params = dict()
                params["model_folder"] = openpose_models
                params["net_resolution"] = "-1x256" #the detection performance and the GPU usage depends on this parameter, has to be numbers multiple of 16, the default is "-1x368", and the fastest performance is "-1x160"
                #params["maximize_positives"] = True
                opWrapper.stop()
                opWrapper.configure(params)
                opWrapper.start()
                time_change=time.time()
                #datum = op.Datum()

        #print("PERFORMANCE",performance)
        #print("MIN DIS",min(self.distance))
        ####################################################################################################
        datum.cvInputData = color_image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #Keypoints extraction using OpenPose
        keypoints=datum.poseKeypoints
        self.image_show=datum.cvOutputData #for visualization 
         
        if keypoints is None: #if there is no human skeleton detected
            #print('No human detected')
            self.n_human=0
        else: #if there is at least 1 human skeleton detected
            #Feature extraction
            self.feature_extraction_3D(keypoints,depth_array,therm_array,n_joints,n_features)
        #Thermal detection flag
        if thermal_info==False:
            self.thermal_detection=False
        else:
            if self.n_human>0: 
                self.thermal_detection=True #if a human was detected, it means there is thermal detection
            else: #if no human was detected, we should evaluate if there is thermal detection or not
                n_pixels=therm_array.shape[0]*therm_array.shape[1]
                filt=np.argwhere(therm_array>temp_thresh)
                if filt.shape[0]>detection_thresh*n_pixels:
                    self.thermal_detection=True #if there are enough number of pixels with high intensity, it means there is thermal detection even if the human is not detected by the Openpose
                else:
                    self.thermal_detection=False   
        #Publish     
        if self.n_human>0:
            #print('Human detection')
            msg.posture = [int(x) for x in list(self.posture[:,0])] #to ensure publish int
            msg.posture_prob = list(self.posture[:,1])
            msg.centroid_x =list(self.centroid[:,0])
            msg.centroid_y =list(self.centroid[:,1])
            msg.position_x =list(self.centroid_3d[:,0])
            msg.position_y =list(self.centroid_3d[:,1])
            msg.distance = list(self.distance[:,0])
            msg.orientation = [int(x) for x in list(self.orientation[:,0])] #to ensure publish int
            msg.camera_id= [int(x) for x in list(self.camera_id[:,0])] #to ensure publish int
            msg.image_size= self.image_size  #asumming both cameras has the same image size
            msg.thermal_detection=int(self.thermal_detection)
            msg.intensity=[int(x) for x in list(self.intensity[:,0])] #to ensure publish int 
            pub.publish(msg)
        elif self.n_human==0 and self.thermal_detection==True:
            #print('Thermal detection')
            msg.thermal_detection=self.thermal_detection
            pub.publish(msg)
    
################################################################################################################            
    def feature_extraction_3D(self,poseKeypoints,depth_array,therm_array,n_joints,n_features):
        posture=np.zeros([len(poseKeypoints[:,0,0]),2])
        centroid=np.zeros([len(poseKeypoints[:,0,0]),2]) 
        centroid_3d=np.zeros([len(poseKeypoints[:,0,0]),2]) 
        features=np.zeros([len(poseKeypoints[:,0,0]),n_features]) 
        orientation=np.zeros([len(poseKeypoints[:,0,0]),1])
        distance=np.zeros([len(poseKeypoints[:,0,0]),1]) 
        camera_id=np.zeros([len(poseKeypoints[:,0,0]),1]) 
        intensity=np.zeros([len(poseKeypoints[:,0,0]),1])
        index_to_keep=[]
        for kk in range(0,len(poseKeypoints[:,0,0])):
            #Orientation inference using nose, ears, eyes keypoints
            if poseKeypoints[kk,0,0]!=0 and poseKeypoints[kk,15,0]!=0 and poseKeypoints[kk,16,0]!=0 : 
                orientation[kk,:]=0 # facing the robot
            elif poseKeypoints[kk,0,0]==0 and poseKeypoints[kk,15,0]==0 and poseKeypoints[kk,16,0]==0 : 
                orientation[kk,:]=1 # giving the back
            elif poseKeypoints[kk,0,0]!=0 and (poseKeypoints[kk,15,0]==0 and poseKeypoints[kk,17,0]==0)  and poseKeypoints[kk,16,0]!=0: 
                orientation[kk,:]=2  #showing the left_side
            elif poseKeypoints[kk,0,0]!=0 and poseKeypoints[kk,15,0]!=0 and (poseKeypoints[kk,18,0]==0 and poseKeypoints[kk,16,0]==0): 
                orientation[kk,:]=3  #showing the right_side
           
            ### 3D Feature extraction####
            #Using only the important joints
            joints_x_init=poseKeypoints[kk,0:n_joints,0]
            joints_y_init=poseKeypoints[kk,0:n_joints,1]   
            joints_z_init=[0]*n_joints   
            joints_temp_init=[0]*n_joints
            for k in range(0,n_joints):
                #in case keypoints are out of image range (necessary when two images were merged)
                if int(joints_y_init[k])>=len(depth_array[:,0]):
                    joints_y_init[k]=len(depth_array[:,0])-1
                if int(joints_x_init[k])>=len(depth_array[0,:]):
                    joints_x_init[k]=len(depth_array[0,:])-1
                joints_z_init[k]=depth_array[int(joints_y_init[k]),int(joints_x_init[k])]
                joints_temp_init[k]=therm_array[int(joints_y_init[k]),int(joints_x_init[k])]
            #Normalization and scaling
            #Translation
            J_sum_x=0
            J_sum_y=0
            J_sum_z=0
            for k in range(0,n_joints):   
               J_sum_x=joints_x_init[k]+J_sum_x
               J_sum_x=joints_y_init[k]+J_sum_y
               J_sum_z=joints_z_init[k]+J_sum_z
            J_mean_x=J_sum_x/(n_joints) 
            J_mean_y=J_sum_y/(n_joints)
            J_mean_z=J_sum_z/(n_joints)
            joints_x_trans=joints_x_init-J_mean_x 
            joints_y_trans=joints_y_init-J_mean_y
            joints_z_trans=joints_z_init-J_mean_z
            #Normalization   
            J_sum2=0
            valid=0
            temp_max=0
            joint_temp_max=0
            for k in range(0,n_joints):  
                J_sum2=joints_x_trans[k]**2+joints_y_trans[k]**2+joints_z_trans[k]**2+J_sum2  
                if joints_x_trans[k]!=0 and joints_y_trans[k]!=0 and joints_z_trans[k]!=0:
                    valid=valid+1
                    if temp_max<joints_temp_init[k]:
                        temp_max=joints_temp_init[k] #updating the joint highest temp
                        joint_temp_max=k #index of the joint with the highest temp
            Js=sqrt(J_sum2/(n_joints))
            #Find pixel with max temp around the joint with the highest temp of the skeleton
            if thermal_info==True and temp_max>0:
                #Defining the limits of the search
                init_search_x=int(joints_x_init[joint_temp_max]-(self.image_size[1]*search_area/2))
                if init_search_x<0:
                    init_search_x=0
                end_search_x=int(joints_x_init[joint_temp_max]+(self.image_size[1]*search_area/2))
                if end_search_x>2*self.image_size[1]-2:
                    end_search_x=2*self.image_size[1]-2
                init_search_y=int(joints_y_init[joint_temp_max]-(self.image_size[0]*search_area/2))
                if init_search_y<0:
                    init_search_y=0
                end_search_y=int(joints_y_init[joint_temp_max]+(self.image_size[0]*search_area/2))
                if end_search_y>self.image_size[0]-1:
                    end_search_y=self.image_size[0]-1
                #Search
                for i in range(init_search_x,end_search_x):
                    if temp_max>=temp_thresh:
                        break #no more search
                    for j in range(init_search_y,end_search_y):
                        pixel_temp=therm_array[j,i]
                        if temp_max<pixel_temp:
                            temp_max=pixel_temp
                            if temp_max>=temp_thresh:
                                break #no more search
            #only continue if there are enough joints detected on the skeleton and/or temp_max satisfy the threshold
            if Js!=0 and valid>=joints_min and ((temp_max>=temp_thresh and thermal_info==True) or thermal_info==False):
                intensity[kk,0]=temp_max
                joints_x = joints_x_trans/Js      
                joints_y = joints_y_trans/Js
                joints_z = joints_z_trans/Js
                   
                #Distances from each joint to the neck joint
                dist=[0]*n_joints
                for k in range(0,n_joints):
                   dist[k]=sqrt((joints_x[k]-joints_x[1])**2+(joints_y[k]-joints_y[1])**2+(joints_z[k]-joints_z[1])**2)  
            
                #Vectors between joints
                v1_2=[joints_x[1]-joints_x[2], joints_y[1]-joints_y[2], joints_z[1]-joints_z[2]]  
                v2_3=[joints_x[2]-joints_x[3], joints_y[2]-joints_y[3], joints_z[2]-joints_z[3]]  
                v3_4=[joints_x[3]-joints_x[4], joints_y[3]-joints_y[4], joints_z[3]-joints_z[4]]  
                v1_5=[joints_x[1]-joints_x[5], joints_y[1]-joints_y[5], joints_z[1]-joints_z[5]]  
                v5_6=[joints_x[5]-joints_x[6], joints_y[5]-joints_y[6], joints_z[5]-joints_z[6]]  
                v6_7=[joints_x[6]-joints_x[7], joints_y[6]-joints_y[7], joints_z[6]-joints_z[7]]  
                v1_0=[joints_x[1]-joints_x[0], joints_y[1]-joints_y[0], joints_z[1]-joints_z[0]]  
                v0_15=[joints_x[0]-joints_x[15], joints_y[0]-joints_y[15], joints_z[0]-joints_z[15]]  
                v15_17=[joints_x[15]-joints_x[17], joints_y[15]-joints_y[17], joints_z[15]-joints_z[17]]  
                v0_16=[joints_x[0]-joints_x[16], joints_y[0]-joints_y[16], joints_z[0]-joints_z[16]]
                v16_18=[joints_x[16]-joints_x[18], joints_y[16]-joints_y[18], joints_z[16]-joints_z[18]]  
                v1_8=[joints_x[1]-joints_x[8], joints_y[1]-joints_y[8], joints_z[1]-joints_z[8]]
                v8_9=[joints_x[8]-joints_x[9], joints_y[8]-joints_y[9], joints_z[8]-joints_z[9]]  
                v9_10=[joints_x[9]-joints_x[10], joints_y[9]-joints_y[10], joints_z[9]-joints_z[10]]  
                v10_11=[joints_x[10]-joints_x[11], joints_y[10]-joints_y[11], joints_z[10]-joints_z[11]]  
                v8_12=[joints_x[8]-joints_x[12], joints_y[8]-joints_y[12], joints_z[8]-joints_z[12]]  
                v12_13=[joints_x[12]-joints_x[13], joints_y[12]-joints_y[13], joints_z[12]-joints_z[13]]  
                v13_14=[joints_x[13]-joints_x[14], joints_y[13]-joints_y[14], joints_z[13]-joints_z[14]] 
        
                #Angles between joints  
                angles=[0]*(n_joints-2) #17 angles
                angles[0] = atan2(LA.norm(np.cross(v15_17,v0_15)),np.dot(v15_17,v0_15))
                angles[1] = atan2(LA.norm(np.cross(v0_15,v1_0)),np.dot(v0_15,v1_0))
                angles[2] = atan2(LA.norm(np.cross(v16_18,v0_16)),np.dot(v16_18,v0_16))
                angles[3] = atan2(LA.norm(np.cross(v0_16,v1_0)),np.dot(v0_16,v1_0))
                angles[4] = atan2(LA.norm(np.cross(v1_0,v1_2)),np.dot(v1_0,v1_2))
                angles[5] = atan2(LA.norm(np.cross(v1_2,v2_3)),np.dot(v1_2,v2_3))
                angles[6] = atan2(LA.norm(np.cross(v2_3,v3_4)),np.dot(v2_3,v3_4))
                angles[7] = atan2(LA.norm(np.cross(v1_0,v1_5)),np.dot(v1_0,v1_5))
                angles[8] = atan2(LA.norm(np.cross(v1_5,v5_6)),np.dot(v1_5,v5_6))
                angles[9] = atan2(LA.norm(np.cross(v5_6,v6_7)),np.dot(v5_6,v6_7))
                angles[10] = atan2(LA.norm(np.cross(v1_2,v1_8)),np.dot(v1_2,v1_8))
                angles[11] = atan2(LA.norm(np.cross(v1_8,v8_9)),np.dot(v1_8,v8_9))
                angles[12] = atan2(LA.norm(np.cross(v8_9,v9_10)),np.dot(v8_9,v9_10))
                angles[13] = atan2(LA.norm(np.cross(v9_10,v10_11)),np.dot(v9_10,v10_11))
                angles[14] = atan2(LA.norm(np.cross(v1_8,v8_12)),np.dot(v1_8,v8_12))
                angles[15] = atan2(LA.norm(np.cross(v8_12,v12_13)),np.dot(v8_12,v12_13))
                angles[16] = atan2(LA.norm(np.cross(v12_13,v13_14)),np.dot(v12_13,v13_14))
                
                #HUMAN FEATURES CALCULATION
                features[kk,:]=dist+angles  
                #HUMAN POSTURE RECOGNITION
                X=np.array(features[kk,:]).transpose()
                posture[kk,0]=model_rf.predict([X])
                prob_max=0
                prob=model_rf.predict_proba([X])
                for ii in range(0,prob.shape[1]): #depends of the number of gestures to classified
                    if prob[0,ii]>=prob_max:
                        prob_max=prob[0,ii]
                posture[kk,1]=prob_max 
                #HUMAN DISTANCE AND CENTROID CALCULATION
                n_joints_cent=0
                dist_sum=0
                x_sum=0
                y_sum=0
                
                for k in range(0,n_joints):
                    if joints_x_init[k]!=0 and joints_y_init[k]!=0 and joints_z_init[k]!=0:
                        if k==0 or k==1 or k==2 or k==8 or k==5 or k==9 or k==12: #Only consider keypoints in the center of the body
                            dist_sum=joints_z_init[k]+dist_sum
                            x_sum=x_sum+joints_x_init[k]
                            y_sum=y_sum+joints_y_init[k]
                            n_joints_cent=n_joints_cent+1
                #Only continue if there is at least 1 joint with x*y*z!=0 in the center of the body
                if n_joints_cent!=0:
                    distance[kk,:]=dist_sum/n_joints_cent
                    if joints_x_init[1]!=0 and joints_y_init[1]!=0 and joints_z_init[1]!=0: #If neck joint exists, then this is choosen as centroid
                        centroid[kk,0]=joints_x_init[1]
                        centroid[kk,1]=joints_y_init[1]
                    else: #the centroid is an average
                        centroid[kk,0]=x_sum/n_joints_cent
                        centroid[kk,1]=y_sum/n_joints_cent
                    #Only continue if the human is not detected in between the two images merged
                    width=self.image_size[1]                       
                    if centroid[kk,0]<=width-width*avoid_area or centroid[kk,0]>=width+width*avoid_area: 
                        index_to_keep=index_to_keep+[kk]   
                        #CAMERA ID
                        if centroid[kk,0]<=width: #camera front
                            camera_id[kk]=0
                            #HUMAN XY POSITION IN PIXELS INTO 3D WORLD
                            centroid_3d[kk,0] = distance[kk,0] #z-axis in camera 3d world is the positive x-axis of robot frame
                            #MAP CENTROID COORDINATES FROM THE CROPPED AND ROTATED IMAGE TO THE ORIGINAL IMAGE (UNDISTORTED)
                            if thermal_info==True:
                                orig_centroid_y=centroid[kk,0]+resize_param[1]
                            else: #means that the original image was not cropped
                                orig_centroid_y=centroid[kk,0]
                            if image_rotation==270:
                                centroid_3d[kk,1] = -(orig_centroid_y - intr_param[3]) * distance[kk,0]  / intr_param[2] #y-axis in camera 3d world is the negative y-axis of the robot frame
                            else: #rotation==0
                                centroid_3d[kk,1] = -(orig_centroid_y - intr_param[1]) * distance[kk,0]  / intr_param[0] #y-axis in camera 3d world is the negative y-axis of the robot frame   
                        else:#camera back
                            camera_id[kk]=1
                            #HUMAN XY POSITION IN PIXELS INTO 3D WORLD
                            centroid_3d[kk,0] = -distance[kk,0] #z-axis in camera 3d world is the negative x-axis of robot frame
                            #MAP CENTROID COORDINATES FROM THE CROPPED AND ROTATED IMAGE TO THE ORIGINAL IMAGE (UNDISTORTED)
                            if thermal_info==True:
                                orig_centroid_y=centroid[kk,0]-width+resize_param[1]
                            else: #means that the original image was not cropped
                                orig_centroid_y=centroid[kk,0]-width
                            if image_rotation==270:
                                centroid_3d[kk,1] = (orig_centroid_y - intr_param[3]) * distance[kk,0] / intr_param[2] #y-axis in camera 3d world is the positive y-axis of the robot frame    
                            else: #rotation==0
                                centroid_3d[kk,1] = (orig_centroid_y - intr_param[1]) * distance[kk,0] / intr_param[0] #y-axis in camera 3d world is the positive y-axis of the robot frame                       
                        

        #return features,posture,orientation,distance,centroid,camera_id
        if index_to_keep!=[]:
            self.features=features[np.array(index_to_keep)]
            self.posture=posture[np.array(index_to_keep)]
            self.orientation=orientation[np.array(index_to_keep)]
            self.distance=distance[np.array(index_to_keep)]
            self.centroid=centroid[np.array(index_to_keep)]
            self.centroid_3d=centroid_3d[np.array(index_to_keep)]
            self.camera_id=camera_id[np.array(index_to_keep)]
            self.intensity=intensity[np.array(index_to_keep)]
            self.n_human=len(distance)
        else:
            self.n_human=0

""" Get Perspective Projection Matrix """
def get_M(theta, phi, gamma, dx, dy, dz,width, height):
    
    w = width
    h = height
    d = np.sqrt(height**2 + width**2)
    f = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
    #f = d /1.8
    dz=f
    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -w/0.3], #originally -w/2
                    [0, 1, -h/3], #originally -h/2
                    [0, 0, 1],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, w/0.3, 0],
                    [0, f, h/3, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))
        
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node
    time_init=time.time() 
    human=human_class()  
    rospy.init_node('human_detector_camera',anonymous=True)
    # Setup and call subscription
    if n_cameras==1:
        #Camara front
        if thermal_info==True:
            thermal_front_sub=message_filters.Subscriber('/flir_module_driver/thermal/image_raw', Image) #old topic name only for a single camera
        image_front_sub = message_filters.Subscriber('/camera/color/image_raw', Image) #old topic name only for a single camera
        depth_front_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image) #old topic name only for a single camera
        if thermal_info==True:
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub, thermal_front_sub], 1, 1)
            ts.registerCallback(human.rgbd_thermal_1_callback)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub], 1, 1)    
            ts.registerCallback(human.rgbd_1_callback)
    else: #n_camaras==2
        #Camara front and back
        if thermal_info==True:
            thermal_front_sub=message_filters.Subscriber('/flir_module_driver1/thermal/image_raw', Image) #new topic names
            thermal_back_sub=message_filters.Subscriber('/flir_module_driver2/thermal/image_raw', Image) #new topic names
        image_front_sub = message_filters.Subscriber('/camera1/color/image_raw', Image) #new topic names
        depth_front_sub = message_filters.Subscriber('/camera1/aligned_depth_to_color/image_raw', Image) #new topic names
        image_back_sub = message_filters.Subscriber('/camera2/color/image_raw', Image) #new topic names
        depth_back_sub = message_filters.Subscriber('/camera2/aligned_depth_to_color/image_raw', Image) #new topic names
        if thermal_info==True:
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub, thermal_front_sub,image_back_sub, depth_back_sub, thermal_back_sub], 1, 1)
            ts.registerCallback(human.rgbd_thermal_2_callback)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub,image_back_sub, depth_back_sub], 1, 1)    
            ts.registerCallback(human.rgbd_2_callback)
    if openpose_visual==False:
        rospy.spin()
    else:
        #Rate setup
        rate = rospy.Rate(1/0.01) # ROS publishing rate in Hz
        while not rospy.is_shutdown():
            #cv2.imshow("Human detector",human.color_image )
            #cv2.waitKey(10)  
            color_image=human.color_image
            depth_array=human.depth_array
            therm_array=human.therm_array
            human.processing(color_image,depth_array,therm_array)
            image=human.image_show
            centroids_x=human.centroid[:,0]
            centroids_y=human.centroid[:,1]
            for i in range(0,1):
                if human.camera_id[i,0]==1:
                    print("X",human.centroid_3d[i,0])
                    print("Y",human.centroid_3d[i,1])
                    print("Dist",human.distance[i,:])
                    print("camera",human.camera_id[i,:])
            if openpose_visual==True:            
                intensity_image=cv2.cvtColor(therm_array,cv2.COLOR_GRAY2RGB)
                if thermal_info==True:
                    #print("COLOR",image.shape)
                    #print("INTENSITY",intensity_image.shape)
                    image = cv2.addWeighted(image,0.7,intensity_image,0.7,0)
                if human.n_human>0:
                    for k in range(0,len(centroids_x)):    
                        center_coordinates = (int(centroids_x[k]), int(centroids_y[k])) 
                        image = cv2.circle(image, center_coordinates, 5, (255, 0, 0), 20) #BLUE           
                image=cv2.resize(image,(int(image.shape[1]*2.5),int(image.shape[0]*2.5)))
                cv2.imshow("Human detector",image  )
                cv2.waitKey(10)  
            
