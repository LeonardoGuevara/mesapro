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
import ros_numpy
##########################################################################################
#Initializating cv_bridge
bridge = CvBridge()
#THERMAL INFORMATION
#thermal_info=rospy.get_param("/hri_camera_detector/thermal_info") #you have to change /hri_camera_detector/ if the node is not named like this
thermal_info=True
#Parameters to resize rgbd images from 640x480 size to 160x120 and reoriented them 
#image_rotation=rospy.get_param("/hri_camera_detector/image_rotation") #you have to change /hri_camera_detector/ if the node is not named like this
image_rotation=270 #it can be 0,90,270 measured clockwise        
if image_rotation==270:
    resize_param=[120,130,285,380] #[y_init_up,x_init_left,n_pixels_x,n_pixels_y] assuming portrait mode with image_rotation=270, keeping original aspect ratio 3:4,i.e 285/380 = 120/160 = 3/4
elif image_rotation==90: #IT IS NOT WELL TUNNED YET
    resize_param=[120,105,285,380] 
else: #image_rotation==0 #IT IS NOT WELL TUNNED YET
    resize_param=[120,130,380,285] 
#VISUALIZATION VARIABLES
#n_cameras=rospy.get_param("/hri_camera_detector/n_cameras") #you have to change /hri_camera_detector/ if the node is not named like this
n_cameras=1 # 1 means that the back camera is emulated by reproducing the front camera image
#openpose_visual=rospy.get_param("/hri_camera_detector/openpose_visual") #you have to change /hri_camera_detector/ if the node is not named like this
openpose_visual=True #to show or not a window with the human detection delivered by openpose
#RGBD CAMERA INTRINSIC,DISTORTION PARAMETERS
#config_direct=rospy.get_param("/hri_camera_detector/config_direct") #you have to change /hri_camera_detector/ if the node is not named like this
config_direct="/home/leo/rasberry_ws/src/mesapro/config/"
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
        #therm_image_front = bridge.imgmsg_to_cv2(therm_front, "mono8") #Gray scale image
        therm_image_front = ros_numpy.numpify(therm_front) # replacing cv_bridge
        
        if image_rotation==90:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_t_rot_front=cv2.rotate(therm_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_t_rot_front=therm_image_front
        img_t_rot_front=cv2.resize(img_t_rot_front,(resize_param[2],resize_param[3])) #resize to match the rgbd field of view
        
        color_image = bridge.imgmsg_to_cv2(rgb_front, "bgr8")
        print("COLOR OLD",color_image[50,50,:])
        
        color_image = ros_numpy.numpify(rgb_front) # replacing cv_bridge
        color_image = color_image[...,[2,1,0]].copy() #from bgr to rgb
        print("COLOR NEW",color_image[50,50,:])
        color_image_front=cv2.undistort(color_image, mtx, dist) #undistort image 
        if image_rotation==90:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation==270:
            img_rgb_rot_front=cv2.rotate(color_image_front,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: #0 degrees
            img_rgb_rot_front=color_image_front            
        
        img_rgb_rz_front=np.zeros((img_t_rot_front.shape[0],img_t_rot_front.shape[1],3),np.uint8) # crop to match the thermal field of view
        img_rgb_rz_front=img_rgb_rot_front[resize_param[0]:resize_param[0]+img_t_rot_front.shape[0],resize_param[1]:resize_param[1]+img_t_rot_front.shape[1],:]   
        
        #depth_image = bridge.imgmsg_to_cv2(depth_front, "passthrough")
        depth_image = ros_numpy.numpify(depth_front)
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
            thermal_front_sub=message_filters.Subscriber('/flir_module_driver1/thermal/image_raw', Image) #new topic name only for a single camera
        image_front_sub = message_filters.Subscriber('/camera1/color/image_raw', Image) #new topic name only for a single camera
        depth_front_sub = message_filters.Subscriber('/camera1/aligned_depth_to_color/image_raw', Image) #new topic name only for a single camera
        if thermal_info==True:
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub, thermal_front_sub], 5, 1)
            ts.registerCallback(human.rgbd_thermal_1_callback)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub], 5, 1)    
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
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub, thermal_front_sub,image_back_sub, depth_back_sub, thermal_back_sub], 5, 1)
            ts.registerCallback(human.rgbd_thermal_2_callback)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub,image_back_sub, depth_back_sub], 5, 1)    
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
            #human.processing(color_image,depth_array,therm_array)
            if openpose_visual==True:            
                intensity_image=cv2.cvtColor(therm_array,cv2.COLOR_GRAY2RGB)
                image=color_image
                if thermal_info==True:
                    #print("COLOR",image.shape)
                    #print("INTENSITY",intensity_image.shape)
                    image = cv2.addWeighted(image,0.7,intensity_image,0.7,0)
                cv2.imshow("Human detector",image  )
                cv2.waitKey(10)  
            
