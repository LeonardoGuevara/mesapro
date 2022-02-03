#! /usr/bin/python3

#required packages
import rospy 
import message_filters #to sync the messages
from sensor_msgs.msg import Image
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from cv_bridge import CvBridge
import cv2
import sys

##########################################################################################
##Openpose initialization 
#openpose_python=rospy.get_param("/hri_camera_detector/openpose_python") #you have to change /hri_camera_detector/ if the node is not named like this
#openpose_models=rospy.get_param("/hri_camera_detector/openpose_models") #you have to change /hri_camera_detector/ if the node is not named like this
openpose_python='/home/leo/rasberry_ws/src/mesapro/openpose/build/python'
openpose_models="/home/leo/rasberry_ws/src/mesapro/openpose/models"
try:
    sys.path.append(openpose_python);
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
params = dict()
params["model_folder"] = openpose_models
params["net_resolution"] = "-1x256" #the detection performance and the GPU usage depends on this parameter, has to be numbers multiple of 16, the default is "-1x368", and the fastest performance is "-1x160"
#params["camera_resolution"]= "848x480"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
#Feature extraction variables
n_joints=19

#Initializating cv_bridge
bridge = CvBridge()
visualization=True
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        #Variables to store temporally the info of human detectection at each time instant
        self.n_human=1 # considering up to 1 human to track initially
        self.centroid=np.zeros([self.n_human,2]) #x,y (pixels) of the human centroid, from camera
        self.image_size=[120,160] #initial condition of each camera
        self.camera_id=np.zeros([self.n_human,1]) #to know which camera detected the human
        self.image=np.zeros((120,160), np.uint8) #initial condition
        
    def image_callback(self,thermal_front,rgb_front,depth_front):
        #Front camera info extraction
        img_thermal = bridge.imgmsg_to_cv2(thermal_front, "bgr8") #theated as a colored image even if is a gray scale image
        
        #Rotate camera (assuming it is placed on the robot in portrait mode)
        img_t_rot=cv2.rotate(img_thermal,cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_rgb=bridge.imgmsg_to_cv2(rgb_front, "bgr8") 
        #Rotate camera (assuming it is placed on the robot in portrait mode)
        print(img_rgb.shape)
        img_rgb_rot=cv2.rotate(img_rgb,cv2.ROTATE_90_COUNTERCLOCKWISE)
        #Parameters to crop rgb images from any size to 160x120
        y=150 #330 for 1280x720, 150 for 640x480
        x=125 #100 for 1280x720, 125 for 640x480
        h=img_rgb_rot.shape[0]-230 #630 for 1280x720, 230 for 640x480
        w=img_rgb_rot.shape[1]-180 #230 for 1280x720, 180 for 640x480
        img_rgb_crop = img_rgb_rot[y:y+h, x:x+w]
        img_rgb_rz=cv2.resize(img_rgb_crop, (120, 160))
        ##################################################################################
        #Back camera info extraction
        #image_back= image_front
        #Rotate camera (assuming it is placed on the robot in portrait mode)
        #image_b_rot=cv2.rotate(image_back,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        #imageToProcess=np.append(image_f_rot,image_b_rot,axis=1) 
        
        imageToProcess=np.append(img_t_rot,img_rgb_rz,axis=1) 
        #imageToProcess=img_t_rot
        #imageToProcess=img_rgb_rot
        ##Scale only for visualization purposes
        if visualization==True:
            imageToProcess=cv2.resize(imageToProcess, (640, 480))
            self.image=imageToProcess
        
        ###################################################################################
        datum.cvInputData = imageToProcess#image_front
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #Keypoints extraction using OpenPose
        keypoints=datum.poseKeypoints
        self.image=datum.cvOutputData
        if keypoints is None: #if there is no human skeleton detected
            print('No human detected')
            self.n_human=0
        else: #if there is at least 1 human skeleton detected
            #Feature extraction
            self.centroid_extraction(keypoints,n_joints)
            #self.image=imageToProcess
            self.image_size = self.image.shape
            print('Human detection')
    
    def centroid_extraction(self,poseKeypoints,n_joints):
        centroid=np.zeros([len(poseKeypoints[:,0,0]),2]) 
        camera_id=np.zeros([len(poseKeypoints[:,0,0]),1]) 
        index_to_keep=[]
        for kk in range(0,len(poseKeypoints[:,0,0])):
            #Using only the important joints
            joints_x_init=poseKeypoints[kk,0:n_joints,0]
            joints_y_init=poseKeypoints[kk,0:n_joints,1]   
            
            #HUMAN CENTROID CALCULATION
            n_joints_cent=0
            x_sum=0
            y_sum=0
            #no_zero=0 #number of joints which are not 0
            for k in range(0,n_joints):
                if joints_x_init[k]!=0 and joints_y_init[k]!=0:
                    #no_zero=no_zero+1
                    if k==0 or k==1 or k==2 or k==8 or k==5 or k==9 or k==12: #Only consider keypoints in the center of the body
                        x_sum=x_sum+joints_x_init[k]
                        y_sum=y_sum+joints_y_init[k]
                        n_joints_cent=n_joints_cent+1
            if n_joints_cent!=0:
                centroid[kk,0]=x_sum/n_joints_cent
                centroid[kk,1]=y_sum/n_joints_cent
                index_to_keep=index_to_keep+[kk]
            #elif no_zero!=0:
            #    centroid[kk,0]=sum(joints_x_init)/no_zero
            #    centroid[kk,1]=sum(joints_y_init)/no_zero
            #    index_to_keep=index_to_keep+[kk]
                
            for k in range(0,len(camera_id)):
                if centroid[k,0]<=self.image_size[1]: #camera front
                    camera_id[k]=0
                else:#camera back
                    camera_id[k]=1
        #return features,posture,orientation,distance,centroid,camera_id
        if index_to_keep!=[]:
            self.centroid=centroid[np.array(index_to_keep)]
            self.camera_id=camera_id[np.array(index_to_keep)]
            self.n_human=len(camera_id)
        else:
            self.n_human=0
################################################################################################################            
    
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node
    human=human_class()  
    rospy.init_node('human_detector_thermal',anonymous=True)
    # Setup and call subscription
    #Camara front
    thermal_front_sub=message_filters.Subscriber('/flir_module_driver/thermal/image_raw', Image)
    rgb_front_sub = message_filters.Subscriber('camera/color/image_raw', Image)
    depth_front_sub = message_filters.Subscriber('camera/aligned_depth_to_color/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([thermal_front_sub,rgb_front_sub, depth_front_sub], 1, 0.01)
    ts.registerCallback(human.image_callback)
    
    #rospy.Subscriber('/flir_module_driver/thermal/temp_meas_range', TempMeasRange,human.param_callback)
    #rospy.spin()
    #Rate setup
    rate = rospy.Rate(1/0.01) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        if visualization==True:
            print("MAIN")
            n_human=human.n_human
            centroid=human.centroid
            image=human.image
            print("N_human",n_human)
            print("Centroid",centroid)
            if n_human>0:
                for i in range(0,n_human):
                    center_coordinates = (int(centroid[i,0]), int(centroid[i,1]))                        
                    #print(center_coordinates)              
                    image = cv2.circle(image, center_coordinates, 5, (0, 255,0), 10)
            cv2.imshow("Thermal clustering",image)
            cv2.waitKey(10)  
        