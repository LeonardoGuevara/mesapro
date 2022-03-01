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
from cv_bridge import CvBridge, CvBridgeError
import joblib
#import pickle
import time
import cv2
from mesapro.msg import human_detector_msg
##########################################################################################

##Importing RF model for posture recognition
posture_classifier_model=rospy.get_param("/hri_camera_detector/posture_classifier_model") #you have to change /hri_camera_detector/ if the node is not named like this
#posture_classifier_model="/home/leo/rasberry_ws/src/mesapro/config/classifier_model_3D_v2.joblib"
model_rf = joblib.load(posture_classifier_model)   
##Openpose initialization 
openpose_python=rospy.get_param("/hri_camera_detector/openpose_python") #you have to change /hri_camera_detector/ if the node is not named like this
openpose_models=rospy.get_param("/hri_camera_detector/openpose_models") #you have to change /hri_camera_detector/ if the node is not named like this
#openpose_python='/home/leo/rasberry_ws/src/openpose/build/python'
#openpose_models="/home/leo/rasberry_ws/src/openpose/models"
try:
    sys.path.append(openpose_python);
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
params = dict()
params["model_folder"] = openpose_models
params["net_resolution"] = "-1x256" #the detection performance and the GPU usage depends on this parameter, has to be numbers multiple of 16, the default is "-1x368", and the fastest performance is "-1x160"
#params["maximize_positives"] = True
#params["camera_resolution"]= "848x480"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
#Initializating cv_bridge
bridge = CvBridge()
#Setup ROS publiser
pub = rospy.Publisher('human_info_camera', human_detector_msg,queue_size=1) #small queue means priority to new data
msg = human_detector_msg()
#Feature extraction variables
n_joints=19
n_features=36
joints_min=7 #minimum number of joints to consider a detection
performance="normal" #OpenPose performance
time_threshold=3 #3 seconds as the minimum time between two consecute changes of openpose performance
time_change=0 #initial value
#General purposes variables
visualization=True #to show or not a window with the human detection on the photos
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
        self.image_size=[848,400] #initial condition of each camera
        self.camera_id=np.zeros([self.n_human,1]) #to know which camera detected the human
        self.image_show=np.zeros((848,400,3), np.uint8) #image used for visualization, initial value
        self.color_image=np.zeros((848,400,3), np.uint8) #rgb image
        self.depth_array=np.zeros((848,400,1), np.uint8) #depth map
        
    def camera_callback(self,image_front, depth_front):
        
        #print("DATA FROM CAMERA")
        #try:
        #Front camera info extraction
        color_image_front = bridge.imgmsg_to_cv2(image_front, "bgr8")
        #color_image_front=cv2.resize(color_image_front, (848,400))
        depth_image_front = bridge.imgmsg_to_cv2(depth_front, "passthrough")
        #depth_image_front=cv2.resize(depth_image_front, (848,400))
        depth_array_front = np.array(depth_image_front, dtype=np.float32)/1000
        self.image_size=depth_array_front.shape
        
        ##################################################################################
        #Back camera info extraction
        color_image_back=color_image_front
        depth_array_back=depth_array_front
        #Here the images from two cameras has to be merged in a single image (front image left, back image back)
        color_image=np.append(color_image_front,color_image_back,axis=1) 
        depth_array=np.append(depth_array_front,depth_array_back,axis=1) 
        #color_image=color_image_front
        #depth_array=depth_array_front
        #######################################################################################
        #except CvBridgeError as e:
        #    rospy.logerr("CvBridge Error: {0}".format(e))
        self.color_image=color_image
        self.depth_array=depth_array
        
    def processing(self,color_image,depth_array):
        global performance, time_change
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

        print("PERFORMANCE",performance)
        #print("MAX DIS",max(self.distance))
        print("MIN DIS",min(self.distance))
        
        datum.cvInputData = color_image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #Keypoints extraction using OpenPose
        keypoints=datum.poseKeypoints
        self.image_show=datum.cvOutputData
            
        if keypoints is None: #if there is no human skeleton detected
            print('No human detected')
            self.n_human=0
        else: #if there is at least 1 human skeleton detected
            #Feature extraction
            self.feature_extraction_3D(keypoints,depth_array,n_joints,n_features)
            print('Human detection')
            #Publish     
            if self.n_human>0:
                msg.posture = [int(x) for x in list(self.posture[:,0])] #to ensure publish int
                msg.posture_prob = list(self.posture[:,1])
                msg.centroid_x =list(self.centroid[:,0])
                msg.centroid_y =list(self.centroid[:,1])
                msg.distance = list(self.distance[:,0])
                msg.orientation = [int(x) for x in list(self.orientation[:,0])] #to ensure publish int
                msg.camera_id= [int(x) for x in list(self.camera_id[:,0])] #to ensure publish int
                msg.image_size= self.image_size
                pub.publish(msg)
            #cv2.imshow("System outputs",datum.cvOutputData)
            #cv2.waitKey(5)
           
    
################################################################################################################            
    def feature_extraction_3D(self,poseKeypoints,depth_array,n_joints,n_features):
        posture=np.zeros([len(poseKeypoints[:,0,0]),2])
        centroid=np.zeros([len(poseKeypoints[:,0,0]),2]) 
        features=np.zeros([len(poseKeypoints[:,0,0]),n_features]) 
        orientation=np.zeros([len(poseKeypoints[:,0,0]),1])
        distance=np.zeros([len(poseKeypoints[:,0,0]),1]) 
        camera_id=np.zeros([len(poseKeypoints[:,0,0]),1]) 
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
            #print("JOINT X",joints_x_init)
            #print("JOINT Y",joints_y_init)
            for k in range(0,n_joints):
                #in case keypoints are out of image range (necessary when two images were merged)
                if int(joints_y_init[k])>=len(depth_array[:,0]):
                    joints_y_init[k]=len(depth_array[:,0])-1
                if int(joints_x_init[k])>=len(depth_array[0,:]):
                    joints_x_init[k]=len(depth_array[0,:])-1
                joints_z_init[k]=depth_array[int(joints_y_init[k]),int(joints_x_init[k])]
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
            for k in range(0,n_joints):  
                J_sum2=joints_x_trans[k]**2+joints_y_trans[k]**2+joints_z_trans[k]**2+J_sum2  
                if joints_x_trans[k]!=0 and joints_y_trans[k]!=0 and joints_z_trans[k]!=0:
                    valid=valid+1
            Js=sqrt(J_sum2/(n_joints))
            
            if Js!=0 and valid>=joints_min: #only continue if there are enough joints detected on the skeleton
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
                #no_zero=0 #number of joints which are not 0
                for k in range(0,n_joints):
                    if joints_x_init[k]!=0 and joints_y_init[k]!=0 and joints_z_init[k]!=0:
                        #no_zero=no_zero+1
                        if k==0 or k==1 or k==2 or k==8 or k==5 or k==9 or k==12: #Only consider keypoints in the center of the body
                            dist_sum=joints_z_init[k]+dist_sum
                            x_sum=x_sum+joints_x_init[k]
                            y_sum=y_sum+joints_y_init[k]
                            n_joints_cent=n_joints_cent+1
                if n_joints_cent!=0:
                    distance[kk,:]=dist_sum/n_joints_cent
                    centroid[kk,0]=x_sum/n_joints_cent
                    centroid[kk,1]=y_sum/n_joints_cent
                    #Only continue if the human is not detected in between the two images merged
                    if centroid[kk,0]<=self.image_size[1]-self.image_size[1]*0.1 or centroid[kk,0]>=self.image_size[1]+self.image_size[1]*0.1: 
                        index_to_keep=index_to_keep+[kk]    
                        if centroid[kk,0]<=self.image_size[1]: #camera front
                            camera_id[kk]=0
                        else:#camera back
                            camera_id[kk]=1
                    
        #return features,posture,orientation,distance,centroid,camera_id
        if index_to_keep!=[]:
            self.features=features[np.array(index_to_keep)]
            self.posture=posture[np.array(index_to_keep)]
            self.orientation=orientation[np.array(index_to_keep)]
            self.distance=distance[np.array(index_to_keep)]
            self.centroid=centroid[np.array(index_to_keep)]
            self.camera_id=camera_id[np.array(index_to_keep)]
            self.n_human=len(distance)
        else:
            self.n_human=0

        
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node
    time_init=time.time() 
    human=human_class()  
    rospy.init_node('human_detector_camera',anonymous=True)
    # Setup and call subscription
    #Camara front
    image_front_sub = message_filters.Subscriber('camera/camera1/color/image_raw', Image) #old topic names
    depth_front_sub = message_filters.Subscriber('camera/camera1/aligned_depth_to_color/image_raw', Image) #old topic names
    ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub], 5, 1) #1,0.01
    ts.registerCallback(human.camera_callback)
    if visualization==False:
        rospy.spin()
    else:
        #Rate setup
        rate = rospy.Rate(1/0.01) # ROS publishing rate in Hz
        while not rospy.is_shutdown():	
            human.processing(human.color_image,human.depth_array)
            if visualization==True:
                centroids_x=human.centroid[:,0]
                centroids_y=human.centroid[:,1]
                color_image=human.image_show 
                if human.n_human>0:
                    for k in range(0,len(centroids_x)):    
                        center_coordinates = (int(centroids_x[k]), int(centroids_y[k])) 
                        color_image = cv2.circle(color_image, center_coordinates, 5, (255, 0, 0), 20) #BLUE           
    
                cv2.imshow("Human detector",color_image)
                cv2.waitKey(10)  
            