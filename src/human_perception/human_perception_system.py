#! /usr/bin/python

#required packages
import rospy #tf
import geometry_msgs.msg
from geometry_msgs.msg import PoseArray
import message_filters #to sync the messages
#from sklearn.ensemble import RandomForestClassifier
from sensor_msgs.msg import Image
import sys
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from numpy import linalg as LA
from cv_bridge import CvBridge, CvBridgeError
import joblib
#import os
import cv2
#import yaml
import time
from mesapro.msg import human_msg
##########################################################################################

##Importing RF model for posture recognition
posture_classifier_model="/home/leo/rasberry_ws/src/mesapro/config/classifier_model_3D_v2.joblib"
model_rf = joblib.load(posture_classifier_model)   
##Openpose initialization 
open_pose_python='/home/leo/rasberry_ws/src/mesapro/openpose/build/python'
open_pose_models="/home/leo/rasberry_ws/src/mesapro/openpose/models"
try:
    sys.path.append(open_pose_python);
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
params = dict()
params["model_folder"] = open_pose_models
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
#Initializating cv_bridge
bridge = CvBridge()
#Setup ROS publiser
pub = rospy.Publisher('human_info', human_msg)
msg = human_msg()
pub_hz=0.01 #publising rate in seconds
#Feature extraction variables
n_joints=19
n_features=36
#Parameters for Matching
areas_percent=[0,0.25,0.41,0.58,0.75,1] #in pixels percent from area 0 to 4 (left to right), reused for areas 5 to 9
areas_angle_front=[90,40,10,0,-10,-40,-90] #in degrees from area 0 to 4 (left to right)
areas_angle_back=[-90,-140,-170,-180,180,170,140,90] #in degrees from area 5 to 9 (left to right)
meter_threshold=1.2 #meters
pixel_threshold=100 #pixels
tracking_threshold=3 #times a data is received
w_distance=[0.2,0.8] #weights used for calculating a distance weighted average during matching old with new data
#Parameters for Motion inference
n_samples=10 #number of samples used for the motion inference
speed_threshold=0.6  # < means static, > means slow motion 
#Paremeters for human Tracking
image_width=840 #initial value in pixels, it will be updated when the first image is taken from the camera
threshold_no_data=2 #seconds needed to remove a human from tracking list
posture_threshold=0.7 #minimum probability from openpose to start tracking new human using camera info
#General purposes variables
main_counter=0
visualization='off' #to show or not a window with the human detection on the photos
new_data=[0,0]     #flag to know if a new data from LiDAR or Camera is available, first element is for LiDAR, second for Camera
time_data=[0,0]    #list with the time in which data was taken from Lidar or camera
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        self.speed=0 #absolute value of the robot linear velocity
    
    def robot_callback(self,rob):
        self.speed=abs(rob.linear.x)

class human_class:
    def __init__(self): #It is done only the first iteration
        #Variables to store temporally the info of human detectection at each time instant
        self.n_human=1 # considering up to 1 human to track initially
        self.position=np.zeros([self.n_human,2]) #[x,y] from lidar local frame
        self.posture=np.zeros([self.n_human,2]) #from camera [posture_label,posture_probability]
        self.centroid=np.zeros([self.n_human,2]) #x,y (pixels) of the human centroid, from camera
        self.features=np.zeros([self.n_human,n_features]) #distances and angles of each skeleton, from camera
        self.orientation=np.zeros([self.n_human,1]) # it can be "front" or "back" if the human is facing the robot or not , from camera
        self.distance=np.zeros([self.n_human,1])  # distance between the robot and the average of the skeleton joints distances taken from the depth image, from camera
        
        #Variables to store the info of relevant humans tracked along the time
        self.position_track=np.zeros([self.n_human,2])
        self.posture_track=np.zeros([self.n_human,2])
        self.centroid_track=np.zeros([self.n_human,2])
        self.features_track=np.zeros([self.n_human,n_features])
        self.orientation_track=np.zeros([self.n_human,1])
        self.distance_track=np.zeros([self.n_human,1]) 
        self.sensor=np.zeros([self.n_human,1]) # it can be 0 if the data is from camera and Lidar, 1 if the data is  only from the LiDAR or 2 if the data is only from de camera
        self.motion_track=np.zeros([self.n_human,1]) #from lidar + camara
        self.time_track=np.ones([self.n_human,2])*(time.time()-time_init) #Vector with the time when a new data is collected from lidar or camera
        self.speed_track=np.zeros([self.n_human,1]) #human speed 
        self.speed_buffer=np.zeros([self.n_human,n_samples]) #buffer with the human speed recorded during n_points
        self.counter_motion=np.zeros([self.n_human,1]) # vector with the number of samples that have been recorded for motion inference
        self.area=np.zeros([self.n_human,1]) # vector with the number of the area of the image in which the human was detected, it can be 0-3, 0 is the left area of the image and 3 is the right area. HUman captired by Lidar has to be mapped to this areas too.
        self.counter_old=np.ones([self.n_human,2]) # vector with the number of times the human tracked has not been uptated with data from lidar or camera, counter>0 means is not longer detected, counter<=0 means is being detected 
        self.image=np.zeros((800,400,3), np.uint8) #initial value
        
    def lidar_callback(self,legs):
        print("DATA FROM LIDAR")
        pos=self.position
        if new_data[0]==0: #to read a new data only if the previous data was already used
            if legs.poses is None: #if there is no human legs detected
                print('No human detected from lidar')
                #self.counter_old=self.counter_old+1
            else:
                k=0
                for pose in legs.poses:
                    if k<len(pos):
                        ##########################################################################################################
                        #THIS DISTANCE NEEDS TO BE CALIBRATED ACCORDING TO THE LIDAR LOCATION RESPECT TO THE FRONT CAMERA POSITION (The camera location is the reference origin)
                        pos[k,0] = pose.position.x -0.55 #+6
                        pos[k,1] = pose.position.y +0.45 #+10
                        ############################################################################################################
                    else: #if there are more human detected than before
                        position_new=np.array([pose.position.x,pose.position.y])
                        pos=np.append(pos,[position_new],axis=0)
                    k=k+1
                if k<len(pos): #if there are less human detected than before
                    pos=pos[0:k,:]    
                self.position=pos
                new_data[0]=1 # update flag for new data
                time_data[0]=time.time()-time_init # update time when the last data was received
        
    def camera_callback(self,image_front, depth_front):
        print("DATA FROM CAMERA")
        global image_width
        if new_data[1]==0: #to read a new data only if the previous data was already used
            try:
                color_image = bridge.imgmsg_to_cv2(image_front, "bgr8")
                depth_image = bridge.imgmsg_to_cv2(depth_front, "passthrough")
                depth_array = np.array(depth_image, dtype=np.float32)/1000
                image_width = depth_array.shape[1]
                ##################################################################################
                #Here the images from two cameras has to be merged in a single image
                #######################################################################################
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
    
            datum.cvInputData = color_image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            #Keypoints extraction using OpenPose
            data=datum.poseKeypoints
            if data is None: #if there is no human skeleton detected
                print('No human detected from camera')
            else: #if there is at least 1 human skeleton detected
                #Feature extraction
                [self.features,self.posture,self.orientation,self.distance,self.centroid]=self.feature_extraction_3D(data,depth_array,n_joints,n_features)
                #print("HUMAN DISTANCE",list(self.distance))
                
                self.image=datum.cvOutputData               
            new_data[1]=1 # update flag for new data
            time_data[1]=time.time()-time_init # update time when the last data was received
                


################################################################################################################            
    def feature_extraction_3D(self,poseKeypoints,depth_array,n_joints,n_features):
        if len(self.posture)!=len(poseKeypoints[:,0,0]): #to readjust the vectors length in case there are more/less humans than expected
            posture=np.zeros([len(poseKeypoints[:,0,0]),2])
            centroid=np.zeros([len(poseKeypoints[:,0,0]),2]) 
            features=np.zeros([len(poseKeypoints[:,0,0]),n_features]) 
            orientation=np.zeros([len(poseKeypoints[:,0,0]),1])
            distance=np.zeros([len(poseKeypoints[:,0,0]),1]) 
        else:
            features=self.features
            posture=self.posture
            orientation=self.orientation
            distance=self.distance
            centroid=self.centroid
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
            for k in range(0,n_joints):
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
            for k in range(0,n_joints):  
                J_sum2=joints_x_trans[k]**2+joints_y_trans[k]**2+joints_z_trans[k]**2+J_sum2    
            Js=sqrt(J_sum2/(n_joints))
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
            for ii in range(prob.shape[1]): #depends of the number of gestures to classified
                if prob[0,ii]>=prob_max:
                    prob_max=prob[0,ii]
            posture[kk,1]=prob_max 
            #HUMAN DISTANCE AND CENTROID CALCULATION
            n_joints_dist=0
            n_joints_cent=0
            dist_average=0
            x_average=0
            y_average=0
            for k in range(0,n_joints):
                if joints_z_init[k]!=0:
                    dist_average=joints_z_init[k]+dist_average
                    n_joints_dist=n_joints_dist+1
                if joints_x_init[k]!=0 and joints_y_init[k]!=0:       
                    x_average=x_average+joints_x_init[k]
                    y_average=y_average+joints_y_init[k]
                    n_joints_cent=n_joints_cent+1
            distance[kk,:]=dist_average/n_joints_dist
            centroid[kk,0]=x_average/n_joints_cent
            centroid[kk,1]=y_average/n_joints_cent
            
        return features, posture, orientation, distance , centroid
            
    def human_tracking(self):
        sensor=self.sensor
        n_human=self.n_human
        position=self.position_track
        posture=self.posture_track
        motion=self.motion_track
        speed_buffer=self.speed_buffer
        time_track=self.time_track
        speed=self.speed_track
        counter_motion=self.counter_motion
        centroid=self.centroid_track
        features=self.features_track
        orientation=self.orientation_track
        distance=self.distance_track
        area=self.area
        counter_old=self.counter_old
        position_new=self.position
        posture_new=self.posture
        centroid_new=self.centroid
        features_new=self.features
        orientation_new=self.orientation
        distance_new=self.distance
        rob_speed=robot.speed
        ##############################################################################################################################################
        #TO MERGE NEW HUMAN DETECTION WITH TRACKED LIST, OR TO ADD NEW DETECTION TO THE TRACKING LIST, OR TO REMOVE OLD DETECTIONS OR FALSE POSITIVES FROM THE LIST
        
        #####New LiDAR info#####################################################################################        
        if new_data[0]==1:
            print("NEW DATA LIDAR")
            error_threshold=meter_threshold #meters
            diff=np.zeros([n_human,len(position_new[:,0])]) #vector with the error between distances
            area_new=np.zeros([len(position_new[:,0]),1]) #vector with the area of the image where the new human is detected
            new_human_flag=np.zeros([len(position_new[:,0]),1]) #assuming all are new humans
            for k in range(0,n_human): 
                for kk in range(0,len(position_new[:,0])):
                    distance_lidar=sqrt((position_new[kk,0])**2+(position_new[kk,1])**2) 
                    diff[k,kk]=abs(distance[k,:]-distance_lidar)        
                counter_no_new_data=0 # counter to know if the k-th human tracked is not longer detected
                for kk in range(0,len(position_new[:,0])):
                    if new_human_flag[kk]==0: # Consider the kk-th new human only if it was not matched with another tracked human before
                        #Determining the area where the new human is detected
                        ##############################################################################################################################
                        #It depends how to interpret the X-Y frame used by the human leg detector
                        angle=atan2(position_new[kk,1],position_new[kk,0])#+(pi/2) # pi/2 is because the lidar XY-frame is 90 degrees rotated
                        #############################################################################################################################
                        if angle>pi: #  to keep the angle between [-180,+180]
                            angle=angle-2*pi
                        if angle<-pi:
                            angle=angle+2*pi
                        print("ANGLE",angle*180/pi)
                        #Front areas
                        if angle<=areas_angle_front[5]*(pi/180) and angle>=areas_angle_front[6]*(pi/180):
                            area_new[kk,0]=4
                        elif angle<=areas_angle_front[4]*(pi/180) and angle>areas_angle_front[5]*(pi/180):
                            area_new[kk,0]=3
                        elif angle<=areas_angle_front[3]*(pi/180) and angle>areas_angle_front[4]*(pi/180):
                            area_new[kk,0]=2
                        elif angle<=areas_angle_front[2]*(pi/180) and angle>areas_angle_front[3]*(pi/180):
                            area_new[kk,0]=2
                        elif angle<=areas_angle_front[1]*(pi/180) and angle>areas_angle_front[2]*(pi/180):
                            area_new[kk,0]=1
                        elif angle<=areas_angle_front[0]*(pi/180) and angle>areas_angle_front[1]*(pi/180):
                            area_new[kk,0]=0 
                        #Back areas
                        if angle<=areas_angle_back[6]*(pi/180) and angle>areas_angle_back[7]*(pi/180):
                            area_new[kk,0]=9
                        elif angle<=areas_angle_back[5]*(pi/180) and angle>areas_angle_back[6]*(pi/180):
                            area_new[kk,0]=8
                        elif angle<=areas_angle_back[4]*(pi/180) and angle>areas_angle_back[5]*(pi/180):
                            area_new[kk,0]=7
                        elif angle<=areas_angle_back[2]*(pi/180) and angle>=areas_angle_back[3]*(pi/180):
                            area_new[kk,0]=7
                        elif angle<=areas_angle_back[1]*(pi/180) and angle>areas_angle_back[2]*(pi/180):
                            area_new[kk,0]=6
                        elif angle<areas_angle_back[0]*(pi/180) and angle>areas_angle_back[1]*(pi/180):
                            area_new[kk,0]=5 
                        
                        ###############################################################################################################################
                        #Determine if a new data match with the k-th human tracked
                        if diff[k,kk]<error_threshold and abs(area[k,0]-area_new[kk,0])<=1: # if a new detection match with a previos detected in distance and area
                            new_index=kk
                            time_diff=time_track[k,0]-time_data[0]    
                            #Updating speed and time_track
                            dist_lidar=sqrt(position_new[new_index,0]**2+position_new[new_index,1]**2)
                            motion_diff=distance[k,:]-dist_lidar
                            speed[k,:]=abs(motion_diff/time_diff)-rob_speed
                            time_track[k,0]=time_data[0]
                            if counter_motion[k]<n_samples: #while the recorded data is less than n_points                
                                speed_buffer[k,int(counter_motion[k])]=speed[k,:]
                                counter_motion[k]=counter_motion[k]+1
                            else: #to removed old data and replace for newer data
                                speed_buffer[k,0:n_samples-1]=speed_buffer[k,1:n_samples]
                                speed_buffer[k,n_samples-1]=speed[k,:]
                            #Updating position, area, counter_old and distance
                            position[k,:]=position_new[new_index,:]
                            area[k,0]=area_new[new_index,0]
                            counter_old[k,0]=counter_old[k,0]-1
                            new_human_flag[new_index]=1 #it is not a new human
                            distance[k,:]=w_distance[0]*distance[k,:]+w_distance[1]*dist_lidar
                            #For Update the sensor    
                            if sensor[k]==2: #if before it was only from camera
                                sensor[k,:]=0 #now is from both
                                print('New human matches with tracked human, merged')
                                    
                        else:
                            counter_no_new_data=counter_no_new_data+1 #counter to know if the k-th human tracked does not have a new data
                    else:
                        counter_no_new_data=counter_no_new_data+1 #counter to know if the k-th human tracked does not have a new data
                if counter_no_new_data==len(position_new[:,0]): #If there is no a new detection of the k-th human
                    if sensor[k,:]!=2: # consider only if the data was not originally taken from the lidar or lidar+camera
                        if counter_old[k,0]<0: #if is not detected for one instant, then reset it to 0
                            counter_old[k,0]=0    
                        else:
                            counter_old[k,0]=counter_old[k,0]+1
            
            #To remove an old human tracked
            index_to_keep=[]
            for k in range(0,len(counter_old[:,0])):           
                if counter_old[k,0]<tracking_threshold: # if the counter is still < threshold 
                    index_to_keep=index_to_keep+[k]
                else: # if a human was not detected for longer than the specific threshold               
                    if sensor[k,:]==0:
                        if counter_old[k,1]>=tracking_threshold:
                            print('A human was removed')
                            n_human=n_human-1
                        else:
                            sensor[k,:]=2 #now is only from camera
                            counter_old[k,0]=0 #reset the lidar counter to 0
                            index_to_keep=index_to_keep+[k]
                            print('No more data from lidar, only camera is considered')
                    else: #sensor==1
                        print('A human was removed')
                        n_human=n_human-1
            if n_human>0:
                position=position[np.array(index_to_keep)]
                posture=posture[np.array(index_to_keep)]
                motion=motion[np.array(index_to_keep)]
                time_track=time_track[np.array(index_to_keep)]
                speed=speed[np.array(index_to_keep)]
                speed_buffer=speed_buffer[np.array(index_to_keep)]
                counter_motion=counter_motion[np.array(index_to_keep)]
                counter_old=counter_old[np.array(index_to_keep)]
                centroid=centroid[np.array(index_to_keep)]
                features=features[np.array(index_to_keep)]
                orientation=orientation[np.array(index_to_keep)]
                distance=distance[np.array(index_to_keep)]
                sensor=sensor[np.array(index_to_keep)]
                area=area[np.array(index_to_keep)]
            else:
                n_human=1
                position=np.zeros([n_human,2])
                posture=np.zeros([n_human,2])
                motion=np.zeros([n_human,1])
                time_track=np.ones([n_human,2])*(time.time()-time_init) #init with the current time
                speed=np.zeros([n_human,1])
                speed_buffer=np.zeros([n_human,n_samples])
                counter_motion=np.zeros([n_human,1])
                counter_old=np.ones([n_human,2]) #init in 1
                centroid=np.zeros([n_human,2])
                features=np.zeros([n_human,n_features])
                orientation=np.zeros([n_human,1])
                distance=np.zeros([n_human,1])
                sensor=np.zeros([n_human,1])
                area=np.zeros([n_human,1])
            
            #To include a new human to be tracked
            for k in range(0,len(new_human_flag)):
                if new_human_flag[k]==0:
                    print('New human tracked from the lidar')
                    n_human=n_human+1
                    #Human perception
                    position=np.append(position,[position_new[k,:]],axis=0)
                    posture=np.append(posture,np.zeros([1,2]),axis=0)
                    centroid=np.append(centroid,np.zeros([1,2]),axis=0)
                    features=np.append(features,np.zeros([1,n_features]),axis=0)
                    orientation=np.append(orientation,np.zeros([1,1]),axis=0)
                    distance=np.append(distance,np.zeros([1,1]),axis=0)
                    distance[n_human-1,0]=sqrt((position_new[k,0])**2+(position_new[k,1])**2) 
                    sensor=np.append(sensor,np.ones([1,1]),axis=0) #1 because it is a lidar type data
                    #Motion inference
                    motion=np.append(motion,np.zeros([1,1]),axis=0) #new human detection starts with motion label 0 = "not_defined"
                    time_track=np.append(time_track,np.ones([1,2])*time_data[0],axis=0)  
                    speed=np.append(speed,np.zeros([1,1]),axis=0) #initial speed is 0
                    speed_buffer=np.append(speed_buffer,np.zeros([1,n_samples]),axis=0) 
                    speed_buffer[n_human-1,0]=speed[k,0] # first data in the speed_buffer
                    counter_motion=np.append(counter_motion,np.ones([1,1]),axis=0) # first data was recorded
                    #Human Tracking
                    counter_old=np.append(counter_old,np.ones([1,2]),axis=0) #it begins in 1 to be sure it is not a false positive
                    area=np.append(area,[area_new[k]],axis=0)
        #####New camera info###########################################################################################################
        if new_data[1]==1: 
            print("NEW DATA CAMERA")
            diff=np.zeros([n_human,len(centroid_new[:,0])])
            area_new=np.zeros([len(centroid_new[:,0]),1]) #vector with the area of the image where the new human is detected
            new_human_flag=np.zeros([len(centroid_new[:,0]),1]) #assuming all are new humans
            error_threshold=meter_threshold #meters
            for k in range(0,n_human):
                for kk in range(0,len(distance_new[:,0])):
                    diff[k,kk]=abs(distance[k,:]-distance_new[kk,:])
                counter_no_new_data=0 # counter to know if the k-th human tracked is not longer detected
                for kk in range(0,len(centroid_new[:,0])):
                    if new_human_flag[kk]==0: # Consider the kk-th new human only if it was not matched with another tracked human before
                        #Determining the area where the new human is detected by each camera
                        #Front camera
                        if centroid_new[kk,0]>=areas_percent[4]*image_width and centroid_new[kk,0]<=areas_percent[5]*image_width:
                            area_new[kk,0]=4
                        elif centroid_new[kk,0]>=areas_percent[3]*image_width and centroid_new[kk,0]<=areas_percent[4]*image_width:
                            area_new[kk,0]=3
                        elif centroid_new[kk,0]>=areas_percent[2]*image_width and centroid_new[kk,0]<=areas_percent[3]*image_width:
                            area_new[kk,0]=2
                        elif centroid_new[kk,0]>=areas_percent[1]*image_width and centroid_new[kk,0]<=areas_percent[2]*image_width:
                            area_new[kk,0]=1
                        elif centroid_new[kk,0]>=areas_percent[0]*image_width and centroid_new[kk,0]<=areas_percent[1]*image_width:
                            area_new[kk,0]=0    
                        #Back camera
                        elif centroid_new[kk,0]>=image_width+areas_percent[4]*image_width and centroid_new[kk,0]<=image_width+areas_percent[5]*image_width:
                            area_new[kk,0]=9
                        elif centroid_new[kk,0]>=image_width+areas_percent[3]*image_width and centroid_new[kk,0]<=image_width+areas_percent[4]*image_width:
                            area_new[kk,0]=8
                        elif centroid_new[kk,0]>=image_width+areas_percent[2]*image_width and centroid_new[kk,0]<=image_width+areas_percent[3]*image_width:
                            area_new[kk,0]=7
                        elif centroid_new[kk,0]>=image_width+areas_percent[1]*image_width and centroid_new[kk,0]<=image_width+areas_percent[2]*image_width:
                            area_new[kk,0]=6
                        elif centroid_new[kk,0]>=image_width+areas_percent[0]*image_width and centroid_new[kk,0]<=image_width+areas_percent[1]*image_width:
                            area_new[kk,0]=5 
                            
                        if diff[k,kk]<error_threshold and abs(area[k,0]-area_new[kk,0])<=1: # if a new detection match with a previos detected in distance and area
                            new_index=kk
                            time_diff=time_track[k,1]-time_data[1]                        
                            #Updating speed and time_track
                            motion_diff=distance[k,:]-distance_new[new_index,:]
                            speed[k,:]=abs(motion_diff/time_diff)-rob_speed
                            time_track[k,1]=time_data[1]   
                            if counter_motion[k]<n_samples: #while the recorded data is less than n_points                
                                speed_buffer[k,int(counter_motion[k])]=speed[k,:]
                                counter_motion[k]=counter_motion[k]+1
                            else: #to removed old data and replace for newer data
                                speed_buffer[k,0:n_samples-1]=speed_buffer[k,1:n_samples]
                                speed_buffer[k,n_samples-1]=speed[k,:]
                            #Updating posture, centroid, features, orientation, distance, area, counter_old
                            posture[k,:]=posture_new[new_index,:]
                            centroid[k,:]=centroid_new[new_index,:]
                            features[k,:]=features_new[new_index,:]
                            orientation[k,:]=orientation_new[new_index,:]
                            distance[k,:]=w_distance[0]*distance[k,:]+w_distance[1]*distance_new[new_index,:]
                            area[k,0]=area_new[new_index,0]
                            counter_old[k,1]=counter_old[k,1]-1
                            new_human_flag[new_index]=1 #it is not a new human
                            #For motion inference and Update the sensor    
                            if sensor[k]==1: #if before it was only from lidar
                                sensor[k,:]=0 #now is from both
                                print('New human matches with tracked human, merged')
                            
                        else:
                            counter_no_new_data=counter_no_new_data+1 #counter to know if the k-th human tracked does not have a new data
                    else:
                        counter_no_new_data=counter_no_new_data+1 #counter to know if the k-th human tracked does not have a new data
                
                if counter_no_new_data==len(centroid_new[:,0]): #If there is no a new detection of the k-th human
                    if sensor[k,:]!=1: #consider only if the data was originally taken from the camera or camara+lidar
                        if counter_old[k,1]<0: #if is not detected for one instant, then reset it to 0
                            counter_old[k,1]=0    
                        else:
                            counter_old[k,1]=counter_old[k,1]+1
                    
            #To remove an old human tracked
            index_to_keep=[]
            for k in range(0,len(counter_old[:,0])):           
                if counter_old[k,1]<tracking_threshold: # if the counter is still < threshold 
                    index_to_keep=index_to_keep+[k]
                else: # if a human was not detected for longer than the specific threshold               
                    if sensor[k,:]==0:
                        if counter_old[k,0]>=tracking_threshold:
                            print('A human was removed')
                            n_human=n_human-1
                        else:
                            sensor[k,:]=1 #now is only from lidar
                            counter_old[k,1]=0 #reset the camera counter to 0
                            index_to_keep=index_to_keep+[k]
                            print('No more data from camera, only lidar is considered')
                    else: #sensor==1
                        print('A human was removed')
                        n_human=n_human-1
            if n_human>0:
                position=position[np.array(index_to_keep)]
                posture=posture[np.array(index_to_keep)]
                motion=motion[np.array(index_to_keep)]
                time_track=time_track[np.array(index_to_keep)]
                speed=speed[np.array(index_to_keep)]
                speed_buffer=speed_buffer[np.array(index_to_keep)]
                counter_motion=counter_motion[np.array(index_to_keep)]
                counter_old=counter_old[np.array(index_to_keep)]
                centroid=centroid[np.array(index_to_keep)]
                features=features[np.array(index_to_keep)]
                orientation=orientation[np.array(index_to_keep)]
                distance=distance[np.array(index_to_keep)]
                sensor=sensor[np.array(index_to_keep)]
                area=area[np.array(index_to_keep)]
            else:
                n_human=1
                position=np.zeros([n_human,2])
                posture=np.zeros([n_human,2])
                motion=np.zeros([n_human,1])
                time_track=np.ones([n_human,2])*(time.time()-time_init) #init with the current time
                speed=np.zeros([n_human,1])
                speed_buffer=np.zeros([n_human,n_samples])
                counter_motion=np.zeros([n_human,1])
                counter_old=np.ones([n_human,2]) #init in 1
                centroid=np.zeros([n_human,2])
                features=np.zeros([n_human,n_features])
                orientation=np.zeros([n_human,1])
                distance=np.zeros([n_human,1])
                sensor=np.zeros([n_human,1])
                area=np.zeros([n_human,1])
                
            #To include a new human to be tracked
            for k in range(0,len(new_human_flag)):
                if new_human_flag[k]==0 and posture_new[k,1]>=posture_threshold: # only if the openpose probability is reliable 
                    print('New human tracked from the camera')
                    n_human=n_human+1
                    #Human perception
                    position=np.append(position,np.zeros([1,2]),axis=0)
                    posture=np.append(posture,[posture_new[k,:]],axis=0)
                    centroid=np.append(centroid,[centroid_new[k,:]],axis=0)
                    features=np.append(features,[features_new[k,:]],axis=0)
                    orientation=np.append(orientation,[orientation_new[k,:]],axis=0)
                    distance=np.append(distance,[distance_new[k,:]],axis=0)
                    sensor=np.append(sensor,2*np.ones([1,1]),axis=0) #2 because it is a camera type data
                    #Motion infenrece
                    motion=np.append(motion,np.zeros([1,1]),axis=0) #new human detection starts with motion label 0 = "not_defined"
                    time_track=np.append(time_track,np.ones([1,2])*time_data[1],axis=0) 
                    speed=np.append(speed,np.zeros([1,1]),axis=0) #initial speed is 0
                    speed_buffer=np.append(speed_buffer,np.zeros([1,n_samples]),axis=0) 
                    speed_buffer[n_human-1,0]=speed[k,0] # first data in the speed_buffer
                    counter_motion=np.append(counter_motion,np.ones([1,1]),axis=0) # first data was recorded
                    #Human tracking
                    counter_old=np.append(counter_old,np.ones([1,2]),axis=0) #it begins in 1 to be sure it is not a false positive       
                    area=np.append(area,[area_new[k]],axis=0)
        ############################################################################################################################
        #TO ENSURE THAT THERE ARE NOT REPEATED HUMANS TO BE TRACKED  
        if new_data[0]==1 or new_data[1]==1:
            repeated_index=np.zeros([n_human,1]) #initially all humans are considered different
            error_threshold=meter_threshold #meters            
            for k in range(0,n_human):
                dist_1=distance[k,:]     
                for i in range (0,n_human):
                    if k!=i and repeated_index[k]==0 and repeated_index[i]==0:
                        dist_2=distance[i,:]   
                        dist_diff=abs(dist_1-dist_2)  
                        area_diff=abs(area[k,0]-area[i,0])
                        if dist_diff<=error_threshold and area_diff<=1:
                            print('Repeated human in track_list, merged')
                            if min(counter_old[k,:])<=min(counter_old[i,:]):
                                repeated_index[i]=1
                            else:
                                repeated_index[k]=1
            if max(repeated_index)>0:   
                index_to_keep=[]                    
                for k in range(0,n_human):
                    if repeated_index[k]==0:
                        index_to_keep=index_to_keep+[k]
                position=position[np.array(index_to_keep)]
                posture=posture[np.array(index_to_keep)]
                motion=motion[np.array(index_to_keep)]
                time_track=time_track[np.array(index_to_keep)]
                speed=speed[np.array(index_to_keep)]
                speed_buffer=speed_buffer[np.array(index_to_keep)]
                counter_motion=counter_motion[np.array(index_to_keep)]
                counter_old=counter_old[np.array(index_to_keep)]
                centroid=centroid[np.array(index_to_keep)]
                features=features[np.array(index_to_keep)]
                orientation=orientation[np.array(index_to_keep)]
                distance=distance[np.array(index_to_keep)]
                sensor=sensor[np.array(index_to_keep)]
                area=area[np.array(index_to_keep)]          
                n_human=len(counter_old[:,0])
                
        ############################################################################################################################
        #TO REMOVED HUMAN TRACKED AFTER THE SENSORS WERE NOT PUBLISHING DATA FOR A LONG TIME  
        current_time=time.time()-time_init   
        for i in range(0,n_human):  
            if sensor[i]==2: #only camera
                if current_time-time_track[i,1]>=threshold_no_data:
                    counter_old[i,1]=tracking_threshold #to remove this human
            if sensor[i]==1: #only lidar
                if current_time-time_track[i,0]>=threshold_no_data:
                    counter_old[i,0]=tracking_threshold #to remove this human
            if sensor[i]==0: #data from camera+lidar
                if current_time-time_track[i,1]>=threshold_no_data:
                    counter_old[i,1]=tracking_threshold #to remove this human
                if current_time-time_track[i,0]>=threshold_no_data:
                    counter_old[i,0]=tracking_threshold #to remove this human 
        
        #To remove an old human tracked
        index_to_keep=[]
        for k in range(0,len(counter_old[:,0])):
            if sensor[k]==2: #From camera         
                if counter_old[k,1]<tracking_threshold: # if the counter is still < threshold 
                    index_to_keep=index_to_keep+[k]
                else: # if a human was not detected for longer than the specific threshold               
                    print('A human was removed')
                    n_human=n_human-1
            elif sensor[k]==0: #from camera+lidar
                if counter_old[k,0]>counter_old[k,1]: #if lidar data is older than camera data
                    if counter_old[k,0]<tracking_threshold: # if the counter is still < threshold 
                        index_to_keep=index_to_keep+[k]
                    else: # if a human was not detected for longer than the specific threshold
                        sensor[k,:]=2 #now is only from camera
                        counter_old[k,0]=0 #reset the lidar counter to 0
                        index_to_keep=index_to_keep+[k]
                        print('No more data from lidar, only camera is considered')
                else: #if camera data is older than lidar data
                    if counter_old[k,1]<tracking_threshold: # if the counter is still < threshold 
                        index_to_keep=index_to_keep+[k]
                    else: # if a human was not detected for longer than the specific threshold
                        sensor[k,:]=1 #now is only from lidar
                        counter_old[k,1]=0 #reset the camera counter to 0
                        index_to_keep=index_to_keep+[k]
                        print('No more data from camera, only lidar is considered')
            else: #from lidar
                if counter_old[k,0]<tracking_threshold: # if the counter is still < threshold 
                    index_to_keep=index_to_keep+[k]
                else: # if a human was not detected for longer than the specific threshold               
                    print('A human was removed')
                    n_human=n_human-1
        
        if n_human>0:
            position=position[np.array(index_to_keep)]
            posture=posture[np.array(index_to_keep)]
            motion=motion[np.array(index_to_keep)]
            time_track=time_track[np.array(index_to_keep)]
            speed=speed[np.array(index_to_keep)]
            speed_buffer=speed_buffer[np.array(index_to_keep)]
            counter_motion=counter_motion[np.array(index_to_keep)]
            counter_old=counter_old[np.array(index_to_keep)]
            centroid=centroid[np.array(index_to_keep)]
            features=features[np.array(index_to_keep)]
            orientation=orientation[np.array(index_to_keep)]
            distance=distance[np.array(index_to_keep)]
            sensor=sensor[np.array(index_to_keep)]
            area=area[np.array(index_to_keep)]
        else:
            n_human=1
            position=np.zeros([n_human,2])
            posture=np.zeros([n_human,2])
            motion=np.zeros([n_human,1])
            time_track=np.ones([n_human,2])*(time.time()-time_init) #init with the current time
            speed=np.zeros([n_human,1])
            speed_buffer=np.zeros([n_human,n_samples])
            counter_motion=np.zeros([n_human,1])
            counter_old=np.ones([n_human,2]) #init in 1
            centroid=np.zeros([n_human,2])
            features=np.zeros([n_human,n_features])
            orientation=np.zeros([n_human,1])
            distance=np.zeros([n_human,1])
            sensor=np.zeros([n_human,1])
            area=np.zeros([n_human,1])
            
        return sensor,n_human,position,posture,centroid,features,orientation,distance,motion,time_track,speed,speed_buffer,counter_motion,area,counter_old

    def human_motion_inference(self):
        n_human=self.n_human
        speed=self.speed_buffer
        motion=self.motion_track
        orientation=self.orientation_track
        posture=self.posture_track
        sensor=self.sensor
        for ii in range(0,n_human):
            speed_mean=np.mean(speed[ii,:])
            if abs(speed_mean)<speed_threshold: # if human is  mostly static
                motion[ii]=1
            else: #if human is moving
                motion[ii]=2
    
        return motion



###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node
    time_init=time.time()       
    human=human_class()  
    robot=robot_class()
    rospy.init_node('human_perception_system',anonymous=True)
    # Setup and call subscription
    rospy.Subscriber('/nav_vel',geometry_msgs.msg.Twist,robot.robot_callback) 
    image_front_sub = message_filters.Subscriber('camera/camera1/color/image_raw', Image)
    depth_front_sub = message_filters.Subscriber('camera/camera1/aligned_depth_to_color/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub], 1, 0.01)
    ts.registerCallback(human.camera_callback)
    rospy.Subscriber('/people_tracker/pose_array',PoseArray,human.lidar_callback) 
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        main_counter=main_counter+1
        #Human tracking
        [human.sensor,human.n_human,human.position_track,human.posture_track,human.centroid_track,human.features_track,human.orientation_track,human.distance_track,human.motion_track,human.time_track,human.speed_track,human.speed_buffer,human.counter_motion,human.area,human.counter_old]=human.human_tracking()
        #Human motion inference
        human.motion_track=human.human_motion_inference()
        if new_data[0]==1:
            new_data[0]=0
        if new_data[1]==1:
            new_data[1]=0

        if visualization=='on':
            cv2.imshow("Human tracking",human.image)
            cv2.waitKey(15)  
           
        #print(main_counter)    
        #Publish     
        msg.posture = list(human.posture_track[:,0])
        msg.posture_prob = list(human.posture_track[:,1])
        msg.motion = list(human.motion_track[:,0])
        msg.position_x = list(human.position_track[:,0])
        msg.position_y = list(human.position_track[:,1])
        msg.centroid_x =list(human.centroid_track[:,0])
        msg.centroid_y =list(human.centroid_track[:,1])
        msg.distance = list(human.distance_track[:,0])
        msg.orientation = list(human.orientation_track[:,0])
        msg.area = list(human.area[:,0])
        msg.sensor = list(human.sensor[:,0])
        msg.sensor_t0 = list(human.time_track[:,0])
        msg.sensor_t1 = list(human.time_track[:,1])
        msg.sensor_c0 = list(human.counter_old[:,0])
        msg.sensor_c1 = list(human.counter_old[:,1])
        pub.publish(msg)
        
        rate.sleep() #to keep fixed the publishing loop rate
        time_end=time.time()
        pub_time=time_end-time_init
        print("TIME",round(pub_time,2))
        