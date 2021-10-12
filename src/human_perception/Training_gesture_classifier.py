# From Python
#!/usr/bin/env python

import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
from numpy import linalg as LA
from math import * 
#Import package to export model
import joblib
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
import rosbag
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import yaml

#Importing global parameters from .yaml file
src_direct=os.getcwd()
config_direct=src_direct[0:len(src_direct)-20]+"config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
direct_param=list(dict.items(parsed_yaml_file["directories_config"]))
ar_param=list(dict.items(parsed_yaml_file["action_recog_config"]))

#Importing RF model for posture recognition
posture_classifier_model=direct_param[2][1]
model_rf = joblib.load(posture_classifier_model)   
#Openpose initialization 
open_pose_python=direct_param[0][1]
open_pose_models=direct_param[1][1]
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
#Initializating cv_bridge
bridge = CvBridge()
#Feature extraction variables
n_joints=ar_param[0][1]
n_features=ar_param[1][1]
labels=ar_param[2][1]
n_labels=len(labels)

mode=0 # it can be 0 for training or 1 for testing 
dist=[0]*n_joints
angles=[0]*(n_joints-2) #17 angles
keypoints=np.zeros([25,3])
#joints_z_init=np.zeros([n_joints,1])
joints_z_init=[0]*n_joints

#Vectors between joints
v1_2=[0,0,0]
v2_3=[0,0,0]
v3_4=[0,0,0]
v1_5=[0,0,0]
v5_6=[0,0,0]
v6_7=[0,0,0]
v1_0=[0,0,0]
v0_15=[0,0,0]
v15_17=[0,0,0]
v0_16=[0,0,0]
v16_18=[0,0,0]
v1_8=[0,0,0]
v8_9=[0,0,0]
v9_10=[0,0,0]
v10_11=[0,0,0]
v8_12=[0,0,0]
v12_13=[0,0,0]
v13_14=[0,0,0]
X=np.zeros([1,len(dist)+len(angles)])#.flatten()
Y=np.zeros([1,1]).flatten()
topic_list=['/camera/camera1/color/image_raw','/camera/camera1/aligned_depth_to_color/image_raw']

##############################################################################
i=0
new_data=[0,0]
for ii in range(0,n_labels):
    if mode==1: #For testing
        folder_name=direct_param[4][1]
    else: # For training
        folder_name=direct_param[3][1]+labels[ii]+"/"   
    files = os.listdir(folder_name)
    for file in files:
        bag = rosbag.Bag(folder_name+file)
        for topic, msg, t in bag.read_messages(topics=topic_list):
            #print(topic)
            if topic==topic_list[0]:
                
                new_data[0]=1
                color_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                #color_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                #cv2.imshow('image',color_image)
                #key = cv2.waitKey(15)
                #print(depth_array.shape)
                #print(depth_array.shape)
            if topic==topic_list[1]:
                new_data[1]=1
                depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                #depth_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                depth_array = np.array(depth_image, dtype=np.float32)/1000
                #print ('center depth:', depth_array[center_idx[0], center_idx[1]])

            if new_data==[1,1] and len(depth_array[0])==len(color_image[0]) and len(depth_array[1])==len(color_image[1]):
                print(i)
                new_data=[0,0]
                #Process and display images
                datum = op.Datum()
                datum.cvInputData = color_image
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                ##################################################################
                ###################################################################
                    
                data=datum.poseKeypoints
                if data is not None:
                    n_human=len(data[:,0,0])
                    if n_human==1:
                       keypoints=datum.poseKeypoints[0,:]  
                    else:
                       skeleton_size=[0] * n_human
                       skeleton_position=[0] * n_human
                       for j in range(0,n_human):
                           neck_node=datum.poseKeypoints[j,1,:]
                           hip_node=datum.poseKeypoints[j,8,:]
                           skeleton_size[j]=sqrt((neck_node[0]-hip_node[0])**2+(neck_node[1]-hip_node[1])**2)
                           skeleton_position[j]=abs(neck_node[0]-420) #assuming images with 848 pixels in x axis
                       biggest=max(skeleton_size)
                       biggest_index=skeleton_size.index(biggest)
                       center=min(skeleton_position)
                       center_index=skeleton_position.index(center)
                       size_ordered=sorted(skeleton_size);
                       # When there is only a big skeleton
                       if (size_ordered[-2]/size_ordered[-1])<0.5:
                           keypoints=datum.poseKeypoints[biggest_index,:,:]
                       else: # When there are two or more big skeletons
                            if center_index==biggest_index:
                                keypoints=datum.poseKeypoints[biggest_index,:,:]
                            else:
                                keypoints=datum.poseKeypoints[center_index,:,:]        
                    
                    ### Feature extraction####
                    #Using only the important joints
                    joints_x_init=keypoints[0:n_joints,0]
                    joints_y_init=keypoints[0:n_joints,1]   
                    #print(joints_x_init[0])
                    #print(joints_y_init[0])
                    #print(joints_z_init[0])
                    #print(max(joints_x_init))
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
                   
                    features=dist+angles;  
                    
                    if mode==1: #TESTING
                        #HUMAN POSTURE RECOGNITION
                        X=np.array(features).transpose()
                        Y=model_rf.predict([X])
                        # Model Accuracy, how often is the classifier correct?
                        print("RESULT:",i,labels[int(Y)])
                    else: #TRAINING
                    
                        X_new=np.array(features)
                        X=np.concatenate((X, [X_new]), axis=0)
                        Y_new=ii                
                        Y=np.concatenate((Y, [Y_new]), axis=0)
                        print("RESULT:",i,labels[int(Y_new)])
                    i=i+1
                    ######################################################################
                    ######################################################################        
                    
                    cv2.imshow("OpenPose Python API", datum.cvOutputData)
                    key = cv2.waitKey(15)
                    #if key == 27: break
            #if i==5 or i==10 or i==15 or i==20:
            #    break
        bag.close()
if mode==0:
    #Training RF model        
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(np.array(X),Y)
    #Exporting the model
    joblib.dump(clf, config_direct+"/classifier_model_3D_v2.joblib")    
    

