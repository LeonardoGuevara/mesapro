#! /usr/bin/python3

#required packages
import sys
import os
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from numpy import linalg as LA
import joblib
import cv2
import yaml
import ros_numpy
import sensor_msgs.msg
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
import rosbag

##########################################################################################
#GLOBAL CONFIG FILE DIRECTORY
config_direct="/home/leo/rasberry_ws/src/mesapro/config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
#FEATURE EXTRACTION PARAMETERS
n_joints=parsed_yaml_file.get("action_recog_config").get("n_joints") #19 body joints from openpose output (25 available)
n_features=parsed_yaml_file.get("action_recog_config").get("n_features") #19 distances + 17 angles = 36 features
dist=[0]*n_joints
angles=[0]*(n_joints-2) #17 angles
joints_min=7 #minimum number of joints to consider a detection
labels=parsed_yaml_file.get("action_recog_config").get("posture_labels") #labels used for training the posture recognition model
n_labels=len(labels)
performance="high" #OpenPose performance, can be "normal" or "high", "normal" as initial condition
##OPENPOSE INITIALIZATION 
openpose_python=parsed_yaml_file.get("directories_config").get("open_pose_python")
openpose_models=parsed_yaml_file.get("directories_config").get("open_pose_models")
try:
    sys.path.append(openpose_python);
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
params = dict()
params["model_folder"] = openpose_models
if performance=="normal":
    params["net_resolution"] = "-1x368" #the detection performance and the GPU usage depends on this parameter, has to be numbers multiple of 16, the default is "-1x368", and the fastest performance is "-1x160"
else:
    params["net_resolution"] = "-1x432" #High performance
#params["maximize_positives"] = True
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
#VISUALIZATION PARAMETERS
openpose_visual=True  #to show or not a window with the human detection delivered by openpose
#TRAINING PARAMETERS
mode=0 # it can be 0 for training or 1 for testing 
topic_list=['/camera/camera1/color/image_raw','/camera1/color/image_raw','/camera/camera1/aligned_depth_to_color/image_raw','/camera1/aligned_depth_to_color/image_raw'] #name of topics (old and new) to be extracted from bag files
X=np.zeros([1,len(dist)+len(angles)])#.flatten()
Y=np.zeros([1,1]).flatten()
training_folder=parsed_yaml_file.get("directories_config").get("training_set") 
testing_folder=parsed_yaml_file.get("directories_config").get("testing_set") 
dist_threshold=7 #maximim distance (m) at which the human detection is considered for training/testing
##TESTING PARAMETERS
posture_classifier_model=n_features=parsed_yaml_file.get("directories_config").get("gesture_classifier_model") #Full name of the gesture_classifier_model to be tested
model_rf = joblib.load(posture_classifier_model)   
#########################################################################################################################
    
def feature_extraction_3D(poseKeypoints,depth_array):
    ### TO SELECT ONLY ONE HUMAN DETECTION (the one in the center and the biggest size)
    data=poseKeypoints
    if data is not None:
        n_human=len(data[:,0,0])
        if n_human==1:
           keypoints=data[0,:]  
        else:
           skeleton_size=[0] * n_human
           skeleton_position=[0] * n_human
           for j in range(0,n_human):
               neck_node=data[j,1,:]
               hip_node=data[j,8,:]
               if neck_node[0]*neck_node[1]*hip_node[0]*hip_node[1]!=0: #if neck and hip nodes exist
                   skeleton_size[j]=np.sqrt((neck_node[0]-hip_node[0])**2+(neck_node[1]-hip_node[1])**2)
                   skeleton_position[j]=abs(neck_node[0]-depth_array.shape[1]/2) #distance with respect to the image center
               else:
                   skeleton_size[j]=0
                   skeleton_position[j]=0
           biggest=max(skeleton_size)
           biggest_index=skeleton_size.index(biggest)
           center=min(skeleton_position)
           center_index=skeleton_position.index(center)
           size_ordered=sorted(skeleton_size);
           # When there is only a big skeleton
           if (size_ordered[-2]/size_ordered[-1])<0.5:
               keypoints=data[biggest_index,:,:]
           else: # When there are two or more big skeletons
                if center_index==biggest_index:
                    keypoints=data[biggest_index,:,:]
                else:
                    keypoints=data[center_index,:,:]       
        
        ### 3D Feature extraction####
        #Using only the important joints
        joints_x_init=keypoints[0:n_joints,0]
        joints_y_init=keypoints[0:n_joints,1]   
        joints_z_init=[0]*n_joints   
        for k in range(0,n_joints):
            #in case keypoints are out of image range 
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
        Js=np.sqrt(J_sum2/(n_joints))
        
        #only continue if there are enough joints detected on the skeleton
        if Js!=0 and valid>=joints_min:
            joints_x = joints_x_trans/Js      
            joints_y = joints_y_trans/Js
            joints_z = joints_z_trans/Js
               
            #Distances from each joint to the neck joint
            for k in range(0,n_joints):
               dist[k]=np.sqrt((joints_x[k]-joints_x[1])**2+(joints_y[k]-joints_y[1])**2+(joints_z[k]-joints_z[1])**2)  
        
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
            
            features=dist+angles  
            n_human=1
            #HUMAN CENTROID AND DISTANCE CALCULATION
            n_joints_cent=0
            x_sum=0
            y_sum=0     
            dist_sum=0
            centroid=[0,0]
            for k in range(0,n_joints):
                if joints_x_init[k]!=0 and joints_y_init[k]!=0 and joints_z_init[k]!=0:
                    if k==0 or k==1 or k==2 or k==8 or k==5 or k==9 or k==12: #Only consider keypoints in the center of the body
                        dist_sum=joints_z_init[k]+dist_sum
                        x_sum=x_sum+joints_x_init[k]
                        y_sum=y_sum+joints_y_init[k]
                        n_joints_cent=n_joints_cent+1
            #Only continue if there is at least 1 joint with x*y*z!=0 in the center of the body
            if n_joints_cent!=0:
                distance=dist_sum/n_joints_cent
                if distance<dist_threshold:
                    if joints_x_init[1]!=0 and joints_y_init[1]!=0 and joints_z_init[1]!=0: #If neck joint exists, then this is choosen as centroid
                        centroid[0]=joints_x_init[1]
                        centroid[1]=joints_y_init[1]
                    else: #the centroid is an average
                        centroid[0]=x_sum/n_joints_cent
                        centroid[1]=y_sum/n_joints_cent
                else:#not valid skeleton detection
                    n_human=0
                    features=[]
                    centroid=[]     
            else:#not valid skeleton detection
                n_human=0
                features=[]
                centroid=[] 
        else: #not valid skeleton detection
            n_human=0
            features=[]
            centroid=[]
    else: #none human detected
        n_human=0
        features=[]
        centroid=[]
    return features,n_human,centroid    

##############################################################################################################
    
counter=0
new_data=[0,0]
for k in range(0,n_labels):
    if mode==1: #For testing
        folder_name=testing_folder
    else: # For training
        folder_name=training_folder+labels[k]+"/"   
    files = os.listdir(folder_name)
    for file in files:
        bag = rosbag.Bag(folder_name+file)
        for topic, msg, t in bag.read_messages(topics=topic_list):
            #################################################################################################
            ## EXTRACTING MSG FROM BAG FILES
            #################################################################################################
            if topic==topic_list[0] or topic==topic_list[1]: #RGB          
                new_data[0]=1
                msg.__class__ = sensor_msgs.msg._Image.Image # to fix problems with msg classes when using rosbag and ros_numpy
                bgr_image = ros_numpy.numpify(msg) #replacing cv_bridge
                rgb_image = bgr_image[...,[2,1,0]].copy() #from bgr to rgb
                if topic==topic_list[0]: #this image don't need to be rotated
                    image_rotation=0
                else: #topic==topic_list[1] this image needs to be rotated
                    image_rotation=270
                if image_rotation==90:
                    img_rgb_rot=cv2.rotate(rgb_image,cv2.ROTATE_90_CLOCKWISE)
                elif image_rotation==270:
                    img_rgb_rot=cv2.rotate(rgb_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
                else: #0 degrees
                    img_rgb_rot=rgb_image  
                color_image=img_rgb_rot
            if topic==topic_list[2] or topic==topic_list[3]: #Depth
                new_data[1]=1
                msg.__class__ = sensor_msgs.msg._Image.Image # to fix problems with msg classes when using rosbag and ros_numpy
                depth_image = ros_numpy.numpify(msg) #replacing cv_bridge
                depth_array = np.array(depth_image, dtype=np.float32)/1000
                if topic==topic_list[2]: #this image don't need to be rotated
                    image_rotation=0
                else: #topic==topic_list[3] this image needs to be rotated
                    image_rotation=270
                if image_rotation==90:
                    img_d_rot=cv2.rotate(depth_array,cv2.ROTATE_90_CLOCKWISE)
                elif image_rotation==270:
                    img_d_rot=cv2.rotate(depth_array,cv2.ROTATE_90_COUNTERCLOCKWISE)
                else: #0 degrees
                    img_d_rot=depth_array
                depth_array=img_d_rot
            #################################################################################################
            ## OPENPOSE DETECTION
            #################################################################################################
            if new_data==[1,1] and len(depth_array[0])==len(color_image[0]) and len(depth_array[1])==len(color_image[1]):
                new_data=[0,0]                
                ####################################################################################################
                #Keypoints extraction using OpenPose
                datum.cvInputData = color_image
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                #Feature extraction
                [features,n_human,centroid]=feature_extraction_3D(datum.poseKeypoints,depth_array)
                if n_human==1: #execute only if there is a valid detection                                      
                    if mode==1: #TESTING
                        #HUMAN POSTURE RECOGNITION
                        X=np.array(features).transpose()
                        Y=model_rf.predict([X])
                        # Model Accuracy, how often is the classifier correct?
                        print("RESULT:",counter,labels[int(Y)])
                    else: #TRAINING                  
                        X_new=np.array(features)
                        X=np.concatenate((X, [X_new]), axis=0)
                        Y_new=k              
                        Y=np.concatenate((Y, [Y_new]), axis=0)
                        print("RESULT:",counter,labels[int(Y_new)])
                    counter=counter+1
                else:
                    print("Not valid OpenPose detection")
                    ######################################################################
                    ######################################################################        
                #Visualization
                image=datum.cvOutputData
                if centroid!=[]:
                    center_coordinates = (int(centroid[0]), int(centroid[1])) 
                    image = cv2.circle(image, center_coordinates, 5, (255, 0, 0), 20) #BLUE   
                cv2.imshow("OpenPose Python API",image)
                cv2.waitKey(10)
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
    joblib.dump(clf, config_direct+"/classifier_model_3D_v4.joblib")    
    


