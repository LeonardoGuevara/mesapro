#! /usr/bin/python

#required packages
import rospy #tf
import geometry_msgs.msg
from geometry_msgs.msg import PoseArray
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import time
from mesapro.msg import human_msg, human_detector_msg, hri_msg 
import threading # Needed for Timer
##########################################################################################
#GENERAL PURPUSES VARIABLES
#Setup ROS publiser
pub = rospy.Publisher('human_info', human_msg,queue_size=1) #small queue means priority to new data
msg = human_msg()
#Parameters for area inference
row_width=1.3 #in meters
angle_area=60 # in degrees mesuared from the local x-axis robot frame
#Parameters for matching
meter_threshold=[0.5,1] # error in meters within two human detections are considered the same human, different for lidar than for camera -> [lidar,camera]
tracking_threshold=3 #times a detection has to be received in order to consider for tracking
w_data=[0.2,0.8] #weights used for calculating a weighted average during matching old with new data -> [old,new]
lidar_to_cam=[-0.35,0] #distances [x,y] in meters used to compensate the difference in position between the lidars and the cameras (The camera location is the reference origin)
#Parameters for Motion and Gesture inference
n_samples=8 #number of samples used for the motion inference
n_samples_gest=4 #number of samples used for the gesture inference, it has to be long enough to avoid recognize unnecesary gestures  (e.g. while rising arms, it can be detected as hands in the middle of the movement)
posture_threshold=0.5 #minimum probability delivered by gesture recognition algoritm to consider a gesture valid
speed_threshold=0.5  # < speed_threshold means static, > speed_threshold means slow motion 
#Paremeters for human Tracking
threshold_no_data=2 #seconds needed to remove an old human tracked from tracking list
#General purposes variables
main_counter=0
new_data=[0,0]     #flag to know if a new human detection from LiDAR or Camera is available, first element is for LiDAR, second for Camera
time_data=[0,0]    #list with the time in which new detection was taken from Lidar or camera
time_diff=[0,0]    #list with the time between two consecutive detections taken from lidar or camera
pub_hz=0.01 #main loop frequency
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        self.speed=0 #absolute value of the robot linear velocity
        self.action_mode="footpath" #it can be "footpath" or "polytunnel", by default is "footpath"
    
    def robot_speed_callback(self,rob):
        self.speed=abs(rob.linear.x)
 
    def robot_action_callback(self,safety_info):
        self.action_mode=safety_info.action_mode 
        
class human_class:
    def __init__(self): #It is done only the first iteration
        #General purposes variables
        self.n_human=1 # considering up to 1 human to track initially
        #Variables to store temporally the info of human detectection at each time instant
        self.position_lidar=np.zeros([self.n_human,2]) #[x,y] from lidar local frame
        self.posture=np.zeros([self.n_human,2]) #from camera [posture_label,posture_probability]
        self.centroid=np.zeros([self.n_human,2]) #x,y (pixels) of the human centroid, from camera
        self.orientation=np.zeros([self.n_human,1]) # it can be "front" or "back" if the human is facing the robot or not , from camera
        self.distance=np.zeros([self.n_human,1])  # distance between the robot and the average of the skeleton joints distances taken from the depth image (from camera local frame)
        self.position_cam=np.zeros([self.n_human,2]) #[x,y] from camera local frame
        
        #Variables to store the info of relevant humans tracked along the time
        self.position_track=np.zeros([self.n_human,2])
        self.posture_track=np.zeros([self.n_human,2])
        self.centroid_track=np.zeros([self.n_human,2])
        self.orientation_track=np.zeros([self.n_human,1])
        self.distance_track=np.zeros([self.n_human,1]) 
        self.sensor=np.zeros([self.n_human,1]) # it can be 0 if the data is from camera and Lidar, 1 if the data is  only from the LiDAR or 2 if the data is only from de camera
        self.motion_track=np.zeros([self.n_human,1]) #from lidar + camara
        self.time_track=np.ones([self.n_human,2])*(time.time()-time_init) #Vector with the time when the last data was collected from lidar or camera of each human tracked
        self.speed_track=np.zeros([self.n_human,1]) #human speed 
        self.speed_buffer=np.zeros([self.n_human,n_samples]) #buffer with the human speed recorded during n_points
        self.posture_buffer=np.zeros([self.n_human,n_samples_gest]) #buffer with the human gesture recorded during n_points
        self.posture_prob_buffer=np.zeros([self.n_human,n_samples_gest]) #buffer with the human gesture probability recorded during n_points
        self.counter_motion=np.zeros([self.n_human,1]) # vector with the number of samples that have been recorded for motion inference
        self.counter_posture=np.zeros([self.n_human,1]) #vector with the number of samples that have been recorded for gesture inference
        self.area=np.zeros([self.n_human,1]) # vector with the number of the area of the image in which the human was detected, it can be 0-3, 0 is the left area of the image and 3 is the right area. HUman captired by Lidar has to be mapped to this areas too.
        self.counter_old=np.ones([self.n_human,2]) # vector with the number of times the human tracked has not been uptated with data from lidar or camera, counter>0 means is not longer detected, counter<=0 means is being detected 
        #Thermal detection
        self.thermal_detection=False #assuming no thermal detection as initial value
        # Safety timers (only for camera messages because leg detector is not publishing continuously)
        self.cam_msg=True                       #flag to know if camera is publishing messages, by default is "True"
        self.time_without_msg=rospy.get_param("/hri_perception/time_without_msg",5) # Maximum time without receiving sensors messages 
        self.timer_safety_cam = threading.Timer(self.time_without_msg,self.safety_timeout_cam) # If "n" seconds elapse, call safety_timeout() for camera
        self.timer_safety_cam.start()
        
    def lidar_callback(self,legs):
        #print("DATA FROM LIDAR")
        if new_data[0]==0:
            if legs.poses is not None: #only if there are legs detected           
                k=0
                pos=self.position_lidar
                for pose in legs.poses:
                    if k<len(pos):
                        ##########################################################################################################
                        #THIS DISTANCE NEEDS TO BE CALIBRATED ACCORDING TO THE LIDAR READINGS RESPECT TO THE CAMERAS POSITION (The camera location is the reference origin)
                        pos[k,0] = pose.position.x + lidar_to_cam[0] # -0.35
                        pos[k,1] = pose.position.y + lidar_to_cam[1]  #+0
                        ############################################################################################################
                    else: #if there are more human detected than before
                        position_new=np.array([pose.position.x,pose.position.y])
                        pos=np.append(pos,[position_new],axis=0)
                    k=k+1
                if k<len(pos): #if there are less human detected than before
                    pos=pos[0:k,:]    
                self.position_lidar=pos
                new_data[0]=1 # update flag for new data only if there are legs detected, not only for receiving empty messages
                time_diff[0]=time.time()-time_init-time_data[0]
                time_data[0]=time.time()-time_init # update time when the last data was received
        
    def camera_callback(self,data):
        #print("Camera message received")
        self.timer_safety_cam.cancel()
        self.timer_safety_cam = threading.Timer(self.time_without_msg,self.safety_timeout_cam) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety_cam.start()
        self.cam_msg=True 
        #print("DATA FROM CAMERA")
        if new_data[1]==0 and len(data.distance)>0: #only if there is at least one human detected by camera
            k=0
            posture=self.posture
            orientation=self.orientation
            distance=self.distance
            centroid=self.centroid
            position_cam=self.position_cam
            for i in range(0,len(data.posture)):
                if i<len(posture):
                    posture[k,0]=data.posture[i]
                    posture[k,1]=data.posture_prob[i]
                    orientation[k,0]=data.orientation[i]
                    distance[k,0]=data.distance[i]
                    centroid[k,0]=data.centroid_x[i]
                    centroid[k,1]=data.centroid_y[i]
                    position_cam[k,0]=data.position_x[i]
                    position_cam[k,1]=data.position_y[i]
                else: #if there are more human detected than before
                    posture_new=np.array([data.posture[i],data.posture_prob[i]])
                    orientation_new=np.array([data.orientation[i]])
                    distance_new=np.array([data.distance[i]])
                    centroid_new=np.array([data.centroid_x[i],data.centroid_y[i]])
                    position_new=np.array([data.position_x[i],data.position_y[i]])
                    
                    posture=np.append(posture,[posture_new],axis=0)
                    orientation=np.append(orientation,[orientation_new],axis=0)
                    distance=np.append(distance,[distance_new],axis=0)
                    centroid=np.append(centroid,[centroid_new],axis=0)
                    position_cam=np.append(position_cam,[position_new],axis=0)
                    
                k=k+1
            if k<len(posture): #if there are less human detected than before
                posture=posture[0:k,:]
                orientation=orientation[0:k,:]
                distance=distance[0:k,:]
                centroid=centroid[0:k,:]
                position_cam=position_cam[0:k,:]
            self.posture=posture
            self.orientation=orientation
            self.distance=distance
            self.centroid=centroid
            self.position_cam=position_cam
            
        
            new_data[1]=1 # update flag for new data
            time_diff[1]=time.time()-time_init-time_data[1] 
            time_data[1]=time.time()-time_init # update time when the last data was received
        self.thermal_detection=data.thermal_detection
            
    def human_tracking(self):
        sensor=self.sensor
        n_human=self.n_human
        position=self.position_track
        posture=self.posture_track
        motion=self.motion_track
        speed_buffer=self.speed_buffer
        posture_buffer=self.posture_buffer
        posture_prob_buffer=self.posture_prob_buffer
        time_track=self.time_track
        speed=self.speed_track
        counter_motion=self.counter_motion
        counter_posture=self.counter_posture
        centroid=self.centroid_track
        orientation=self.orientation_track
        distance=self.distance_track
        position_cam_new=self.position_cam
        area=self.area
        counter_old=self.counter_old
        position_lidar_new=self.position_lidar
        posture_new=self.posture
        centroid_new=self.centroid
        orientation_new=self.orientation
        distance_new=self.distance
        rob_speed=robot.speed
        action_mode=robot.action_mode
        ##############################################################################################################################################
        #TO MERGE NEW HUMAN DETECTION WITH TRACKED LIST, OR TO ADD NEW DETECTION TO THE TRACKING LIST, OR TO REMOVE OLD DETECTIONS OR FALSE POSITIVES FROM THE LIST
        
        #####New LiDAR info#####################################################################################        
        if new_data[0]==1:
            #print("NEW DATA LIDAR")
            diff=np.zeros([n_human,len(position_lidar_new[:,0])]) #vector with the error between distances
            area_new=np.zeros([len(position_lidar_new[:,0]),1]) #vector with the area of the image where the new human is detected
            new_human_flag=np.zeros([len(position_lidar_new[:,0]),1]) #assuming all are new humans
            for k in range(0,n_human): 
                for kk in range(0,len(position_lidar_new[:,0])):
                    distance_lidar=np.sqrt((position_lidar_new[kk,0])**2+(position_lidar_new[kk,1])**2) 
                    diff[k,kk]=abs(distance[k,:]-distance_lidar)        
                counter_no_new_data=0 # counter to know if the k-th human tracked is not longer detected
                for kk in range(0,len(position_lidar_new[:,0])):
                    if new_human_flag[kk]==0: # Consider the kk-th new human only if it was not matched with another tracked human before
                        #Determining the area where the new human is detected
                        ##############################################################################################################################
                        #It depends how to interpret the X-Y frame used by the human leg detector
                        angle=atan2(position_lidar_new[kk,1],position_lidar_new[kk,0])# if local x-axis is aligned to the robot orientation
                        #############################################################################################################################
                        if angle>np.pi: #  to keep the angle between [-180,+180]
                            angle=angle-2*np.pi
                        if angle<-np.pi:
                            angle=angle+2*np.pi
                        area_new[kk,0]=self.area_inference(angle,position_lidar_new[kk,1],position_lidar_new[kk,0],action_mode)                     
                        ###############################################################################################################################
                        #Determine if a new data match with the k-th human tracked
                        if sensor[k]==2: #if previous data is from camera
                            error_threshold=meter_threshold[1] #meters                
                        else:
                            error_threshold=meter_threshold[0] #meters 
                        if diff[k,kk]<error_threshold and ((area[k,0]==area_new[kk,0] and action_mode=="polytunnel") or (abs(area[k,0]-area_new[kk,0])<=1 and action_mode!="polytunnel")):# abs(area[k,0]-area_new[kk,0])<=1: # if a new detection match with a previos detected in distance and area
                            new_index=kk                           
                            #Updating speed,motion and time_track
                            dist_lidar=np.sqrt(position_lidar_new[new_index,0]**2+position_lidar_new[new_index,1]**2)
                            motion_diff=distance[k,:]-dist_lidar
                            speed[k,:]=abs(motion_diff/time_diff[0])-rob_speed
                            time_track[k,0]=time_data[0]
                            #GIVING PRIORITY TO LIDAR DATA FOR MOTION ESTIMATION
                            if sensor[k]==2: # if human was previosly tracked by a camera, then remove the speed_buffer info to only consider newer lidar info
                                counter_motion[k]=0
                            if counter_motion[k]<n_samples: #while the recorded data is less than n_points                
                                speed_buffer[k,int(counter_motion[k])]=speed[k,:]
                                counter_motion[k]=counter_motion[k]+1
                                motion[k]=0
                            else: #to removed old data and replace for newer data
                                speed_buffer[k,0:n_samples-1]=speed_buffer[k,1:n_samples]
                                speed_buffer[k,n_samples-1]=speed[k,:]
                                motion[k]=self.human_motion_inference(speed_buffer[k,:])
                            #Updating position, area, counter_old and distance
                            position[k,0]=w_data[0]*position[k,0]+w_data[1]*position_lidar_new[new_index,0]
                            position[k,1]=w_data[0]*position[k,1]+w_data[1]*position_lidar_new[new_index,1]
                            area[k,0]=area_new[new_index,0]
                            counter_old[k,0]=counter_old[k,0]-1
                            new_human_flag[new_index]=1 #it is not a new human
                            distance[k,:]=w_data[0]*distance[k,:]+w_data[1]*dist_lidar
                            #Updating sensor    
                            if sensor[k]==2: #if before it was only from camera
                                sensor[k,:]=0 #now is from both
                                #print('New human matches with tracked human, merged')
                        else:
                            counter_no_new_data=counter_no_new_data+1 #counter to know if the k-th human tracked does not have a new data
                    else:
                        counter_no_new_data=counter_no_new_data+1 #counter to know if the k-th human tracked does not have a new data
                if counter_no_new_data==len(position_lidar_new[:,0]): #If there is no a new detection of the k-th human
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
                            #print('A human was removed')
                            n_human=n_human-1
                        else:
                            sensor[k,:]=2 #now is only from camera
                            counter_old[k,0]=0 #reset the lidar counter to 0
                            index_to_keep=index_to_keep+[k]
                            #print('No more data from lidar, only camera is considered')
                    else: #sensor==1
                        #print('A human was removed')
                        n_human=n_human-1
            if n_human>0:
                position=position[np.array(index_to_keep)]
                posture=posture[np.array(index_to_keep)]
                motion=motion[np.array(index_to_keep)]
                time_track=time_track[np.array(index_to_keep)]
                speed=speed[np.array(index_to_keep)]
                speed_buffer=speed_buffer[np.array(index_to_keep)]
                posture_buffer=posture_buffer[np.array(index_to_keep)]
                posture_prob_buffer=posture_prob_buffer[np.array(index_to_keep)]
                counter_motion=counter_motion[np.array(index_to_keep)]
                counter_posture=counter_posture[np.array(index_to_keep)]
                counter_old=counter_old[np.array(index_to_keep)]
                centroid=centroid[np.array(index_to_keep)]
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
                posture_buffer=np.zeros([n_human,n_samples_gest])
                posture_prob_buffer=np.zeros([n_human,n_samples_gest])
                counter_motion=np.zeros([n_human,1])
                counter_posture=np.zeros([n_human,1])
                counter_old=np.ones([n_human,2]) #init in 1
                centroid=np.zeros([n_human,2])
                orientation=np.zeros([n_human,1])
                distance=np.zeros([n_human,1])
                sensor=np.zeros([n_human,1])
                area=np.zeros([n_human,1])
            
            #To include a new human to be tracked
            for k in range(0,len(new_human_flag)):
                if new_human_flag[k]==0:
                    #print('New human tracked from the lidar')
                    n_human=n_human+1
                    #Human perception
                    position=np.append(position,[position_lidar_new[k,:]],axis=0)
                    posture=np.append(posture,np.zeros([1,2]),axis=0)
                    centroid=np.append(centroid,np.zeros([1,2]),axis=0)
                    orientation=np.append(orientation,np.zeros([1,1]),axis=0)
                    distance=np.append(distance,np.zeros([1,1]),axis=0)
                    distance[-1,0]=np.sqrt((position_lidar_new[k,0])**2+(position_lidar_new[k,1])**2) 
                    sensor=np.append(sensor,np.ones([1,1]),axis=0) #1 because it is a lidar type data
                    #Posture and Motion inference
                    motion=np.append(motion,np.zeros([1,1]),axis=0) #new human detection starts with motion label 0 = "not_defined"
                    time_track=np.append(time_track,np.ones([1,2])*(time.time()-time_init),axis=0)  
                    speed=np.append(speed,np.zeros([1,1]),axis=0) #initial speed is 0
                    speed_buffer=np.append(speed_buffer,np.zeros([1,n_samples]),axis=0) 
                    speed_buffer[-1,0]=speed[-1,0] # first data in the speed_buffer
                    posture_buffer=np.append(posture_buffer,np.zeros([1,n_samples_gest]),axis=0) 
                    posture_buffer[-1,0]=posture[-1,0] # first data in the posture_buffer
                    posture_prob_buffer=np.append(posture_prob_buffer,np.zeros([1,n_samples_gest]),axis=0) 
                    posture_prob_buffer[-1,0]=posture[-1,1] # first data in the posture_prob_buffer
                    counter_motion=np.append(counter_motion,np.ones([1,1]),axis=0) # first data was recorded
                    counter_posture=np.append(counter_posture,np.ones([1,1]),axis=0) # first data was recorded
                    #Human Tracking
                    counter_old=np.append(counter_old,np.ones([1,2]),axis=0) #it begins in 1 to be sure it is not a false positive
                    area=np.append(area,[area_new[k]],axis=0)
        
        #####New camera info###########################################################################################################
        if new_data[1]==1: 
            #print("NEW DATA CAMERA")
            diff=np.zeros([n_human,len(centroid_new[:,0])])
            area_new=np.zeros([len(centroid_new[:,0]),1]) #vector with the area of the image where the new human is detected
            new_human_flag=np.zeros([len(centroid_new[:,0]),1]) #assuming all are new humans
            error_threshold=meter_threshold[1] #meters
            for k in range(0,n_human):
                for kk in range(0,len(distance_new[:,0])):
                    diff[k,kk]=abs(distance[k,:]-distance_new[kk,:])
                counter_no_new_data=0 # counter to know if the k-th human tracked is not longer detected
                for kk in range(0,len(centroid_new[:,0])):
                    if new_human_flag[kk]==0: # Consider the kk-th new human only if it was not matched with another tracked human before                     
                        ##############################################################################################################################
                        angle=atan2(position_cam_new[kk,1],position_cam_new[kk,0])# if local x-axis is aligned to the robot orientation
                        #############################################################################################################################
                        if angle>np.pi: #  to keep the angle between [-180,+180]
                            angle=angle-2*np.pi
                        if angle<-np.pi:
                            angle=angle+2*np.pi
                        area_new[kk,0]=self.area_inference(angle,position_cam_new[kk,1],position_cam_new[kk,0],action_mode)                     
                        
                        if diff[k,kk]<error_threshold and ((area[k,0]==area_new[kk,0] and action_mode=="polytunnel") or (abs(area[k,0]-area_new[kk,0])<=1 and action_mode!="polytunnel")): #and abs(area[k,0]-area_new[kk,0])<=1: # if a new detection match with a previos detected in distance and area
                            new_index=kk
                            #Updating posture
                            if counter_posture[k]<n_samples_gest: #while the recorded data is less than n_points 
                                posture_buffer[k,int(counter_posture[k])]=posture_new[new_index,0]
                                posture_prob_buffer[k,int(counter_posture[k])]=posture_new[new_index,1]
                                counter_posture[k]=counter_posture[k]+1
                                posture[k,0]=0 #no gesture
                                posture[k,1]=0 #probability 0
                            else: #to removed old data and replace for newer data
                                posture_buffer[k,0:n_samples_gest-1]=posture_buffer[k,1:n_samples_gest]
                                posture_buffer[k,n_samples_gest-1]=posture_new[new_index,0]
                                posture_prob_buffer[k,0:n_samples_gest-1]=posture_prob_buffer[k,1:n_samples_gest]
                                posture_prob_buffer[k,n_samples_gest-1]=posture_new[new_index,1]
                                [posture[k,:],counter_posture[k]]=human.human_gesture_inference(posture_buffer[k,:],posture_prob_buffer[k,:],posture[k,:],counter_posture[k])
                            #Updating centroid, and orientation
                            centroid[k,:]=centroid_new[new_index,:]
                            orientation[k,:]=orientation_new[new_index,:]
                            #Updating time_track
                            time_track[k,1]=time_data[1]                   
                            #GIVING PRIORITY TO LIDAR DATA FOR MOTION ESTIMATION
                            if sensor[k]==2: #only update with camera info if no lidar info is available                          
                                #Updating speed, motion
                                motion_diff=distance[k,:]-distance_new[new_index,:]
                                speed[k,:]=abs(motion_diff/time_diff[1])-rob_speed
                                if counter_motion[k]<n_samples: #while the recorded data is less than n_points                
                                    speed_buffer[k,int(counter_motion[k])]=speed[k,:]
                                    counter_motion[k]=counter_motion[k]+1
                                    motion[k]=0 #static
                                else: #to removed old data and replace for newer data
                                    speed_buffer[k,0:n_samples-1]=speed_buffer[k,1:n_samples]
                                    speed_buffer[k,n_samples-1]=speed[k,:]
                                    motion[k]=self.human_motion_inference(speed_buffer[k,:])
                            #Updating position, distance and area
                            position[k,0]=w_data[0]*position[k,0]+w_data[1]*position_cam_new[new_index,0]
                            position[k,1]=w_data[0]*position[k,1]+w_data[1]*position_cam_new[new_index,1]
                            distance[k,:]=w_data[0]*distance[k,:]+w_data[1]*distance_new[new_index,:]
                            area[k,0]=area_new[new_index,0]                           
                            #Updating counter_old    
                            counter_old[k,1]=counter_old[k,1]-1                          
                            new_human_flag[new_index]=1 #it is not a new human
                            #Updating the sensor    
                            if sensor[k]==1: #if before it was only from lidar
                                sensor[k,:]=0 #now is from both
                                #print('New human matches with tracked human, merged')
                            
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
                            #print('A human was removed')
                            n_human=n_human-1
                        else:
                            sensor[k,:]=1 #now is only from lidar
                            counter_old[k,1]=0 #reset the camera counter to 0
                            index_to_keep=index_to_keep+[k]
                            #print('No more data from camera, only lidar is considered')
                    else: #sensor==1
                        #print('A human was removed')
                        n_human=n_human-1
            if n_human>0:
                position=position[np.array(index_to_keep)]
                posture=posture[np.array(index_to_keep)]
                motion=motion[np.array(index_to_keep)]
                time_track=time_track[np.array(index_to_keep)]
                speed=speed[np.array(index_to_keep)]
                speed_buffer=speed_buffer[np.array(index_to_keep)]
                posture_buffer=posture_buffer[np.array(index_to_keep)]
                posture_prob_buffer=posture_prob_buffer[np.array(index_to_keep)]
                counter_motion=counter_motion[np.array(index_to_keep)]
                counter_posture=counter_posture[np.array(index_to_keep)]
                counter_old=counter_old[np.array(index_to_keep)]
                centroid=centroid[np.array(index_to_keep)]
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
                posture_buffer=np.zeros([n_human,n_samples_gest])
                posture_prob_buffer=np.zeros([n_human,n_samples_gest])
                counter_motion=np.zeros([n_human,1])
                counter_posture=np.zeros([n_human,1])
                counter_old=np.ones([n_human,2]) #init in 1
                centroid=np.zeros([n_human,2])
                orientation=np.zeros([n_human,1])
                distance=np.zeros([n_human,1])
                sensor=np.zeros([n_human,1])
                area=np.zeros([n_human,1])
            #To include a new human to be tracked
            for k in range(0,len(new_human_flag)):
                if new_human_flag[k]==0:# and posture_new[k,1]>=posture_threshold: # only if the gesture estimation probability is reliable 
                    #print('New human tracked from the camera')
                    #print("ADDED",centroid_new[k,0])
                    n_human=n_human+1
                    #Human perception
                    position=np.append(position,[position_cam_new[k,:]],axis=0)
                    posture=np.append(posture,[posture_new[k,:]],axis=0)
                    centroid=np.append(centroid,[centroid_new[k,:]],axis=0)
                    orientation=np.append(orientation,[orientation_new[k,:]],axis=0)
                    distance=np.append(distance,[distance_new[k,:]],axis=0)
                    sensor=np.append(sensor,2*np.ones([1,1]),axis=0) #2 because it is a camera type data
                    #Posture and Motion inference
                    motion=np.append(motion,np.zeros([1,1]),axis=0) #new human detection starts with motion label 0 = "not_defined"
                    time_track=np.append(time_track,np.ones([1,2])*(time.time()-time_init),axis=0) 
                    speed=np.append(speed,np.zeros([1,1]),axis=0) #initial speed is 0
                    speed_buffer=np.append(speed_buffer,np.zeros([1,n_samples]),axis=0) 
                    speed_buffer[-1,0]=speed[-1,0] # first data in the speed_buffer
                    posture_buffer=np.append(posture_buffer,np.zeros([1,n_samples_gest]),axis=0) 
                    posture_buffer[-1,0]=posture[-1,0] # first data in the posture_buffer
                    posture_prob_buffer=np.append(posture_prob_buffer,np.zeros([1,n_samples_gest]),axis=0) 
                    posture_prob_buffer[-1,0]=posture[-1,1] # first data in the posture_prob_buffer
                    counter_motion=np.append(counter_motion,np.ones([1,1]),axis=0) # first data was recorded
                    counter_posture=np.append(counter_posture,np.ones([1,1]),axis=0) # first data was recorded
                    #Human tracking
                    counter_old=np.append(counter_old,np.ones([1,2]),axis=0) #it begins in 1 to be sure it is not a false positive       
                    area=np.append(area,[area_new[k]],axis=0)
        
        ############################################################################################################################
        #TO ENSURE THAT THERE ARE NOT REPEATED HUMANS TO BE TRACKED  
        if new_data[0]==1 or new_data[1]==1:
            repeated_index=np.zeros([n_human,1]) #initially all humans are considered different
            for k in range(0,n_human):
                dist_1=distance[k,:]     
                for i in range (0,n_human):
                    if k!=i and repeated_index[k]==0 and repeated_index[i]==0:
                        dist_2=distance[i,:]   
                        dist_diff=abs(dist_1-dist_2)  
                        if sensor[k]==2 or sensor[i]==2: #if at least one of these two data are from camera
                            error_threshold=meter_threshold[1] #meters                
                        else:
                            error_threshold=meter_threshold[0] #meters 
                        if dist_diff<=error_threshold and ((area[k,0]==area[i,0] and action_mode=="polytunnel") or (abs(area[k,0]-area[i,0])<=1 and action_mode!="polytunnel")):   
                            #print('Repeated human in track_list, merged')
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
                posture_buffer=posture_buffer[np.array(index_to_keep)]
                posture_prob_buffer=posture_prob_buffer[np.array(index_to_keep)]
                counter_motion=counter_motion[np.array(index_to_keep)]
                counter_posture=counter_posture[np.array(index_to_keep)]
                counter_old=counter_old[np.array(index_to_keep)]
                centroid=centroid[np.array(index_to_keep)]
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
                    #print('A human was removed')
                    n_human=n_human-1
            elif sensor[k]==0: #from camera+lidar
                if counter_old[k,0]>counter_old[k,1]: #if lidar data is older than camera data
                    if counter_old[k,0]<tracking_threshold: # if the counter is still < threshold 
                        index_to_keep=index_to_keep+[k]
                    else: # if a human was not detected for longer than the specific threshold
                        sensor[k,:]=2 #now is only from camera
                        counter_old[k,0]=0 #reset the lidar counter to 0
                        index_to_keep=index_to_keep+[k]
                        #print('No more data from lidar, only camera is considered')
                else: #if camera data is older than lidar data
                    if counter_old[k,1]<tracking_threshold: # if the counter is still < threshold 
                        index_to_keep=index_to_keep+[k]
                    else: # if a human was not detected for longer than the specific threshold
                        sensor[k,:]=1 #now is only from lidar
                        counter_old[k,1]=0 #reset the camera counter to 0
                        index_to_keep=index_to_keep+[k]
                        #print('No more data from camera, only lidar is considered')
            else: #from lidar
                if counter_old[k,0]<tracking_threshold: # if the counter is still < threshold 
                    index_to_keep=index_to_keep+[k]
                else: # if a human was not detected for longer than the specific threshold               
                    #print('A human was removed')
                    n_human=n_human-1
        
        if n_human>0:
            position=position[np.array(index_to_keep)]
            posture=posture[np.array(index_to_keep)]
            motion=motion[np.array(index_to_keep)]
            time_track=time_track[np.array(index_to_keep)]
            speed=speed[np.array(index_to_keep)]
            speed_buffer=speed_buffer[np.array(index_to_keep)]
            posture_buffer=posture_buffer[np.array(index_to_keep)]
            posture_prob_buffer=posture_prob_buffer[np.array(index_to_keep)]
            counter_motion=counter_motion[np.array(index_to_keep)]
            counter_posture=counter_posture[np.array(index_to_keep)]
            counter_old=counter_old[np.array(index_to_keep)]
            centroid=centroid[np.array(index_to_keep)]
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
            posture_buffer=np.zeros([n_human,n_samples_gest])
            posture_prob_buffer=np.zeros([n_human,n_samples_gest])
            counter_motion=np.zeros([n_human,1])
            counter_posture=np.zeros([n_human,1])
            counter_old=np.ones([n_human,2]) #init in 1
            centroid=np.zeros([n_human,2])
            orientation=np.zeros([n_human,1])
            distance=np.zeros([n_human,1])
            sensor=np.zeros([n_human,1])
            area=np.zeros([n_human,1])
        
        
        self.sensor=sensor
        self.n_human=n_human
        self.position_track=position
        self.posture_track=posture
        self.centroid_track=centroid
        self.orientation_track=orientation
        self.distance_track=distance
        self.motion_track=motion
        self.time_track=time_track
        self.speed_track=speed
        self.speed_buffer=speed_buffer
        self.posture_buffer=posture_buffer
        self.posture_prob_buffer=posture_prob_buffer
        self.counter_motion=counter_motion
        self.counter_posture=counter_posture
        self.area=area
        self.counter_old=counter_old

    def human_motion_inference(self,speed):
        speed_mean=np.mean(speed)
        if abs(speed_mean)<speed_threshold: # if human is  mostly static
            motion=1
        else: #if human is moving
            motion=2           
        return motion

    def human_gesture_inference(self,posture,posture_prob,current_posture,counter_posture):
        count=0
        post=0
        for k in range(0,n_samples_gest-1):
            if posture[k]==posture[k+1]:
                count=count+1
                post=posture[k]
        if count>=n_samples_gest-1: #if most of the postures in the buffer are the same, then update with new posture
            prob=np.mean(posture_prob)
            if prob<posture_threshold: #if the openpose probability is not reliable, then consider new posture as "no gesture"
                post=0 #no gesture
                prob=0 #probability 0
            if current_posture[0]!=post and current_posture[0]!=0: #reset buffer if change in gesture is detected, except when previos gesture was "no gesture"
                counter_posture=0
                post=0 #no gesture
                prob=0 #probability 0
        else: #keep the last posture label and probability
            post=0#current_posture[0] #0 #no gesture
            prob=0#current_posture[1] #0 #probability 0
        result=np.zeros([1,2])
        result[0,0]=post
        result[0,1]=prob
        return result,counter_posture

    def area_inference(self,angle,pos_y,pos_x,action_mode):
        a=angle_area*(pi/180)
        w=row_width
        n=2 #scaling factor for distance "w"
        m=1 #scaling factor for angle "a"
        if action_mode=="polytunnel":                    
            #Front
            if (pos_y>=0 and pos_y<=(3/2)*w and angle>=a and angle<=pi/2) or (pos_y>(3/2)*w and pos_x>0): #if belongs to 0
                area=0
            elif pos_y>=w/2 and pos_y<=(3/2)*w and angle>=0 and angle<=a: # if belongs to area 1
                area=1
            elif pos_y>=-(3/2)*w and pos_y<=-w/2 and angle<=0 and angle>=-a: # if belongs to area 3
                area=3
            elif (pos_y>=-(3/2)*w and pos_y<=0 and angle<=-a and angle>=-pi/2) or (pos_y<=-(3/2)*w and pos_x>0): #if belongs to 4   
                area=4
            elif pos_y>=-w/2 and pos_y<=w/2  and pos_x>=0: # if belongs to area 2 
                area=2
            
            #Back
            elif (pos_y>=-(3/2)*w and pos_y<=0 and angle>=-pi+a and angle<=-pi/2) or (pos_y<=-(3/2)*w and pos_x<0): #if belongs to 5   
                area=5
            elif pos_y>=-(3/2)*w and pos_y<=-w/2 and angle>=-pi and angle<=-pi+a: # if belongs to area 6
                area=6
            elif pos_y>=w/2 and pos_y<=(3/2)*w and angle<=pi and angle>=pi-a: # if belongs to area 8
                area=8
            elif (pos_y>=0 and pos_y<=(3/2)*w and angle<=pi-a and angle>=pi/2) or (pos_y>=(3/2)*w and pos_x<0): #if belongs to 9
                area=9
            elif pos_y>= -w/2 and pos_y<=w/2 and  pos_x<=0: # if belongs to area 7 
                area=7
        else: #"footpath"
            a=m*a
            w=n*w
            #Front
            if (pos_y>=w/2 and pos_y<=(3/2)*w and angle>=a and angle<=pi/2) or (pos_y>(3/2)*w and pos_x>0): #if belongs to 0
                area=0
            elif pos_y>=w/2 and pos_y<=(3/2)*w and angle>=0 and angle<=a: # if belongs to area 1
                area=1
            elif pos_y>=-w/2 and pos_y<=w/2 and pos_x>=0: # if belongs to area 2
                area=2
            elif pos_y>=-(3/2)*w and pos_y<=-w/2 and angle<=0 and angle>=-a: # if belongs to area 3
                area=3
            elif (pos_y>=-(3/2)*w and pos_y<=-w/2 and angle<=-a and angle>=-pi/2) or (pos_y<=-(3/2)*w and pos_x>0): #if belongs to 4   
                area=4
            #Back
            elif (pos_y>=-(3/2)*w and pos_y<=-w/2 and angle>=-pi+a and angle<=-pi/2) or (pos_y<=-(3/2)*w and pos_x<0): #if belongs to 5   
                area=5
            elif pos_y>=-(3/2)*w and pos_y<=-w/2 and angle>=-pi and angle<=-pi+a: # if belongs to area 6
                area=6
            elif pos_y>= -w/2 and pos_y<=w/2 and pos_x<=0: # if belongs to area 7
                area=7
            elif pos_y>=w/2 and pos_y<=(3/2)*w and angle<=pi and angle>=pi-a: # if belongs to area 8
                area=8
            elif (pos_y>=w/2 and pos_y<=(3/2)*w and angle<=pi-a and angle>=pi/2) or (pos_y>=(3/2)*w and pos_x<0): #if belongs to 9
                area=9
        
        return area
    
    
    def safety_timeout_cam(self):
        print("No camera messages received in a long time")
        self.cam_msg=False #to let the safety system know that human_perception is not getting info from the cameras
        
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node
    time_init=time.time()       
    human=human_class()  
    robot=robot_class()
    rospy.init_node('human_perception_system',anonymous=True)
    # Setup and call subscription
    rospy.Subscriber('/nav_vel',geometry_msgs.msg.Twist,robot.robot_speed_callback) 
    rospy.Subscriber('/people_tracker/pose_array',PoseArray,human.lidar_callback) 
    rospy.Subscriber('/human_info_camera',human_detector_msg,human.camera_callback)  
    rospy.Subscriber('human_safety_info',hri_msg,robot.robot_action_callback) 
    #Rate setup
    rate = rospy.Rate(1/pub_hz)  # main loop frecuency in Hz
    while not rospy.is_shutdown():	
        main_counter=main_counter+1
        #Human tracking
        human.human_tracking()     
        if new_data[0]==1:
            new_data[0]=0
        if new_data[1]==1:
            new_data[1]=0
        #Publish human_perception messages only if camaras are still publishing messages  
        if human.cam_msg==True:
            if human.n_human>1 or (human.n_human==1 and (human.motion_track[0,0]+human.centroid_track[0,0]+human.centroid_track[0,1]+human.posture_track[0,0]+human.position_track[0,0]+human.position_track[0,1]+human.distance_track[0,0])!=0):
                msg.n_human = human.n_human
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
                msg.thermal_detection=human.thermal_detection
            else: #no human detected
                msg.n_human = 0 
                msg.posture = []
                msg.posture_prob = []
                msg.motion = []
                msg.position_x = []
                msg.position_y = []
                msg.centroid_x = []
                msg.centroid_y = []
                msg.distance = []
                msg.orientation = []
                msg.area = []
                msg.sensor = []
                msg.thermal_detection=human.thermal_detection
            pub.publish(msg)
        rate.sleep() #to keep fixed the publishing loop rate
        