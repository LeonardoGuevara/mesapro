#!/usr/bin/env python

#required packages
import rospy #tf
from sensor_msgs.msg import Joy
import geometry_msgs.msg, nav_msgs.msg
from geometry_msgs.msg import Pose
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import marvelmind_nav.msg
import std_msgs.msg
#import tf
from tf.transformations import euler_from_quaternion
from mesapro.msg import human_msg, hri_msg, robot_msg
from geometry_msgs.msg import PoseStamped
import time

##########################################################################################
picker_step=0.05 # maximum picker step each time an action is triggered 
n_samples=10 #number of samples used for the motion inference
#new_data=[0,0,0,0,0] #joy, actor0, actor1, robot_pos, robot_vel
speed_threshold=[0.6,1]  # [static, slow motion] m/s
areas_angle=[150,100,80,30,0] #in degrees
main_counter=0
#Main class definition

class human_class:
    def __init__(self): #It is done only the first iteration
        #INFORMATION SIMULATED IN GAZEBO
        self.position=np.zeros([2,3]) #x,y,theta
        self.position_global=np.zeros([2,3])
        self.posture=np.zeros([2,2]) #from camera [posture_label,posture_probability]
        self.posture[0,1]=1 #perfect gesture recognition
        self.posture[1,1]=1 #perfect gesture recognition
        self.orientation=np.zeros([2,1]) # it can be "front" or "back" if the human is facing the robot or not , from camera
        self.distance=np.zeros([2,1])  # distance between the robot and the average of the skeleton joints distances taken from the depth image, from camera
        self.sensor=np.zeros([2,1]) # it can be 0 if the data is from camera and Lidar, 1 if the data is  only from the LiDAR or 2 if the data is only from de camera
        self.sensor_t=np.zeros([2,2])
        self.sensor_c=np.zeros([2,2])
        self.sensor_c[0,1]=-1 # to make the camera info the latest one 
        self.sensor_c[1,1]=-1 # to make the camera info the latest one
        self.sensor_t[0,1]=1 # to make the camera info the latest one 
        self.sensor_t[1,1]=1 # to make the camera info the latest one
        self.motion=np.zeros([2,1]) 
        self.centroid=np.zeros([2,2])
        self.area=np.zeros([2,1]) #initially considered on the side of the robot
        self.speed=np.zeros([2,1]) 
        self.speed_buffer=np.zeros([2,n_samples]) #buffer with the human speed recorded during n_points
        self.counter_motion=np.zeros([2,1]) # vector with the number of samples that have been recorded for motion inference
        self.time=np.zeros([2,1])# time of data recorded for each human
        
        
    def actor00_callback(self,p1):
        #print("ACTOR00 NEW DATA")
        #if new_data[1]==0:
        #Sensor type, time, counter, centroid are not required for the demo, thus they are always 0
        
        #Positions from virtual picker
        pos = p1.pose.position
        pose= np.array([pos.x,pos.y,0])
        self.position_global[0,:]=pose
        #Human Motion and distance
        time_new=time.time()-time_init
        #print("ROBBOT",robot.position)
        #print("PICKER",pose)
        distance_new=sqrt((robot.position[0]-pose[0])**2+(robot.position[1]-pose[1])**2)
        #to include the thorlvard dimensions
        distance_new=distance_new-1
        if distance_new<0:
            distance_new=0
         
        self.speed[0,0]=abs((distance_new-self.distance[0,0])/(time_new-self.time[0]))-abs(robot.speed)
        #print("HUMAN SPEED",(distance_new-self.distance[0,0])/(time_new-self.time[0]))
        #print("ROBOT SPEED",robot.input[1])
        #print("ROBOT SPEED", robot.speed)
        
        k=0
        if self.counter_motion[k]<n_samples: #while the recorded data is less than n_points                
            self.speed_buffer[k,int(self.counter_motion[k])]=self.speed[k,:]
            self.counter_motion[k]=self.counter_motion[k]+1
        else: #to removed old data and replace for newer data
            self.speed_buffer[k,0:n_samples-1]=self.speed_buffer[k,1:n_samples]
            self.speed_buffer[k,n_samples-1]=self.speed[k,:]
        self.distance[0,0]=distance_new
        self.time[0]=time_new
        ii=0
        if self.counter_motion[ii]>=n_samples:
            speed_mean=np.mean(self.speed_buffer[ii,:])
            if abs(speed_mean)<speed_threshold[0]: # if human is  mostly static
                self.motion[ii]=1
            else: #if human is moving 
                self.motion[ii]=2
        else:
            self.motion[ii]=0
        #Human Area (using only y-position in global frame)
        #print("AREA ERROR",abs(pose[0]-robot.position[0]))
        #if abs(pose[1]-robot.position[1])<=2:
        #    self.area[0,0]=2
        #else:
        #    self.area[0,0]=0
        #self.area[0,0]=2
        #Transform human_position from global frame to local frame
        
        #self.position[0,:]=pose-robot.position
        aux=pose-robot.position
        aux_a=atan2(aux[1],aux[0])
        dist=sqrt(aux[1]**2+aux[0]**2)
        pos_x=(cos(-robot.position[2]+aux_a)*dist)
        pos_y=(sin(-robot.position[2]+aux_a)*dist)
        self.position[0,0]=pos_x
        self.position[0,1]=pos_y
        #new_data[1]=1

    def actor01_callback(self,p2):
        #print("ACTOR01 NEW DATA")
        #if new_data[2]==0:
        #Sensor type, time, counter, centroid are not required for the demo, thus they are always 0
        
        #Positions from virtual picker
        pos = p2.pose.position
        pose= np.array([pos.x,pos.y,0])
        self.position_global[1,:]=pose
        #Human Motion and distance
        time_new=time.time()-time_init
        distance_new=sqrt((robot.position[0]-pose[0])**2+(robot.position[1]-pose[1])**2)
        #to include the thorlvard dimensions
        distance_new=distance_new-1
        if distance_new<0:
           distance_new=0
        #self.speed[1,0]= abs((distance_new-self.distance[1,0])/(time_new-self.time[1]))-abs(robot.input[1])
        self.speed[1,0]= abs((distance_new-self.distance[1,0])/(time_new-self.time[1]))-abs(robot.speed)
        k=1
        if self.counter_motion[k]<n_samples: #while the recorded data is less than n_points                
            self.speed_buffer[k,int(self.counter_motion[k])]=self.speed[k,:]
            self.counter_motion[k]=self.counter_motion[k]+1
        else: #to removed old data and replace for newer data
            self.speed_buffer[k,0:n_samples-1]=self.speed_buffer[k,1:n_samples]
            self.speed_buffer[k,n_samples-1]=self.speed[k,:]
        self.distance[1,0]=distance_new
        self.time[1]=time_new
        ii=1
        if self.counter_motion[ii]>=n_samples:
            speed_mean=np.mean(self.speed_buffer[ii,:])
            if abs(speed_mean)<speed_threshold[0]: # if human is  mostly static
                self.motion[ii]=1
            else: #if human is moving
                self.motion[ii]=2
        else:
            self.motion[ii]=0
        #Human Area (using only y-position in global frame)
        #if abs(pose[1]-robot.position[1])<=1:
        #    self.area[1,0]=2
        #else:
        #    self.area[1,0]=0
        #self.area[1,0]=2
        #Transform human_position from global frame to local frame (the one which is actually measured by the real robot)
        aux=pose-robot.position
        aux_a=atan2(aux[1],aux[0])
        dist=sqrt(aux[1]**2+aux[0]**2)
        pos_x=(cos(-robot.position[2]+aux_a)*dist)
        pos_y=(sin(-robot.position[2]+aux_a)*dist)
        self.position[1,0]=pos_x
        self.position[1,1]=pos_y
        
        #new_data[2]=1

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        #INFORMATION SIMULATED IN GAZEBO 
        self.position=np.zeros([1,3]) #[x,y,theta]
        #self.position_past=np.zeros([1,3]) #[x,y,theta]
        self.input=np.zeros([2,1]) #w,v control signals
        self.speed=0 #absolute value of speed
        self.time=0 #time of data received from odometry

    def robot_callback_pos(self,pose):
        #print("ROBOTS NEW DATA")
        #if new_data[3]==0:
        #pos = odom.pose.pose
        pos_x=pose.position.x
        pos_y=pose.position.y
        quat = pose.orientation    
        # From quaternion to Euler
        angles = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        theta = angles[2]
        pos_theta=np.unwrap([theta])[0]
        
        #robot.position_past=robot.position
        self.position=np.array([pos_x, pos_y, pos_theta])
        #time_new=time.time()-time_init
        #print("ROBOT POSITION",robot.position) 
        #robot.speed=sqrt((robot.position_past[0]-pos.position.x)**2+(robot.position_past[1]-pos.position.y)**2)/(time_new-robot.time)
        #robot.time=time_new
        #new_data[3]=1
    
    def robot_callback_vel(self,rob):
        #print("ROBOTS NEW DATA")
        #if new_data[4]==0:
        self.input[1] = rob.linear.x
        self.input[0] = rob.angular.z
        self.speed=abs(robot.input[1])
        #new_data[4]=1
       


def joy_callback(data):
    #print("JOY NEW DATA")
    buttons=data.buttons
    axes=data.axes  
    #if new_data[0]==0:
    if np.shape(buttons)[0]>0:
        #new_data[0]=1 
        if buttons[4]>0: #L1 to control picker01 
            if buttons[0]>0: #square is two arms (approach) 
                human.posture[1,0]=1
            if buttons[1]>0: #triangle is right hand (stop)
                human.posture[1,0]=8
            if buttons[2]>0: #circle is two hands (move away)
                human.posture[1,0]=2
            if buttons[9]>0: #option to change the human orientation to back
                human.orientation[1]=1
            if buttons[8]>0: #start to change the human orientation to front
                human.orientation[1]=0
            if buttons[15]>0: #up to change the human area to any (in front)
                human.area[1]=2
            if buttons[17]>0: #down to change the human area to any (back)
                human.area[1]=7
            if buttons[14]>0: #left to change the human area to 0 (on the side)
                human.area[1]=0    
                
            if np.shape(axes)[0]!=0:
                #Picker01
                if axes[0]<0: #left
                    human.position_global[1,0]=human.position_global[1][0]-abs(axes[0])*picker_step
                if axes[0]>0: #rigth
                    human.position_global[1,0]=human.position_global[1][0]+abs(axes[0])*picker_step
                if axes[1]>0: #up
                    human.position_global[1,1]=human.position_global[1][1]-abs(axes[1])*picker_step
                if axes[1]<0: #down
                    human.position_global[1,1]=human.position_global[1][1]+abs(axes[1])*picker_step
            if buttons[3]>0: #X to reset gesture
                if human.posture[1,0]!=0:
                    human.posture[1,0]=0 #to reset human gesture
         
        elif buttons[6]>0: #R1 to control picker00 
            if buttons[0]>0: #square is two arms (approach) 
                human.posture[0,0]=1
            if buttons[1]>0: #triangle is right hand (stop)
                human.posture[0,0]=8
            if buttons[2]>0: #circle is two hands (move away)
                human.posture[0,0]=2
            if buttons[9]>0: #option to change the human orientation to back
                human.orientation[0]=1
            if buttons[8]>0: #start to change the human orientation to front
                human.orientation[0]=0
            if buttons[15]>0: #up to change the human area to any (in front)
                human.area[0]=2
            if buttons[17]>0: #down to change the human area to any (back)
                human.area[0]=7
            if buttons[14]>0: #left to change the human area to 0 (on the side)
                human.area[0]=0
                
                
            if np.shape(axes)[0]!=0:
                #Picker00
                if axes[0]<0: #left
                    human.position_global[0,0]=human.position_global[0][0]-abs(axes[0])*picker_step
                if axes[0]>0: #right
                    human.position_global[0,0]=human.position_global[0][0]+abs(axes[0])*picker_step
                if axes[1]>0: #up
                    human.position_global[0,1]=human.position_global[0][1]-abs(axes[1])*picker_step
                if axes[1]<0: #down
                    human.position_global[0,1]=human.position_global[0][1]+abs(axes[1])*picker_step
            if buttons[3]>0: #X to reset gesture
                if human.posture[0,0]!=0:
                    human.posture[0,0]=0 #to reset human gesture
    

############################################################################################
# Main Script
if __name__ == '__main__':
    time_init=time.time()
    # Initialize our node
    rospy.init_node('virtual_picker_simulation',anonymous=True)
    human=human_class()
    robot=robot_class()
    #Setup ROS publiser
    pub_human = rospy.Publisher('human_info', human_msg)
    rospy.Subscriber('joy',Joy,joy_callback)  
    rospy.Subscriber('/picker01/posestamped',PoseStamped, human.actor00_callback)
    rospy.Subscriber('/picker02/posestamped',PoseStamped, human.actor01_callback)
    rospy.Subscriber('/robot_pose', Pose, robot.robot_callback_pos) 
    rospy.Subscriber('/nav_vel',geometry_msgs.msg.Twist,robot.robot_callback_vel)    
    #Rate setup
    pub_hz=0.01 #publising rate in seconds
    rate = rospy.Rate(1/pub_hz) # ROS Rate in Hz
    while not rospy.is_shutdown():
        main_counter=main_counter+1  
        #print("main_counter",main_counter)
        #Setup ROS publiser
        pub_human = rospy.Publisher('human_info', human_msg)
        msg = human_msg()
         #Publish Human_info from Gazebo
        msg.posture = list(human.posture[:,0])
        msg.posture_prob = list(human.posture[:,1])
        msg.motion = list(human.motion[:,0])
        msg.position_x = list(human.position[:,0])
        msg.position_y = list(human.position[:,1])
        msg.centroid_x =list(human.centroid[:,0])
        msg.centroid_y =list(human.centroid[:,1])
        msg.distance = list(human.distance[:,0])
        msg.orientation = list(human.orientation[:,0])
        msg.area = list(human.area[:,0])
        msg.sensor = list(human.sensor[:,0])
        msg.sensor_t0 = list(human.sensor_t[:,0])
        msg.sensor_t1 = list(human.sensor_t[:,1])
        msg.sensor_c0 = list(human.sensor_c[:,0])
        msg.sensor_c1 = list(human.sensor_c[:,1])
        pub_human.publish(msg)
        rate.sleep() #to keep fixed the control loop rate
        #if new_data[1]==1:
        #    new_data[1]=0
        #if new_data[2]==1:
        #    new_data[2]=0
        #if new_data[3]==1:
        #    new_data[3]=0
        #if new_data[4]==1:
        #    new_data[4]=0
        
        #if new_data[0]==1:
        #setup publiser in ROS
        pub=rospy.Publisher('/hedge_pos_a',marvelmind_nav.msg.hedge_pos_a, queue_size=10)
        msg= marvelmind_nav.msg.hedge_pos_a()
        #Publish command for Picker 1
        msg.address= 1
        msg.timestamp_ms=0
        msg.x_m = human.position_global[0][0]
        msg.y_m = human.position_global[0][1]
        msg.z_m = 0.0
        #msg.flag=0
        pub.publish(msg)
        #print('PICKER 1 -- X: %4.1f Y: %4.1f'%(human.position_global[0][0],human.position_global[0][1]))
        #Publish command for Picker 2
        msg.address= 2
        msg.timestamp_ms=0
        msg.x_m = human.position_global[1][0]
        msg.y_m = human.position_global[1][1]
        msg.z_m = 0.0
        #msg.flag=0
        pub.publish(msg)
        #print('PICKER 2 -- X: %4.1f Y: %4.1f'%(human.position_global[1][0],human.position_global[1][1]))
        rate.sleep() #to keep fixed the control loop rate
        #if new_data[0]==1:    
        #    new_data[0]=0
                
        
            

   

