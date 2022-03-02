#!/usr/bin/env python

#required packages
import rospy #tf
from sensor_msgs.msg import Joy
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import marvelmind_nav.msg
from tf.transformations import euler_from_quaternion
from mesapro.msg import human_msg
from geometry_msgs.msg import PoseStamped
import time

##########################################################################################
picker_step=0.05 # maximum picker step each time an action is triggered 
picker_step_angle=0.1 # maximim picker step in radians
n_samples=10 #number of samples used for the motion inference
speed_threshold=[0.6,1]  # [static, slow motion] m/s
areas_angle=[150,100,80,30,0] #in degrees
dist_detection=7 #range in meters of human detection
dist_thermal_detection=10 #range in meters of thermal detection
main_counter=0

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
        self.motion=np.zeros([2,1]) 
        self.centroid=np.zeros([2,2])
        self.area=np.zeros([2,1]) #initially considered on the side of the robot
        self.speed=np.zeros([2,1]) 
        self.speed_buffer=np.zeros([2,n_samples]) #buffer with the human speed recorded during n_points
        self.counter_motion=np.zeros([2,1]) # vector with the number of samples that have been recorded for motion inference
        self.time=np.zeros([2,1])# time of data recorded for each human
        self.thermal_detection=False #initial condition
        
    def actor00_callback(self,p1):
        #print("ACTOR00 NEW DATA")
        #if new_data[1]==0:
        #Sensor type, time, counter, centroid are not required for the demo, thus they are always 0
        
        #Positions from virtual picker
        pos = p1.pose.position
        quat = p1.pose.orientation    
        # From quaternion to Euler
        angles = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        theta = angles[2]
        pos_theta=np.unwrap([theta])[0]
        pose= np.array([pos.x,pos.y,pos_theta])
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

        #Transform human_position from global frame to local frame
        aux=pose-robot.position
        aux_a=atan2(aux[1],aux[0])
        dist=sqrt(aux[1]**2+aux[0]**2)
        pos_x=(cos(-robot.position[2]+aux_a)*dist)
        pos_y=(sin(-robot.position[2]+aux_a)*dist)
        self.position[0,0]=pos_x
        self.position[0,1]=pos_y
        #Area
        self.area[0]=area_inference_lidar(pos_y,pos_x)
        #Orientation
        #if self.area[0]>=0 and self.area[0]<=4:
        #    if pose[2]<=robot.position[2]+pi/2 and pose[2]>=robot.position[2]-pi/2:
        #        orientation=1 #back
        #    else:  #front
        #        orientation=0
        #else:
        #    if pose[2]<=robot.position[2]+pi/2 and pose[2]>=robot.position[2]-pi/2:
        #        orientation=0 #front
        #    else:  #back
        #        orientation=1
        #self.orientation[0]=orientation
        #new_data[1]=1

    def actor01_callback(self,p2):
        #print("ACTOR01 NEW DATA")
        #if new_data[2]==0:
        #Sensor type, time, counter, centroid are not required for the demo, thus they are always 0
        
        #Positions from virtual picker
        pos = p2.pose.position
        quat = p2.pose.orientation    
        # From quaternion to Euler
        angles = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        theta = angles[2]
        pos_theta=np.unwrap([theta])[0]
        pose= np.array([pos.x,pos.y,pos_theta])
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
       
        #Transform human_position from global frame to local frame (the one which is actually measured by the real robot)
        aux=pose-robot.position
        aux_a=atan2(aux[1],aux[0])
        dist=sqrt(aux[1]**2+aux[0]**2)
        pos_x=(cos(-robot.position[2]+aux_a)*dist)
        pos_y=(sin(-robot.position[2]+aux_a)*dist)
        self.position[1,0]=pos_x
        self.position[1,1]=pos_y
        #Area
        self.area[1]=area_inference_lidar(pos_y,pos_x)
        #Orientation
        #if self.area[1]>=0 and self.area[1]<=4:
        #    if pose[2]<=robot.position[2]+pi/2 and pose[2]>=robot.position[2]-pi/2:
        #        orientation=1 #back
        #    else:  #front
        #        orientation=0
        #else:
        #    if pose[2]<=robot.position[2]+pi/2 and pose[2]>=robot.position[2]-pi/2:
        #        orientation=0 #front
        #    else:  #back
        #        orientation=1
        #self.orientation[1]=orientation
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

        pos_x=pose.position.x
        pos_y=pose.position.y
        quat = pose.orientation    
        # From quaternion to Euler
        angles = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        theta = angles[2]
        pos_theta=np.unwrap([theta])[0]

        self.position=np.array([pos_x, pos_y, pos_theta])

    
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
            #if buttons[15]>0: #up to change the human area to any (in front)
            #    human.area[1]=2
            #if buttons[17]>0: #down to change the human area to any (back)
            #    human.area[1]=7
            #if buttons[14]>0: #left to change the human area to 0 (on the side)
            #    human.area[1]=0    
                
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
                
                if axes[4]>0: #left trigger
                    human.position_global[1,2]=human.position_global[1][2]+abs(axes[4])*picker_step_angle
                if axes[5]>0: #right trigger
                    human.position_global[1,2]=human.position_global[1][2]-abs(axes[5])*picker_step_angle
                
                
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
            #if buttons[15]>0: #up to change the human area to any (in front)
            #    human.area[0]=2
            #if buttons[17]>0: #down to change the human area to any (back)
            #    human.area[0]=7
            #if buttons[14]>0: #left to change the human area to 0 (on the side)
            #    human.area[0]=0
                
                
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
                
                if axes[4]>0: #left trigger
                    human.position_global[0,2]=human.position_global[0][2]+abs(axes[4])*picker_step_angle
                if axes[5]>0: #right trigger
                    human.position_global[0,2]=human.position_global[0][2]-abs(axes[5])*picker_step_angle
                
            if buttons[3]>0: #X to reset gesture
                if human.posture[0,0]!=0:
                    human.posture[0,0]=0 #to reset human gesture
    
def area_inference_lidar(pos_y,pos_x):
    row_width=1.3 #in meters
    angle_area=60 # in degrees mesuared from the local x-axis robot frame
    angle=atan2(pos_y,pos_x)# local x-axis is aligned to the robot orientation
    if angle>pi: #  to keep the angle between [-180,+180]
        angle=angle-2*pi
    if angle<-pi:
        angle=angle+2*pi
                        
    #Front
    if (pos_y>=row_width/2 and pos_y<=(3/2)*row_width and angle>=angle_area*(pi/180) and angle<=pi/2) or (pos_y>(3/2)*row_width and pos_x>0): #if belongs to 0
        area=0
    elif pos_y>=row_width/2 and pos_y<=(3/2)*row_width and angle>=0 and angle<=angle_area*(pi/180): # if belongs to area 1
        area=1
    elif pos_y>=-row_width/2 and pos_y<=row_width/2 and pos_x>=0: # if belongs to area 2
        area=2
    elif pos_y>=-(3/2)*row_width and pos_y<=-row_width/2 and angle<=0 and angle>=-angle_area*(pi/180): # if belongs to area 3
        area=3
    elif (pos_y>=-(3/2)*row_width and pos_y<=-row_width/2 and angle<=-angle_area*(pi/180) and angle>=-pi/2) or (pos_y<=-(3/2)*row_width and pos_x>0): #if belongs to 4   
        area=4
    #Back
    elif (pos_y>=-(3/2)*row_width and pos_y<=-row_width/2 and angle>=-pi+angle_area*(pi/180) and angle<=-pi/2) or (pos_y<=-(3/2)*row_width and pos_x<0): #if belongs to 5   
        area=5
    elif pos_y>=-(3/2)*row_width and pos_y<=-row_width/2 and angle>=-pi and angle<=-pi+angle_area*(pi/180): # if belongs to area 6
        area=6
    elif pos_y>= -row_width/2 and pos_y<=row_width/2 and pos_x<=0: # if belongs to area 7
        area=7
    elif pos_y>=row_width/2 and pos_y<=(3/2)*row_width and angle<=pi and angle>=pi-angle_area*(pi/180): # if belongs to area 8
        area=8
    elif (pos_y>=row_width/2 and pos_y<=(3/2)*row_width and angle<=pi-angle_area*(pi/180) and angle>=pi/2) or (pos_y>=(3/2)*row_width and pos_x<0): #if belongs to 9
        area=9
    
    return area
############################################################################################
# Main Script
if __name__ == '__main__':
    time_init=time.time()
    # Initialize our node
    rospy.init_node('virtual_picker_simulation',anonymous=True)
    human=human_class()
    robot=robot_class()
    #Setup ROS publiser
    pub_human = rospy.Publisher('human_info', human_msg, queue_size=1)
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
        if human.distance[0,0]<=dist_detection and human.distance[1,0]<=dist_detection: 
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
            msg.n_human = 2
        elif human.distance[0,0]>dist_detection and human.distance[1,0]>dist_detection: 
            msg.n_human= 0
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
        elif human.distance[0,0]<=dist_detection and human.distance[1,0]>dist_detection:
            msg.posture = [human.posture[0,0]]
            msg.posture_prob = [human.posture[0,1]]
            msg.motion = [human.motion[0,0]]
            msg.position_x = [human.position[0,0]]
            msg.position_y = [human.position[0,1]]
            msg.centroid_x =[human.centroid[0,0]]
            msg.centroid_y =[human.centroid[0,1]]
            msg.distance = [human.distance[0,0]]
            msg.orientation = [human.orientation[0,0]]
            msg.area = [human.area[0,0]]
            msg.sensor = [human.sensor[0,0]]
            msg.n_human = 1
        elif human.distance[0,0]>dist_detection and human.distance[1,0]<=dist_detection:
            msg.posture = [human.posture[1,0]]
            msg.posture_prob = [human.posture[1,1]]
            msg.motion = [human.motion[1,0]]
            msg.position_x = [human.position[1,0]]
            msg.position_y = [human.position[1,1]]
            msg.centroid_x =[human.centroid[1,0]]
            msg.centroid_y =[human.centroid[1,1]]
            msg.distance = [human.distance[1,0]]
            msg.orientation = [human.orientation[1,0]]
            msg.area = [human.area[1,0]]
            msg.sensor = [human.sensor[1,0]]
            msg.n_human = 1
        if human.distance[0,0]<=dist_thermal_detection or human.distance[1,0]<=dist_thermal_detection:
            msg.thermal_detection=True
        else:
            msg.thermal_detection=False
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
        
        #setup publiser in ROS
        pub=rospy.Publisher('/hedge_pos_a',marvelmind_nav.msg.hedge_pos_a, queue_size=10)
        msg= marvelmind_nav.msg.hedge_pos_a()
        #Publish command for Picker 1
        msg.address= 1
        msg.timestamp_ms=0
        msg.x_m = human.position_global[0][0]
        msg.y_m = human.position_global[0][1]
        msg.z_m = 0
        pub.publish(msg)
        #Publish command for Picker 2
        msg.address= 2
        msg.timestamp_ms=0
        msg.x_m = human.position_global[1][0]
        msg.y_m = human.position_global[1][1]
        msg.z_m = 0
        pub.publish(msg)
        
        #pub=rospy.Publisher('/picker01/posestamped',PoseStamped)
        #msg=PoseStamped()
        #msg.header.frame_id = "map"
        #msg.pose.position.x=human.position_global[0][0]
        #msg.pose.position.y=human.position_global[0][1]
        #msg.pose.orientation.z=2#human.position_global[0][2]
        #msg.pose.orientation.x=0#human.position_global[0][2]
        #msg.pose.orientation.y=0#human.position_global[0][2]
        #msg.pose.orientation.w = 1.0
        #pub.publish(msg)
        #pub=rospy.Publisher('/picker02/posestamped',PoseStamped)
        #msg=PoseStamped()
        #msg.pose.position.x=human.position_global[1][0]
        #msg.pose.position.y=human.position_global[1][1]
        #msg.pose.orientation.z=human.position_global[1][2]
        #pub.publish(msg)
        
        
        #print('PICKER 2 -- X: %4.1f Y: %4.1f'%(human.position_global[1][0],human.position_global[1][1]))
        rate.sleep() #to keep fixed the control loop rate
        #if new_data[0]==1:    
        #    new_data[0]=0
                
        
            

   

