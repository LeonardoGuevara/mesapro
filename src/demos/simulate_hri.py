#! /usr/bin/python

#required packages
import rospy #tf
import geometry_msgs.msg, nav_msgs.msg
from sensor_msgs.msg import Joy
import message_filters #to sync the messages
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import os
import yaml
#import tf
import time
from mesapro.msg import human_msg, hri_msg, robot_msg
##########################################################################################

#Importing global parameters from .yaml file
src_direct=os.getcwd()
config_direct=src_direct[0:len(src_direct)-9]+"config/"
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
direct_param=list(dict.items(parsed_yaml_file["directories_config"]))
ar_param=list(dict.items(parsed_yaml_file["action_recog_config"]))
hs_param=list(dict.items(parsed_yaml_file["human_safety_config"]))
#For Motion inference and matching
speed_threshold=[0.2,0.4]  # [static, slow motion] m/s
areas_angle=[150,100,80,30,0] #in degrees
n_samples=10 #number of samples used for the motion inference
#For human/robot simulation
robot_speed=[0.5,0.1,0] #[normal speed, reduced speed, stop]
picker_speed=[3, 0.5] #average picker speed [angular,linear]
#General purposes variables
main_counter=0
new_data=[0,0,0,0] #flag to know if the safety_msg and gazebo_info has been received
simulation=0 #flag to control the simulation with a button
demo=2 #demo 2: human in gazebo, demo 3: topological navigation
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        #INFORMATION SIMULATED IN GAZEBO
        self.position=np.zeros([2,3]) #x,y,theta
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
        self.area=np.zeros([2,1])
        self.speed=np.zeros([2,1]) 
        self.speed_buffer=np.zeros([2,n_samples]) #buffer with the human speed recorded during n_points
        self.counter_motion=np.zeros([2,1]) # vector with the number of samples that have been recorded for motion inference
        self.time=np.zeros([2,1])# time of data recorded for each human
        
        self.p0_input=np.zeros([2,1])
        self.p1_input=np.zeros([2,1])

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        #INFORMATION SIMULATED IN GAZEBO (DEMO2) OR ONLY IN THIS SCRIPT (DEMO1)
        self.position=np.zeros([1,3]) #[x,y,theta]
        self.input=np.zeros([2,1]) #w,v control signals
        self.operation=1 #["UV-C_treatment","moving_to_picker_location", "wait_for_command_to_approach", "approaching_to_picker","wait_for_command_to_move_away", "moving_away_from_picker"]

class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=0    
        self.human_command=0
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        #self.goal_index=0 #index of the human whom location is considered as goal
       
def safety_callback(safety_info):
    #print("SAFETY NEW DATA")
    if new_data[0]==0:
        hri.status=safety_info.hri_status
        hri.audio_message=safety_info.audio_message
        hri.safety_action=safety_info.safety_action
        hri.human_command=safety_info.human_command
        hri.critical_index=safety_info.critical_index   
        new_data[0]=1

def actor00_callback_d2(p1):
    #print("ACTOR00 NEW DATA")
    if new_data[1]==0:
        #Sensor type, time, counter, centroid are not required for the demo, thus they are always 0
        
        #Positions from gazebo
        pose=odometry(p1)
        #Human orientation
        #if abs(pose[2]-robot.position[2])>pi:
        #    human.orientation[0,0]=0 #always facing the robot
        #else:
        #    human.orientation[0,0]=1 #always giving the back the robot
        #Human Motion and distance
        time_new=time.time()-time_init
        distance_new=sqrt((robot.position[0]-pose[0])**2+(robot.position[1]-pose[1])**2)
        #to include the thorlvard dimensions
        distance_new=distance_new-1
        #if distance_new<0:
        #    distance_new=0
        human.speed[0,0]= (distance_new-human.distance[0,0])/(time_new-human.time[0])+robot.input[1]
        k=0
        if human.counter_motion[k]<n_samples: #while the recorded data is less than n_points                
            human.speed_buffer[k,int(human.counter_motion[k])]=human.speed[k,:]
            human.counter_motion[k]=human.counter_motion[k]+1
        else: #to removed old data and replace for newer data
            human.speed_buffer[k,0:n_samples-1]=human.speed_buffer[k,1:n_samples]
            human.speed_buffer[k,n_samples-1]=human.speed[k,:]
        human.distance[0,0]=distance_new
        human.time[0]=time_new
        ii=0
        if human.counter_motion[ii]>=n_samples:
            speed_mean=np.mean(human.speed_buffer[ii,:])
            if abs(speed_mean)>=0 and abs(speed_mean)<speed_threshold[0]: # if human is  mostly static
                human.motion[ii]=1
            elif abs(speed_mean)>=speed_threshold[0] and abs(speed_mean)<speed_threshold[1]: #if human is moving slowly
                if speed_mean<0:
                    human.motion[ii]=2
                else:
                    human.motion[ii]=4
            else: #if human is moving fast
                if speed_mean<0:
                    human.motion[ii]=3
                else:
                    human.motion[ii]=5
        else:
            human.motion[ii]=0
        #Human Area (using only y-position in global frame)
        #print("AREA ERROR",abs(pose[0]-robot.position[0]))
        if abs(pose[1]-robot.position[1])<=2:
            human.area[0,0]=2
        else:
            human.area[0,0]=0
        #Transform human_position from global frame to local frame
        human.position[0,:]=pose-robot.position
        new_data[1]=1

def actor01_callback_d2(p2):
    #print("ACTOR01 NEW DATA")
    if new_data[2]==0:
        #Sensor type, time, counter, centroid are not required for the demo, thus they are always 0
        
        #Positions from gazebo
        pose=odometry(p2)
        #Human orientation
        #print("pose",pose[2])
        #print("position",robot.position[2])
        #if abs(pose[2]-robot.position[2])>pi:
        #    human.orientation[1,0]=0 #always facing the robot
        #else:
        #    human.orientation[1,0]=1 #always giving the back the robot
        #Human Motion and distance
        time_new=time.time()-time_init
        distance_new=sqrt((robot.position[0]-pose[0])**2+(robot.position[1]-pose[1])**2)
        #to include the thorlvard dimensions
        distance_new=distance_new-1
        #if distance_new<0:
        #   distance_new=0
        human.speed[1,0]= (distance_new-human.distance[1,0])/(time_new-human.time[1])+robot.input[1]
        k=1
        if human.counter_motion[k]<n_samples: #while the recorded data is less than n_points                
            human.speed_buffer[k,int(human.counter_motion[k])]=human.speed[k,:]
            human.counter_motion[k]=human.counter_motion[k]+1
        else: #to removed old data and replace for newer data
            human.speed_buffer[k,0:n_samples-1]=human.speed_buffer[k,1:n_samples]
            human.speed_buffer[k,n_samples-1]=human.speed[k,:]
        human.distance[1,0]=distance_new
        human.time[1]=time_new
        ii=1
        if human.counter_motion[ii]>=n_samples:
            speed_mean=np.mean(human.speed_buffer[ii,:])
            if abs(speed_mean)>=0 and abs(speed_mean)<speed_threshold[0]: # if human is  mostly static
                human.motion[ii]=1
            elif abs(speed_mean)>=speed_threshold[0] and abs(speed_mean)<speed_threshold[1]: #if human is moving slowly
                if speed_mean<0:
                    human.motion[ii]=2
                else:
                    human.motion[ii]=4
            else: #if human is moving fast
                if speed_mean<0:
                    human.motion[ii]=3
                else:
                    human.motion[ii]=5
        else:
            human.motion[ii]=0
        #Human Area (using only y-position in global frame)
        if abs(pose[1]-robot.position[1])<=1:
            human.area[1,0]=2
        else:
            human.area[1,0]=0
        #Transform human_position from global frame to local frame
        human.position[1,:]=pose-robot.position
        new_data[2]=1


def robot_callback_d2(rob):
    #print("ROBOTS NEW DATA")
    if new_data[3]==0:
        pose=odometry(rob)
        robot.position=pose
        new_data[3]=1
        

def odometry(odom):
    # Generate a simplified pose
    pos = odom.pose.pose
    quat = pos.orientation
    # From quaternion to Euler
    #angles = tf.transformations.euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
    #theta = angles[2]
    theta=0
    theta=np.unwrap([theta])[0]
    pose= np.array([pos.position.x, pos.position.y, theta])
    return pose

def joy_callback(data):
    #print("JOY NEW DATA")
    global simulation
    buttons=data.buttons
    axes=data.axes  
    if np.shape(axes)[0]!=0:
        #Picker01
        if axes[0]!=0:
            human.p1_input[0]=axes[0]*picker_speed[0]
        if axes[1]!=0:
            human.p1_input[1]=axes[1]*picker_speed[1]
        #Picker00
        if axes[2]!=0:
            human.p0_input[0]=axes[2]*picker_speed[0]
        if axes[3]!=0:
            human.p0_input[1]=axes[3]*picker_speed[1]
    else: 
        pass
    
    if np.shape(buttons)[0]>0:
        if buttons[4]>0: #L1 to control picker01 gesture
            if buttons[0]>0: #square is two arms (approach) 
                human.posture[1,0]=1
            if buttons[1]>0: #triangle is right hand (stop)
                human.posture[1,0]=8
            if buttons[2]>0: #circle is two hands (move away)
                human.posture[1,0]=2
        elif buttons[6]>0: #R1 to control picker00 gesture
                if buttons[0]>0: #square is two arms (approach) 
                    human.posture[0,0]=1
                if buttons[1]>0: #triangle is right hand (stop)
                    human.posture[0,0]=8
                if buttons[2]>0: #circle is two hands (move away)
                    human.posture[0,0]=2
        elif buttons[3]>0: #X to control the simulation
            if simulation==0:
                simulation=1 #to start simulation
            if human.posture[0,0]!=0:
                human.posture[0,0]=0 #to reset human gesture
            if human.posture[1,0]!=0:
                human.posture[1,0]=0 #to reset human gesture
            if human.p1_input[0]!=0:
                human.p1_input[0]=0 #to reset human motion
            if human.p1_input[1]!=0:
                human.p1_input[1]=0 #to reset human motion
            if human.p0_input[0]!=0:
                human.p0_input[0]=0 #to reset human motion
            if human.p0_input[1]!=0:
                human.p0_input[1]=0 #to reset human motion
        if buttons[9]>0: #option to change the human orientation to back
            print("OPTIONS")
            human.orientation[1]=1
            human.orientation[0]=1
        if buttons[8]>0: #option to change the human orientation to front
            print("START")
            human.orientation[1]=0
            human.orientation[0]=0
            #if simulation==1:
            #    simulation=0
            #print("simulation",simulation)
    else:
        pass  
    

###############################################################################################
# Main Script

if __name__ == '__main__':
    time_init=time.time()       
    # Initialize our node       
    if demo==2:
        human=human_class()
        hri=hri_class()  
        robot=robot_class()
        #Setup ROS publiser
        pub_robot = rospy.Publisher('robot_info', robot_msg)
        rob_msg = robot_msg()
        pub_human = rospy.Publisher('human_info', human_msg)
        msg = human_msg()
        pub_picker00=rospy.Publisher('actor00/cmd_vel',geometry_msgs.msg.Twist, queue_size=10)
        pub_picker01=rospy.Publisher('actor01/cmd_vel',geometry_msgs.msg.Twist, queue_size=10)
        pub_thorvald=rospy.Publisher('/turtle1/cmd_vel',geometry_msgs.msg.Twist, queue_size=10)
        msg_control= geometry_msgs.msg.Twist()
        pub_hz=0.01 #publising rate in seconds
        rospy.init_node('demo_actuation',anonymous=True)
        # Setup and call subscription
        rospy.Subscriber('human_safety_info',hri_msg,safety_callback)  
        rospy.Subscriber('actor00/odom',nav_msgs.msg.Odometry,actor00_callback_d2)  
        rospy.Subscriber('actor01/odom',nav_msgs.msg.Odometry,actor01_callback_d2)   
        rospy.Subscriber('odometry/gazebo',nav_msgs.msg.Odometry,robot_callback_d2)  
        rospy.Subscriber('joy',Joy,joy_callback)  
        #Rate setup
        rate = rospy.Rate(1/pub_hz) # ROS publishing rate in Hz
        while not rospy.is_shutdown():	      
            main_counter=main_counter+1  
            print("main_counter",main_counter)
            if simulation==1:
                ############################################################################################
                if new_data[0]==1 or new_data[1]==1 or new_data[2]==1 or new_data[3]==1:
                    if robot.operation==1: #if robot is moving to picker location 
                        #print("ROBOT OPERATION 1")
                        robot.input[1]=robot_speed[0] #normal speed
                        if hri.status==2: #if distance <=3.6m
                           #print("DISTANCE <3.6m")
                            robot.operation=2 #wait for command to approach
                            robot.input[1]=robot_speed[2] #stop
                        if hri.human_command==1: #if human command is "approach"
                            robot.operation=3 
                            robot.input[1]=robot_speed[1] #reduced speed
                        if hri.human_command==2: #if human command is "move_away"
                            robot.operation=5
                            robot.input[1]=-robot_speed[0] #normal speed, reverse
                        if hri.human_command==3: #if human command is "stop"
                            robot.operation=2 #wait for command to approach
                            robot.input[1]=robot_speed[2] #stop
                    
                    elif robot.operation==2: #if robot is waiting for a human command to approach
                        robot.input[1]=robot_speed[2] #stop
                        if hri.human_command==1: #if human command is "approach"
                            robot.operation=3 
                            robot.input[1]=robot_speed[1] #reduced speed
                        if hri.human_command==2: #if human command is "move_away"
                            robot.operation=5
                            robot.input[1]=-robot_speed[0] #normal speed, reverse
                    elif robot.operation==3: #if robot is approaching to picker (already identified)
                        robot.input[1]=robot_speed[1] #reduced speed
                        if hri.status==3: #if distance <=1.2m
                            robot.operation=4 #wait for command to move away
                            robot.input[1]=robot_speed[2] #stop
                        if hri.human_command==2: #if human command is "move_away"
                            robot.operation=5
                            robot.input[1]=-robot_speed[0] #normal speed, reverse
                        if hri.human_command==3: #if human command is "stop"
                            robot.operation=4 #wait for command to move away
                            robot.input[1]=robot_speed[2] #stop
                    elif robot.operation==4: #if robot is waiting for a human command to move away
                        robot.input[1]=robot_speed[2] #stop
                        if hri.human_command==1: #if human command is "approach"
                            robot.operation=3 
                            robot.input[1]=robot_speed[1] #reduced speed
                        if hri.human_command==2: #if human command is "move_away"
                            robot.operation=5
                            robot.input[1]=-robot_speed[0] #normal speed, reverse
                    elif robot.operation==5: #if robot is moving away from the picker (it can be after collecting tray or because of human command)
                        robot.input[1]=-robot_speed[0] #normal speed, reverse
                        if hri.human_command==1: #if human command is "approach"
                            robot.operation=3 
                            robot.input[1]=robot_speed[1] #reduced speed
                        if hri.human_command==3: #if human command is "stop"
                            robot.operation=2 #wait for command to approach
                            robot.input[1]=robot_speed[2] #stop
                    else: #uv-c treatment
                        robot.operation=0
                        robot.input[1]=robot_speed[0] #normal speed
                    
                    #To recalculate reduced_speed according to the distance between robot and human
                    if robot.input[1]==robot_speed[1] or hri.safety_action==1:
                        dist=human.distance[hri.critical_index,0]
                        robot.input[1]=0.138*dist #0m/s at 0m and 0.5m/s at 3.6m    
                        #if robot.input[1]<0:
                        #    robot.input[1]=robot_speed[2] #stop
                        if robot.input[1]>=robot_speed[0]:
                            robot.input[1]=robot_speed[0] #normal speed limitation
                        if robot.input[1]<=-robot_speed[0]:
                            robot.input[1]=-robot_speed[0] #normal speed limitation                   
                    #To make the robot stop when a safety_stop is required or when the robot is waiting for a human command
                    if hri.safety_action>=2:
                        robot.input[1]=robot_speed[2] #stop
                    
                    if new_data[0]==1:    
                        new_data[0]=0
                    if new_data[1]==1:
                        new_data[1]=0
                    if new_data[2]==1:
                        new_data[2]=0
                    if new_data[3]==1:
                        new_data[3]=0
            ############################################################################################
            #print("human_position",human.position)
            #print("human_position",list(human.position))
            #print("human_distance",list(human.distance))
            #print("human_posture",list(human.posture))
            print("human_orientation",list(human.orientation))
            #print("human_area",list(human.area))
            #print("critical_index",hri.critical_index)
            #print("robot_position",robot.position)
            #print("robot_operation",robot.operation)
            #Publish Robot new operation
            rob_msg.operation=robot.operation
            pub_robot.publish(rob_msg)        
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
            #Publish Gazebo commands
            msg_control.linear.x = human.p0_input[1][0]
            msg_control.angular.z = human.p0_input[0][0]
            pub_picker00.publish(msg_control)
            msg_control.linear.x = human.p1_input[1][0]
            msg_control.angular.z = human.p1_input[0][0]
            pub_picker01.publish(msg_control)
            msg_control.linear.x = robot.input[1][0]
            msg_control.angular.z = robot.input[0][0]
            pub_thorvald.publish(msg_control)
            rate.sleep() #to keep fixed the publishing loop rate
   