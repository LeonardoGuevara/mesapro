#!/usr/bin/env python

#required packages
import rospy
import yaml
#from topological_navigation.tmap_utils import *
from geometry_msgs.msg import Twist
from mesapro.msg import human_msg, hri_msg
import threading # Needed for Timer
#######################################################################################################################
#GENERAL PURPUSES VARIABLES
pub_hz=0.01 #main loop frequency
#Setup ROS publiser
cmd_pub = rospy.Publisher('/nav_vel', Twist, queue_size=1)
cmd_vel = Twist()
#Importing global parameters from .yaml file
default_config_direct="/home/leo/rasberry_ws/src/mesapro/config/"
config_direct=rospy.get_param("/gesture_control/config_direct",default_config_direct) #you have to change /gesture_control/ if the node is not named like this
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
han_distances=parsed_yaml_file.get("human_safety_config").get("han_distances",[3.6,1]) #distances used when robot is "approaching to picker"
#########################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        self.vel=[0,0,0] # velocity vector [linear.x,linear.y,orientation.z]
        
class human_class:
    def __init__(self): 
        self.pos_x=0
        self.pos_y=0 
        self.n_human=0
        
    def human_callback(self,human_info):
        self.n_human=human_info.n_human
        if hri.critical_index<=self.n_human-1 and self.n_human!=0:
            self.pos_x=human_info.position_x[hri.critical_index]
            self.pos_y=human_info.position_y[hri.critical_index]

class hri_class:
    def __init__(self): #It is done only the first iteration
        self.safety_action=7 # no safety action as initial condition     
        self.human_command=0 # no human command as initial condition
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        self.han_start_dist=han_distances[0]    # Human to robot Distance at which the robot starts to slow down
        self.han_stop_dist=han_distances[1]     # Human to robot Distance at which the robot must stop
        self.time_without_msg=rospy.get_param("/gesture_control/time_without_msg",5) # Maximum time without receiving safety messages
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        # Dynamically reconfigurable parameters
        self.align_tolerance= 0.2               # tolerance to consider the robot aligned to the human (in metres)
        self.x_speed_limit=0.3                  # Maximum speed on the X axis
        self.y_speed_limit=0.3                  # Maximum speed on the Y axis
        self.turning_speed_limit=0.1            # Maximum turning speed
        self.turning_kp=0.5                     # Gains for tunning speed control
        ########################################################################################################### 

    def get_speed(self, error_x,error_y,command):
        #Check if human detected is located in front of the robot or at the back
        speed=[0,0,0] #initial values, [linear.x,linear.y,angular.z]
        if error_x<0:
            backwards_mode=True
        else:
            backwards_mode=False
        
        if command==4: #execute if safety action is "move forwards" or ""move towards the human"
            #Check if the robot face which detected the human is aligned to the human position, if not, rotate with radius 0 till make error_y converge within the aligned threshold
            if backwards_mode==False:
                speed[2]=self.turning_kp*error_y
            else:
                speed[2]=-self.turning_kp*error_y
            #Limit turning speed
            if speed[2]>self.turning_speed_limit:
                speed[2]=self.turning_speed_limit
            if speed[2]<-self.turning_speed_limit:
                speed[2]=-self.turning_speed_limit  
            #Start moving towards the human only if robot is within the aligned threshold
            if abs(error_y)<self.align_tolerance:
                dist=abs(error_x) 
                if dist <= self.han_start_dist:
                    slowdown_delta = self.han_start_dist - self.han_stop_dist
                    current_percent = (dist - self.han_stop_dist) / slowdown_delta
                    if current_percent >0:
                        #print("Limiting speed")
                        speed[0] = (current_percent*self.x_speed_limit)
                    else:
                        #print("stop")
                        speed[0] = 0.0
                else:
                    speed[0] = self.x_speed_limit
        if command==5: #execute if gesture command is "move backwards"
            speed[0] = -self.x_speed_limit
        if command==6: #execute if gesture command is "move right"
            speed[1] = self.y_speed_limit
        if command==7: #execute if gesture command is "move left"
            speed[1] = -self.y_speed_limit
        if backwards_mode==True: #In any case, if the robot is in backward mode, then change the linear speeds signs
            speed[0] = -speed[0]
            speed[1] = -speed[1]
        print("SPEED",speed)
        ###################################################################################################
        return speed
    
    
    def safety_callback(self,safety_info):
        self.safety_action=safety_info.safety_action 
        self.critical_index=safety_info.critical_index
        self.human_command=safety_info.human_command
        #print("Safety message received")
        self.timer_safety.cancel()
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
               
    def safety_timeout(self):
        print("No safety message received in a long time")
        self.safety_action=4 # to stop the robot
       
###############################################################################################
# Main Script
if __name__ == '__main__':   
    hri=hri_class()  
    human=human_class()
    robot=robot_class()
    # Initialize our node    
    rospy.init_node('footpath_gesture_control',anonymous=True)
    rospy.Subscriber('human_safety_info',hri_msg,hri.safety_callback)  
    rospy.Subscriber('human_info',human_msg,human.human_callback)  
    #Rate setup
    rate = rospy.Rate(1/pub_hz)  # main loop frecuency in Hz
    while not rospy.is_shutdown():	 
        if hri.safety_action==6: #Excecute only if gesture control at footpaths is needed.
            robot.vel=hri.get_speed(human.pos_x,human.pos_y,hri.human_command)  
            cmd_vel.linear.x = robot.vel[0]
            cmd_vel.linear.y = robot.vel[1]
            cmd_vel.angular.z = robot.vel[2]
            cmd_pub.publish(cmd_vel)     
        #elif hri.safety_action==3 or hri.safety_action==4: #if safety action is make the robot stop
        #    cmd_vel.linear.x = 0
        #    cmd_vel.linear.y = 0
        #    cmd_vel.angular.z = 0
        #    cmd_pub.publish(cmd_vel)  
        rate.sleep() #to keep fixed the publishing loop rate
