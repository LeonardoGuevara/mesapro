#!/usr/bin/env python

#required packages
import rospy
from tf.transformations import euler_from_quaternion
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import os
import yaml
from std_msgs.msg import String
from topological_navigation.route_search2 import TopologicalRouteSearch2
from topological_navigation.tmap_utils import *
from mesapro.msg import human_msg, hri_msg, robot_msg
##########################################################################################

#Setup ROS publiser
pub_safety = rospy.Publisher('human_safety_info', hri_msg)
safety_msg = hri_msg()
#General purposes variables
main_counter=0
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        #declaration and initial values of important variables
        self.pos_x=0 #X
        self.pos_y=0 #Y
        self.pos_theta=0 #orientation
        self.action=0 #initial value
        self.current_goal_x=0 #current robot goal X
        self.current_goal_y=0 #current robot goal Y
        self.current_goal="Unknown"  #name of current robot goal
        self.current_goal_info="Unknown"   #initial condition
        self.polytunnel=False #to know if robot is moving inside the polytunnel or not
        self.final_goal="Unknown" #name of final robot goal
        
    def robot_info_callback(self,robot_info):
        if robot_info.current_node!="Unknown":
            parent=topo_map.rsearch.get_node_from_tmap2(robot_info.current_node)
            polytunnel = False #initial assuption that the robot is outside the polytunnel nodes, then the human commands are not allowed
            for edge in parent["node"]["edges"]:
                if edge["action"]=="row_traversal":
                    polytunnel = True # If at least an edge is connected with the polytunnels nodes, then the human commands are allowed
                    break
            self.action=robot_info.action
            self.polytunnel=polytunnel
            self.current_goal=robot_info.current_node
            self.current_node_x=parent["node"]["pose"]["position"]["x"]    
            self.current_node_y=parent["node"]["pose"]["position"]["y"]           
            self.current_goal_info=parent
            self.final_goal=robot_info.goal_node
            
    
    def robot_pose_callback(self,pose):
        self.pos_x=pose.position.x
        self.pos_y=pose.position.y
        quat = pose.orientation    
        # From quaternion to Euler
        angles = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        theta = angles[2]
        self.pos_theta=np.unwrap([theta])[0]

        
class human_class:
    def __init__(self): 
        self.posture=[0]
        self.posture_prob=[0]
        self.motion=[0] 
        self.position_x=[0] 
        self.position_y=[0] 
        self.centroid_x=[0]
        self.centroid_y=[0]
        self.distance=[0]
        self.orientation=[0]
        self.area=[0]
        self.sensor=[0]
        self.sensor_t0=[0]
        self.sensor_t1=[0]
        self.sensor_c0=[0]
        self.sensor_c1=[0]
        self.pos_global_x=0 # human pose transformed to global frame, based on robot pose
        self.pos_global_y=0 # human pose transformed to global frame, based on robot pose
        
    def human_callback(self,human_info):
        self.posture=human_info.posture
        self.posture_prob=human_info.posture_prob
        self.motion=human_info.motion   
        self.position_x=human_info.position_x
        self.position_y=human_info.position_y
        self.centroid_x=human_info.centroid_x
        self.centroid_y=human_info.centroid_y
        self.distance=human_info.distance
        self.orientation=human_info.orientation
        self.area=human_info.area
        self.sensor=human_info.sensor
        self.sensor_t0=human_info.sensor_t0
        self.sensor_t1=human_info.sensor_t1
        self.sensor_c0=human_info.sensor_c0
        self.sensor_c1=human_info.sensor_c1  
        [self.pos_global_x,self.pos_global_y]=self.human_pose_global()
        
    def human_pose_global(self):
        #assuming human global pose is always aligned to the robot orientation but with an offset of distance
        dist=self.distance[hri.critical_index]
        ########################################################################################
        #To consider the thorvald dimensions and the lidars/cameras locations respect to gps
        dist=dist+1 #because of extra 1m used to compute distance in virtual_picker_simulation.py
        ###########################################################################################
        area=self.area[hri.critical_index]
        r_theta=robot.pos_theta
        ###########################################################################################
        #assuming the local x-axis is aligned to the robot orientation
        if area<=4: #if human was detected in front of the robot (areas 0 to 4)
            new_theta=r_theta 
        elif area>=5:
            new_theta=r_theta+pi
        pos_y=(sin(new_theta)*dist)+robot.pos_y
        pos_x=(cos(new_theta)*dist)+robot.pos_x
        ##############################################################################################
        return pos_x,pos_y

class map_class:
    def __init__(self): #It is done only the first iteration
        self.map_received = False
    
    def MapCallback(self, msg):
        """
         This Function updates the Topological Map everytime it is called
        """
        self.lnodes = yaml.safe_load(msg.data)
        self.rsearch = TopologicalRouteSearch2(self.lnodes)
        self.map_received = True
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.audio_message=0
        self.safety_action=5 #initial condition     
        self.human_command=0
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        self.risk=False #it is true if the detected human is considered in risk, used to avoid unnecesary stops with human who is not occluding the robot path
        self.new_goal="none" #name of the new final goal
        self.operation="logistics"   # logistics or UVC
        self.aligned=False #True if is the human is aligned to the robot front or back, False if the human is on the sides of the robot   
        self.safe_cond=False #True if the human is static and facing the robot, False if not satisfying this safety condition

    def critical_human_selection(self):  
        polytunnel=robot.polytunnel
        posture=human.posture
        #posture_prob=human.posture_prob
        dist=human.distance
        area=human.area
        sensor=human.sensor
        orientation=human.orientation
        motion=human.motion
        closest_dist=1000 #initial value
        closest_index=0
        n_human=len(dist)
        gesture_priority=False #initial value
        distance_priority=False #initial value
        aligned=False #initial value
        for k in range(0,n_human):
            if dist[k]<=closest_dist:
                closest_index=k
                closest_dist=dist[k]
            if closest_dist<=1.2 and aligned==True: 
                distance_priority=True
            #aligned condition inside the polytunnel includes areas 2 and 7, and outside polytunnel includes 1,2,3 and 6,7,8
            if ((area[k]==2 or area[k]==7) and polytunnel==True) or (((area[k]>=1 and area[k]<=3) or (area[k]>=6 and area[k]<=8)) and polytunnel==False): 
                if aligned==True:
                    if dist[aligned_index]>dist[k]:
                        aligned_index=k
                else:
                    aligned_index=k
                aligned=True
            if (posture[k]==1 or posture[k]==8 or posture[k]==2) and aligned==True: 
                if gesture_priority==True:
                    if dist[gesture_index]>dist[k]:
                        gesture_index=k
                else:
                    gesture_index=k
                gesture_priority=True
            
        #Critical index selection
        if distance_priority==True: # 1rst priority is human within 1.2m in critical areas
            critical_index=closest_index
        elif gesture_priority==True: # 2nd priority is human at any distance performing gestures in critical areas
            critical_index=gesture_index
        elif aligned==True: # if none of the previos conditions ara satistifed, then the closest human but aligned is considered
            critical_index=aligned_index
        else: # if none of the previos conditions are satisfied, then only the closest human is considered
            critical_index=closest_index
        
        #RISK INFERENCE
        if self.operation=="logistics": #if robot operation is logistics
            if robot.current_goal=="Unknown" or robot.final_goal=="Unknown": #if the robot doesn't start moving yet
                risk=False
                safe_cond=False
            else: 
                [risk,safe_cond]=self.risk_analysis(area[critical_index],sensor[critical_index],orientation[critical_index],motion[critical_index],aligned) #risk:=occlusion
        else: #for uv-c treatment
            risk=True #there is always a risk just for being detected
            safe_cond=False
        
        self.critical_index=critical_index
        self.risk=risk
        self.aligned=aligned
        self.safe_cond=safe_cond
    
    def risk_analysis(self,area,sensor,orientation,motion,aligned):
        #assuming that a human is in risk when he/she is closer to the current robot goal then the robot
        h_y=human.pos_global_y
        r_y=robot.pos_y
        goal = topo_map.rsearch.get_node_from_tmap2(robot.final_goal)
        g_y=goal["node"]["pose"]["position"]["y"]
        h_dist=abs(h_y-g_y)
        r_dist=abs(r_y-g_y)
        if aligned==True: 
            if h_dist<=r_dist:
                risk=True
            else:
                risk=False
        else: #if human is on the side of the robot 
            risk=False
        if motion==2 or (sensor!=1 and orientation==1): #In case the human is not static, or he/she is not facing the robot
            safe_cond=False
        else:
            safe_cond=True    
        return risk, safe_cond

    def find_new_goal(self):
        #Asumming that this function is only called when robot is along the rows, not valid outside the polytunnel
        #Asumming the "y" component of row nodes is always greater in upper nodes than lower nodes
        parent=robot.current_goal_info
        r_y=robot.pos_y
        #Human position in global frame
        h_y=human.pos_global_y
        #new motion direction along the row
        if self.safety_action==1: #approach
            if r_y<=h_y:
                goal_dir="up" # search for the upper node in the row
            else:
                goal_dir="down" #search for the lowernode in the row
        else: #move away
            if r_y<=h_y:
                goal_dir="down" #search for the lowernode in the row
            else:
                goal_dir="up" # search for the upper node in the row
        #iterative goal search along the row
        not_goal = True
        while not_goal: 
            children = self.get_connected_nodes_tmap(parent)
            for child_name in children:
                child = topo_map.rsearch.get_node_from_tmap2(child_name)
                child_y=child["node"]["pose"]["position"]["y"]
                parent_y=parent["node"]["pose"]["position"]["y"]
                if goal_dir=="down": 
                    if child_y<=parent_y:
                        parent=child    
                else: # UP
                    if child_y>=parent_y:
                        parent=child
            for edge in parent["node"]["edges"]:
                if edge["action"]=="move_base":
                    not_goal = False
                    break
        goal=parent["node"]["name"]
        return goal
    
    
    def get_connected_nodes_tmap(self, node):
        children=[]
        for edge in node["node"]["edges"]:
            children.append(edge["node"])
        return children
    
    def decision_making(self):
        ##NO RISK OF HUMAN INJURIES AS INITIAL ASUMPTION
        self.status=0 #no human/human but without risk
        self.human_command=0 #no human command
        self.critical_index=0 #by default
        if robot.action==0 or robot.action==2: # robot is moving to an initial goal or moving away from the picker
            self.safety_action=5 # no safety action / keep the previous robot action
            self.audio_message=0 # no message 
            self.new_goal=robot.final_goal # the current goal is not changed             
        elif robot.action==1: # robot was approaching
            self.safety_action=4 # make it stop for safety purposes, waiting for a human order 
            self.audio_message=2 # message to ask the human for new order
            self.new_goal=robot.final_goal # the current goal is not changed
        elif robot.action==3: # if goal was paused while moving to a goal or moving away from the picker
            self.audio_message=0 # no message
            self.safety_action=0 # restart operation making the robot moving to the current goal
            self.new_goal=robot.final_goal # the current goal is not changed
        elif robot.action==4: #if robot is waiting for human order
            self.safety_action=5 # no safety action / keep the previous robot action
            self.audio_message=2 # message to ask the human for new order
            self.new_goal=robot.final_goal # the current goal is not changed
        ## HOW TO UPDATE ROBOT ACTION IN CASE OF HRI
        if len(human.sensor)>1 or (len(human.sensor)==1 and (human.motion[0]+human.centroid_x[0]+human.centroid_y[0]+human.posture[0]+human.position_x[0]+human.position_y[0]+human.distance[0])!=0): #execute only if at least a human is detected     
            #Critical human selection
            self.critical_human_selection()
            #Distance between the closest human detected and the robot
            distance=human.distance[self.critical_index]    
            print("DISTANCE",distance)
            ###UV-C treatment#######################################
            if self.operation!="logistics": #during UV-C treatment
                self.human_command=0 #no human command expected during uv-c treatment
                if self.risk==True: #only execute if there is risk of producing human injuries 
                    if robot.action==0: #if robot is moving to goal   
                        if distance>10: #if human is above 10m from the robot
                            self.status=2 #risky HRI
                            self.safety_action=5  # keep the previous robot action
                        else: #if human is within 10m
                            self.status=3 #dangerous HRI
                            self.safety_action=4 # stop UV-C treatment and wait for a new human command to restart
                        self.audio_message=1 #UVC danger message
                        self.new_goal=robot.final_goal # the current goal is not changed
                    else: #if robot is static        
                        if distance>10: #if human is above 10m from the robot
                            self.status=2 #risky HRI
                            self.safety_action=5 # keep the previous robot action
                        else: #if human is within 10m
                            self.status=3 #dangerous HRI
                            self.safety_action=5 # still waiting for a new human command to restart
                        self.audio_message=1 #UVC danger message
                        self.new_goal=robot.final_goal # the current goal is not changed
            ###LOGISTICS###############################################
            else:
                ##IN CASE THE HUMAN PERFORMS BODY GESTURES
                #In case the picker wants the robot to approch to him/her , only valid inside polytunnels and above 1.2m 
                if human.sensor[self.critical_index]!=1 and human.posture[self.critical_index]==1 and robot.polytunnel==True and distance>1.2 and self.aligned==True: #picker is ordering the robot to approach (using both arms)
                    self.human_command=1 #make the robot approach to the picker from this point i.e robot.operation=1
                    self.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                    self.safety_action=1 # to make the robot approach to the picker
                    self.new_goal=hri.find_new_goal()
                #In case the picker wants the robot to stop 
                elif human.sensor[self.critical_index]!=1 and human.posture[self.critical_index]==8 and self.aligned==True:  #picker is ordering the robot to stop (using right hand)
                    self.human_command=3 #make the robot stop 
                    self.audio_message=2 # start a message to ask the picker for a new order to approach/move away or move to a new goal
                    self.safety_action=4 # make the robot stop and wait for a new command to restart operation
                    self.new_goal=robot.final_goal # the current goal is not changed
                #In case the picker wants the robot to move away, only valid inside polytunnels
                elif human.sensor[self.critical_index]!=1 and human.posture[self.critical_index]==2 and robot.polytunnel==True and self.aligned==True:  #picker is ordering the robot to move away (using both hands)
                    self.human_command=2 #make the robot move away
                    self.audio_message=0 #no message
                    self.safety_action=2 # to make the robot move away from the picker
                    self.new_goal=hri.find_new_goal()
                else:
                    self.human_command=0 # no human gesture                  
                
                ##RISK LEVEL
                if distance>3.6: #if human is above 3.6m and robot is on normal operation
                    self.status=1 # safety HRI
                elif distance>1.2 and distance<=3.6: #if human is between 1.2-3.6m
                    self.status=2 # risky HRI
                    #self.status=3 #dangerous HRI
                else: #if human is within 1.2m
                    self.status=3 # dangerous HRI
                    
                if robot.action==0 or robot.action==2: #if robot is moving to an original goal or moving away from the picker
                    if self.status==1: #if human is above 3.6m and robot is on normal operation
                        if self.human_command==0: #if human is not performing any gesture
                            self.audio_message=0 #no message
                            self.safety_action=5 # keep the previous robot action
                            self.new_goal=robot.final_goal # the current goal is not changed
                    elif self.status==2: #if human is between 1.2-3.6m
                        if self.human_command==0: #if human is not performing any gesture
                            if self.risk==True: # if there is chance to produce injuries
                                self.safety_action=3 # pause operation and continue when human has move away 
                                self.audio_message=3 # message to ask the human for free space to continue moving
                                self.new_goal=robot.final_goal # the current goal is not changed
                            else:
                                self.audio_message=0 #no message
                                self.safety_action=5 # keep the previous robot action 
                                self.new_goal=robot.final_goal # the current goal is not changed
                    
                    else: #if human is within 1.2m or moving or is not facig the robot
                        if self.human_command!=2:  # if picker is not ordering the robot to move away (using both hands)
                            if self.risk==True: # if there is chance to produce injuries
                                self.safety_action=3 # pause operation and continue when human has move away 
                                self.audio_message=3 # message to ask the human for free space to continue moving
                                self.new_goal=robot.final_goal # the current goal is not changed
                            else:
                                self.audio_message=0 #no message
                                self.safety_action=5 # keep the previous robot action
                                self.new_goal=robot.final_goal # the current goal is not changed
                elif robot.action==1: #if robot is approaching to the picker
                    if self.status==1: #if human is above 3.6m 
                        if self.human_command==0: #if human is not performing any gesture
                            self.audio_message=4 # alert to make the picker aware of the robot approaching to him/her
                            self.safety_action=5 # keep it approaching to the picker
                            self.new_goal=robot.final_goal # the current goal is not changed
                    elif self.status==2: #if human is within 1.2m to 3.6m
                        if self.human_command==0 : #if human is not performing any gesture 
                            if self.safe_cond==False and self.risk==True: # if safety condition is not safisfied and if there is risk of producing injuries
                                self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                                self.audio_message=2 # message to ask the human for new order
                                self.new_goal=robot.final_goal # the current goal is not changed
                            else: #if safety condition is satisfied or there is not risk of producing injuries
                                self.audio_message=4 # alert to make the picker aware of the robot approaching to him/her
                                self.safety_action=5 # keep it approaching to the picker
                                self.new_goal=robot.final_goal # the current goal is not changed
                    else: #if human is within 1.2m  or moving or is not facig the robot
                        if self.human_command!=2:  # if picker is not ordering the robot to move away (using both hands) 
                            if self.risk==True: # if there is chance to produce injuries
                                self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                                self.audio_message=2 # message to ask the human for new order
                                self.new_goal=robot.final_goal # the current goal is not changed
                            else:
                                self.audio_message=4 # alert to make the picker aware of the robot approaching to him/her
                                self.safety_action=5 # keep it approaching to the picker
                                self.new_goal=robot.final_goal # the current goal is not changed
                elif robot.action==3: #if robot is in pause mode, waiting till the human is occluding to continue moving
                    if self.status==1: #if human is above 3.6m and robot is on normal operation
                        if self.human_command==0 : #if human is not performing any gesture
                            self.audio_message=0 # no message
                            self.safety_action=0 # restart operation making the robot moving to the current goal
                            self.new_goal=robot.final_goal # the current goal is not changed
                    else: #if human is within 3.6m
                        if self.human_command!=2:  # if picker is not ordering the robot to move away (using both hands)
                            self.safety_action=5 # still in pause operation 
                            self.audio_message=3 # message to ask the human for free space to continue moving
                            self.new_goal=robot.final_goal # the current goal is not changed
                elif robot.action==4: #if robot is waiting for a new human command
                    if self.status==1 or self.status==2: #if human is above 1.2m 
                        if self.human_command==0: #if human is not performing any gesture
                            self.safety_action=5 # still waiting for human order 
                            self.audio_message=2 # message to ask the human for new order
                            self.new_goal=robot.final_goal # the current goal is not changed
                    else: #if human is within 1.2m or moving or is not facig the robot
                        if self.human_command!=2:  # if picker is not ordering the robot to move away (using both hands)
                            self.safety_action=5 # still waiting for another human order 
                            self.audio_message=2 # message to ask the human for new order
                            self.new_goal=robot.final_goal # the current goal is not changed
                    
 
###############################################################################################
# Main Script

if __name__ == '__main__':   
    hri=hri_class()  
    human=human_class()
    robot=robot_class()
    topo_map=map_class()
    # Initialize our node    
    rospy.init_node('human_safety_system',anonymous=True)
    rospy.Subscriber("/topological_map_2", String, topo_map.MapCallback)
    #Waiting for the Topological Map...
    #while not topo_map.map_received:
    #    rospy.sleep(rospy.Duration.from_sec(0.05))
    rospy.Subscriber('human_info',human_msg,human.human_callback)  
    rospy.Subscriber('robot_info',robot_msg,robot.robot_info_callback)  
    rospy.Subscriber('/robot_pose', Pose, robot.robot_pose_callback)  
    robot.base_frame = rospy.get_param("row_traversal/base_frame", "base_link")
    #Rate setup
    rate = rospy.Rate(1/0.01) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	      
        main_counter=main_counter+1   
        hri.decision_making()
        #print("status",hri.status)
        print("##############################################################")
        print("risk",hri.risk)
        print("aligned",hri.aligned)
        print("critical_index",hri.critical_index)
        print("human_command",hri.human_command)
        print("safety_action",hri.safety_action)
        print("robot_action",robot.action)
        print("final_goal",robot.final_goal)
        ############################################################################################
        #Publish SAFETY SYSTEM MESSAGES        
        safety_msg.hri_status = hri.status
        safety_msg.audio_message = hri.audio_message
        safety_msg.safety_action = hri.safety_action
        safety_msg.human_command = hri.human_command
        safety_msg.critical_index = hri.critical_index
        safety_msg.new_goal= hri.new_goal
        safety_msg.operation_mode = hri.operation
        pub_safety.publish(safety_msg)
       
        rate.sleep() #to keep fixed the publishing loop rate
