#!/usr/bin/env python

#required packages
import rospy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import yaml
from std_msgs.msg import String
from topological_navigation.route_search2 import TopologicalRouteSearch2
#from topological_navigation.tmap_utils import *
from mesapro.msg import human_msg, hri_msg, robot_msg
import threading # Needed for Timer
#######################################################################################################################
#GENERAL PURPUSES VARIABLES
pub_hz=0.01 #main loop frequency
#Setup ROS publiser
pub_safety = rospy.Publisher('human_safety_info', hri_msg,queue_size=1)
safety_msg = hri_msg()
#Importing global parameters from .yaml file
default_config_direct="/home/leo/rasberry_ws/src/mesapro/config/"
config_direct=rospy.get_param("/hri_safety_system/config_direct",default_config_direct) #you have to change /hri_safety_system/ if the node is not named like this
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
collision_risk_dist=parsed_yaml_file.get("human_safety_config").get("collision_risk_distances",[3.6,1.2]) #Human to robot distances (m) used to determine the HRI risk during logistics 
uvc_risk_dist=parsed_yaml_file.get("human_safety_config").get("uvc_risk_distances",[10,7]) #Human to robot distances (m) used to determine the HRI risk during uvc treatment
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        self.pos_x=0 #X
        self.pos_y=0 #Y
        self.pos_theta=0 #orientation
        self.action=4 #initial value, "waiting for human command" as initial condition
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
        self.motion=[0] 
        self.distance=[0]
        self.orientation=[0]
        self.area=[0]
        self.sensor=[0]
        self.detection=False #true if a human is detected
        self.thermal_detection=False #true if thermal detection flag is also True
        #Safety Timer
        self.perception_msg=True      # if "False", human perception messages are not published, by default is "True"
        self.time_without_msg=rospy.get_param("/hri_safety_system/time_without_msg",5) # Maximum time without receiving human perception messages
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        
    def human_callback(self,human_info):
        self.posture=human_info.posture
        self.motion=human_info.motion   
        self.distance=human_info.distance
        self.orientation=human_info.orientation
        self.area=human_info.area
        self.sensor=human_info.sensor
        self.thermal_detection=human_info.thermal_detection
        if human_info.n_human>0:
            self.detection=True #there is at least one human detected
        else:
            self.detection=False
        #print("Human perception message received")
        self.timer_safety.cancel()
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        self.perception_msg=True
  
    def safety_timeout(self):
        print("No human perception message received in a long time")
        self.perception_msg=False #to alert that human perception system is not publishing 
        
    def human_pose_global(self,dist,area,r_pos_x,r_pos_y,r_pos_theta):
        #################################################################################################################
        #assuming human global pose is always aligned to the robot orientation but with an offset equal to the distance
        #################################################################################################################
        #To compensate the thorvald dimensions and the lidars/cameras locations (local frame)
        #dist=dist+1 #because of extra 1m used to compute distance in virtual_picker_simulation.py
        #######################################################################################################
        #assuming the local x-axis is aligned to the robot orientation
        if area<=4: #if human was detected in front of the robot (areas 0 to 4)
            new_theta=r_pos_theta 
        elif area>=5:
            new_theta=r_pos_theta+pi
        pos_y=(sin(new_theta)*dist)+r_pos_y
        pos_x=(cos(new_theta)*dist)+r_pos_x
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
        self.status=0 #initial condition 
        self.audio_message=0 #initial condition 
        self.safety_action=5 #initial condition     
        self.human_command=0 #initial condition 
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        self.critical_dist=0 #distance of the critical human
        self.risk=False #it is true if the detected human is considered in risk, used to avoid unnecesary stops with human who is not occluding the robot path
        self.new_goal="none" #name of the new final goal
        self.operation=rospy.get_param("/hri_safety_system/operation_mode","logistics") #it can be "UVC" or "logistics"
        self.safe_cond=False #True if the human is static and facing the robot, False if not satisfying this safety condition
        

    def critical_human_selection(self,polytunnel,posture,dist,area,sensor,orientation,motion,current_goal,final_goal,r_pos_x,r_pos_y,r_pos_theta):  
        ##################################################################################################################
        #assuming aligned condition. Inside the polytunnel includes areas 2 and 7, and outside polytunnel includes 1,2,3 and 6,7,8
        #critical human selection logic is: 1st human within 1.2m, 2nd human performing gesture, 3rd human centered, 4rd closest human   
        ######################################################################################################################
        n_human=len(dist)
        aligned_old=False #initial value
        gesture_old=False #initial value
        danger_old=False #initial value
        centered_old=False #initial value
        command_old=0 #initial value
        critical_index=0 #initial value
        
        if ((area[critical_index]==2 or area[critical_index]==7) and polytunnel==True) or (((area[critical_index]>=1 and area[critical_index]<=3) or (area[critical_index]>=6 and area[critical_index]<=8)) and polytunnel==False):
            aligned_old=True
            if area[critical_index]==2 or area[critical_index]==7:
                centered_old=True
        if dist[critical_index]<=collision_risk_dist[1] and aligned_old==True:
            danger_old=True 
        if (sensor[critical_index]!=1 and posture[critical_index]==1 and danger_old==False and aligned_old==True):
            gesture_old=True
            if polytunnel==True:
                command_old=1 #approach
            else:
                command_old=4 #move forwards
        elif (sensor[critical_index]!=1 and posture[critical_index]==8 and aligned_old==True):
            gesture_old=True
            command_old=3
        elif (sensor[critical_index]!=1 and posture[critical_index]==2 and aligned_old==True):
            gesture_old=True
            if polytunnel==True:
                command_old=2 #move away
            else:
                command_old=5 #move backwards
          
        for k in range(0,n_human):  
            update=False
            aligned_new=False #initial value
            gesture_new=False #initial value
            danger_new=False #initial value
            centered_new=False #initial value
            command_new=0 #initial value
            if ((area[k]==2 or area[k]==7) and polytunnel==True) or (((area[k]>=1 and area[k]<=3) or (area[k]>=6 and area[k]<=8)) and polytunnel==False):
                aligned_new=True
                if area[k]==2 or area[k]==7:
                    centered_new=True
            if dist[k]<=collision_risk_dist[1] and aligned_new==True:
                danger_new=True 
            if (sensor[k]!=1 and posture[k]==1 and danger_new==False and aligned_new==True):
                gesture_new=True
                if polytunnel==True:
                    command_new=1 #approach
                else:
                    command_new=4 #move forwards
            elif (sensor[k]!=1 and posture[k]==8 and aligned_new==True):
                gesture_new=True
                command_new=3
            elif (sensor[k]!=1 and posture[k]==2 and aligned_new==True):
                gesture_new=True
                if polytunnel==True:
                    command_new=2 #move away
                else:
                    command_new=5 #move backwards
            #############################################################
            #THIS IS THE CORE OF THE CRITICAL HUMAN SELECTION
            if danger_old==True:
                if dist[critical_index]>=dist[k]:
                    update=True
            else: #danger_old=False
                if danger_new==False:
                    if aligned_old==True:
                        if aligned_new==True:
                            if centered_old==True:
                                if centered_new==True:
                                    if gesture_old==True:
                                        if gesture_new==True:
                                            if dist[critical_index]>=dist[k]:
                                                update=True
                                    else: #gesture_old==False
                                        if gesture_new==True:
                                            update=True
                                        else: #gesture_new_False
                                            if dist[critical_index]>=dist[k]:  
                                                update=True
                                else: #centered_new=False
                                    if gesture_old==True:
                                        if gesture_new==True:
                                            update=True
                            else: #centered_old=False
                                if centered_new==True:
                                    if gesture_old==True:
                                        if gesture_new==True:
                                            if dist[critical_index]>=dist[k]:
                                                update=True
                                    else: #gesture_old=False
                                        update=True
                                else: #centered_new=False
                                    if gesture_old==True:
                                        if gesture_new==True:
                                            if dist[critical_index]>=dist[k]:
                                                update=True
                                    else: #gesture_old=False
                                        if gesture_new==True:
                                            update=True
                                        else: #gesture_new=False
                                             if dist[critical_index]>=dist[k]:
                                                update=True
                    else: # aligned_old=False
                        if aligned_new==True:
                            update=True
                        else: #aligned_new=False
                            if dist[critical_index]>=dist[k]:
                                update=True                            
                else:#danger_new=True
                    update=True
            
            
            if update==True:
                critical_index=k
                aligned_old=aligned_new
                gesture_old=gesture_new
                danger_old=danger_new
                centered_old=centered_new
                command_old=command_new                
        #######################################################################
        aligned=aligned_old
        human_command=command_old
        critical_dist=dist[critical_index]
        #HUMAN GLOBAL POSITION
        [h_global_x,h_global_y]=human.human_pose_global(critical_dist,area[critical_index],r_pos_x,r_pos_y,r_pos_theta)
        
        #RISK INFERENCE
        if self.operation=="logistics": #if robot operation is logistics
            if current_goal=="Unknown" or final_goal=="Unknown": #if the robot doesn't start moving autonomously yet
                risk=False
                safe_cond=False
            else: 
                [risk,safe_cond]=self.risk_analysis(area[critical_index],sensor[critical_index],orientation[critical_index],motion[critical_index],h_global_y,h_global_x,aligned,final_goal,r_pos_y,r_pos_x) #risk:=occlusion
        else: #for uv-c treatment
            risk=True #there is always a risk just for being detected
            safe_cond=False
        
        self.critical_index=critical_index
        self.critical_dist=critical_dist
        self.human_command=human_command
        self.risk=risk
        self.safe_cond=safe_cond
        return h_global_y
    
    def risk_analysis(self,area,sensor,orientation,motion,h_global_y,h_global_x,aligned,final_goal,r_pos_y,r_pos_x):
        #################################################################################################################
        #assuming that a human is in risk when he/she is aligned to the robot and closer to the current robot goal than the robot
        #################################################################################################################
        goal = topo_map.rsearch.get_node_from_tmap2(final_goal)
        g_y=goal["node"]["pose"]["position"]["y"]
        g_x=goal["node"]["pose"]["position"]["x"]
        h_dist=abs(h_global_y-g_y)+abs(h_global_x-g_x)
        r_dist=abs(r_pos_y-g_y)+abs(r_pos_x-g_x)
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

    def find_new_goal(self,h_global_y,r_pos_y,current_goal_info):
        #################################################################################################################
        #Asumming that this function is only called when robot is inside polytunnel, not valid at footpaths
        #Asumming the "y" component of row nodes is always greater in upper nodes than lower nodes
        #################################################################################################################
        parent=current_goal_info
        #new motion direction along the row
        if self.safety_action==1: #approach
            if r_pos_y<=h_global_y:
                goal_dir="up" # search for the upper node in the row
            else:
                goal_dir="down" #search for the lowernode in the row
        else: #move away
            if r_pos_y<=h_global_y:
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
        #HUMAN INFO
        sensor=human.sensor
        motion=human.motion
        posture=human.posture
        dist=human.distance
        area=human.area
        orientation=human.orientation
        detection=human.detection
        thermal_detection=human.thermal_detection
        perception=human.perception_msg
        #ROBOT INFO
        r_pos_x=robot.pos_x
        r_pos_y=robot.pos_y
        r_pos_theta=robot.pos_theta
        action=robot.action
        current_goal=robot.current_goal
        final_goal=robot.final_goal
        polytunnel=robot.polytunnel
        current_goal_info=robot.current_goal_info
        
        #Default values
        self.status=0 #no human/human but without risk
        self.human_command=0 #no human command
        self.critical_index=0 #by default
        self.critical_dist=100 #by default
        if current_goal=="Unknown" or final_goal=="Unknown": #If the robot doesn't start moving yet
            self.safety_action=5 # no safety action 
            self.audio_message=0 # no message
            self.new_goal=final_goal # the current goal is not changed
        elif perception==False: #If no human perception messages are being published
            self.safety_action=3 # to pause robot operation 
            self.audio_message=8 # to alert that human perception system is not working
            self.new_goal=final_goal # the current goal is not changed
        elif action==5: #if robot operation is "teleoperation"
            self.safety_action=6 # teleoperation mode 
            self.audio_message=9 # to alert that robot is moving in teleoperation mode
            self.new_goal=final_goal # the current goal is not changed
        else: #In any other case, continue with the decision making
            ##ASSUMING NO HUMAN DETECTION AS INITIAL ASUMPTION
            if self.operation=="logistics":
                if action==0 or action==2: # robot is moving to an initial goal or moving away from the picker
                    self.safety_action=5 # no safety action / keep the previous robot action
                    if action==2:
                        self.audio_message=5 # alert robot moving away
                    else: #action=0
                        self.audio_message=6 # alert robot presence
                    self.new_goal=final_goal # the current goal is not changed             
                elif action==1: # robot was approaching
                    self.safety_action=4 # make it stop for safety purposes, waiting for a human order 
                    self.audio_message=2 # message to ask the human for new order
                    self.new_goal=final_goal # the current goal is not changed
                elif action==3: # if goal was paused while moving to a goal or moving away from the picker
                    self.audio_message=6 # alert robot presence
                    self.safety_action=0 # restart operation making the robot moving to the current goal
                    self.new_goal=final_goal # the current goal is not changed
                elif action==4: #if robot is waiting for human order
                    self.safety_action=5 # no safety action / keep the previous robot action
                    self.audio_message=2 # message to ask the human for new order
                    self.new_goal=final_goal # the current goal is not changed   
                elif action==6: #if robot was in gesture control mode at footpaths           
                    self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                    self.audio_message=2 # message to ask the human for new order
                    self.new_goal=final_goal # the current goal is not changed
            else: #UVC
                self.new_goal=final_goal # the current goal is not changed
                self.safety_action=5 # no safety action / keep the previous robot action
                self.audio_message=0 #no message
        
            ## IN CASE OF HUMAN DETECTION, UPDATING ROBOT ACTION     
            if detection==True: #execute only if at least a human is detected     
                #Critical human selection
                h_global_y=self.critical_human_selection(polytunnel,posture,dist,area,sensor,orientation,motion,current_goal,final_goal,r_pos_x,r_pos_y,r_pos_theta)
                print("DISTANCE TO CRITICAL HUMAN",self.critical_dist)
                ###UV-C treatment#######################################
                if self.operation!="logistics": #during UV-C treatment
                    self.human_command=0 #no human command expected during uv-c treatment
                    if self.risk==True: #only execute if there is risk of producing human injuries 
                        if action==0: #if robot is moving to goal   
                            if self.critical_dist>uvc_risk_dist[0]: #if human is above 10m from the robot
                                self.status=1 #safety HRI
                                self.safety_action=5 # keep the previous robot action
                            elif self.critical_dist>=uvc_risk_dist[1] and self.critical_dist<uvc_risk_dist[0]: #if human is between 7-10m from the robot
                                self.status=2 #risky HRI
                                self.safety_action=5  # keep the previous robot action
                            else: #if human is within 7m
                                self.status=3 #dangerous HRI
                                self.safety_action=4 # stop UV-C treatment and wait for a new human command to restart
                            self.audio_message=1 #UVC danger message
                            self.new_goal=final_goal # the current goal is not changed
                        else: #if robot is static        
                            if self.critical_dist>10: #if human is above 10m from the robot
                                self.status=1 #safety HRI
                                self.safety_action=5 # keep the previous robot action
                            elif self.critical_dist>10: #if human is within 7-10m from the robot
                                self.status=2 #risky HRI
                                self.safety_action=5 # keep the previous robot action
                            else: #if human is within 10m
                                self.status=3 #dangerous HRI
                                self.safety_action=5 # still waiting for a new human command to restart
                            self.audio_message=1 #UVC danger message
                            self.new_goal=final_goal # the current goal is not changed
                        
                ###LOGISTICS###############################################
                else:
                    ##RISK LEVEL
                    if self.critical_dist>collision_risk_dist[0]: #if human is above 3.6m and robot is on normal operation
                        self.status=1 # safety HRI
                    elif self.critical_dist>collision_risk_dist[1] and self.critical_dist<=collision_risk_dist[0]: #if human is between 1.2-3.6m
                        self.status=2 # risky HRI
                    else: #if human is within 1.2m
                        self.status=3 # dangerous HRI
                    ##IN CASE THE HUMAN PERFORMS BODY GESTURES
                    #In case the picker wants the robot to approch to him/her , only valid inside polytunnels and above 1.2m 
                    if self.human_command==1: #sensor[self.critical_index]!=1 and posture[self.critical_index]==1:# and polytunnel==True and self.critical_dist>1.2 and self.aligned==True: #picker is ordering the robot to approach (using both arms)
                         #make the robot approach to the picker (polytunnel)
                        self.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                        self.safety_action=1 # to make the robot approach to the picker
                        self.new_goal=self.find_new_goal(h_global_y,r_pos_y,current_goal_info)
                    #In case the picker wants the robot to stop 
                    elif self.human_command==3:# sensor[self.critical_index]!=1 and posture[self.critical_index]==8:# and self.aligned==True:  #picker is ordering the robot to stop (using right hand)
                         #make the robot stop  (polytunnel and footpaths)
                        self.audio_message=2 # start a message to ask the picker for a new order to approach/move away or move to a new goal
                        self.safety_action=4 # make the robot stop and wait for a new command to restart operation
                        self.new_goal=final_goal # the current goal is not changed
                    #In case the picker wants the robot to move away, only valid inside polytunnels
                    elif self.human_command==2:# sensor[self.critical_index]!=1 and posture[self.critical_index]==2 and polytunnel==True and self.aligned==True:  #picker is ordering the robot to move away (using both hands)
                         #make the robot move away (polytunnel)
                        self.audio_message=5 #message moving away
                        self.safety_action=2 # to make the robot move away from the picker
                        self.new_goal=self.find_new_goal(h_global_y,r_pos_y,current_goal_info)
                    #In case the picker wants the robot to move away, only valid at footpaths
                    elif self.human_command==4:# #picker is ordering the robot to move forwards (using both hands), only valid at footpaths
                         #make the robot move forwards (footpaths)
                        self.audio_message=10 #alet of gesture control
                        self.safety_action=7 # to make the robot activate the gesture control at footpaths
                        self.new_goal=final_goal # the current goal is not changed
                    #In case the picker wants the robot to move away, only valid at footpaths
                    elif self.human_command==5:# #picker is ordering the robot to move backwards (using both hands), only valid at footpaths
                         #make the robot move backwards (footpaths)
                        self.audio_message=10 #alert of gesture control
                        self.safety_action=7 # to make the robot activate the gesture control at footpaths
                        self.new_goal=final_goal # the current goal is not changed
                       
                    if action==0 or action==2: #if robot is moving to an original goal or moving away from the picker
                        if self.status==1: #if human is above 3.6m and robot is on normal operation
                            if self.human_command==0: #if human is not performing any gesture
                                if action==2:
                                    self.audio_message=5 # alert robot moving away
                                else: #action=0
                                    self.audio_message=6 # alert robot presence
                                self.safety_action=5 # keep the previous robot action
                                self.new_goal=final_goal # the current goal is not changed
                        elif self.status==2: #if human is between 1.2-3.6m
                            if self.human_command==0: #if human is not performing any gesture
                                if self.risk==True: # if there is chance to produce injuries
                                    self.safety_action=3 # pause operation and continue when human has move away 
                                    self.audio_message=3 # message to ask the human for free space to continue moving
                                    self.new_goal=final_goal # the current goal is not changed
                                else:
                                    if action==2:
                                        self.audio_message=5 # alert robot moving away
                                    else: #action=0
                                        self.audio_message=6 # alert robot presence
                                    self.safety_action=5 # keep the previous robot action 
                                    self.new_goal=final_goal # the current goal is not changed
                        
                        else: #if human is within 1.2m or moving or is not facig the robot
                            if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards (using both hands)
                                if self.risk==True: # if there is chance to produce injuries
                                    self.safety_action=3 # pause operation and continue when human has move away 
                                    self.audio_message=3 # message to ask the human for free space to continue moving
                                    self.new_goal=final_goal # the current goal is not changed
                                else:
                                    if action==2:
                                        self.audio_message=5 # alert robot moving away
                                    else:
                                        self.audio_message=6 # alert robot presence
                                    self.safety_action=5 # keep the previous robot action
                                    self.new_goal=final_goal # the current goal is not changed
                    elif action==1: #if robot is approaching to the picker
                        if self.status==1: #if human is above 3.6m 
                            if self.human_command==0: #if human is not performing any gesture
                                self.audio_message=4 # alert to make the picker aware of the robot approaching to him/her
                                self.safety_action=5 # keep it approaching to the picker
                                self.new_goal=final_goal # the current goal is not changed
                        elif self.status==2: #if human is within 1.2m to 3.6m
                            if self.human_command==0 : #if human is not performing any gesture 
                                if self.safe_cond==False and self.risk==True: # if safety condition is not safisfied and if there is risk of producing injuries
                                    self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                                    self.audio_message=2 # message to ask the human for new order
                                    self.new_goal=final_goal # the current goal is not changed
                                else: #if safety condition is satisfied or there is not risk of producing injuries
                                    self.audio_message=4 # alert to make the picker aware of the robot approaching to him/her
                                    self.safety_action=5 # keep it approaching to the picker
                                    self.new_goal=final_goal # the current goal is not changed
                        else: #if human is within 1.2m  or moving or is not facig the robot
                            if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards (using both hands) 
                                if self.risk==True: # if there is chance to produce injuries
                                    self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                                    self.audio_message=2 # message to ask the human for new order
                                    self.new_goal=final_goal # the current goal is not changed
                                else:
                                    self.audio_message=4 # alert to make the picker aware of the robot approaching to him/her
                                    self.safety_action=5 # keep it approaching to the picker
                                    self.new_goal=final_goal # the current goal is not changed
                    elif action==3: #if robot is in pause mode, waiting till the human is occluding to continue moving
                        if self.status==1: #if human is above 3.6m and robot is on normal operation
                            if self.human_command==0 : #if human is not performing any gesture
                                self.audio_message=6 # alert robot presence
                                self.safety_action=0 # restart operation making the robot moving to the current goal
                                self.new_goal=final_goal # the current goal is not changed
                        else: #if human is within 3.6m
                            if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards (using both hands)
                                self.safety_action=5 # still in pause operation 
                                self.audio_message=3 # message to ask the human for free space to continue moving
                                self.new_goal=final_goal # the current goal is not changed
                    elif action==4: #if robot is waiting for a new human command
                        if self.status==1 or self.status==2: #if human is above 1.2m 
                            if self.human_command==0: #if human is not performing any gesture
                                self.safety_action=5 # still waiting for human order 
                                self.audio_message=2 # message to ask the human for new order
                                self.new_goal=final_goal # the current goal is not changed
                        else: #if human is within 1.2m or moving or is not facig the robot
                            if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards (using both hands)
                                self.safety_action=5 # still waiting for another human order 
                                self.audio_message=2 # message to ask the human for new order
                                self.new_goal=final_goal # the current goal is not changed
                    elif action==6: #if robot is in gesture control mode
                        if self.human_command==0 or self.human_command==3: #if human is not longer performing any gesture for "gesture control" or if is ordering to stop
                            self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                            self.audio_message=2 # message to ask the human for new order
                            self.new_goal=final_goal # the current goal is not changed                       
            
            elif thermal_detection==True and self.operation=="UVC": #even if no human was detected but thermal_detection is TRUE, only for UVC treatment
                self.human_command=0 #no human command expected during uv-c treatment
                self.status=2 #risky HRI
                self.safety_action=5 # keep the previous robot action     
                self.audio_message=1 #UVC danger message
                self.new_goal=final_goal # the current goal is not changed 
     
   
        
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
    rospy.sleep(rospy.Duration.from_sec(2))
    rospy.Subscriber('human_info',human_msg,human.human_callback)  
    rospy.Subscriber('robot_info',robot_msg,robot.robot_info_callback)  
    rospy.Subscriber('/robot_pose', Pose, robot.robot_pose_callback)  
    robot.base_frame = rospy.get_param("row_traversal/base_frame", "base_link")
    #Rate setup
    rate = rospy.Rate(1/pub_hz)  # main loop frecuency in Hz
    while not rospy.is_shutdown():	      
        hri.decision_making()
        #print("status",hri.status)
        print("##############################################################")
        print("risk",hri.risk)
        #print("aligned",hri.aligned)
        print("critical_index",hri.critical_index)
        print("human_command",hri.human_command)
        print("safety_action",hri.safety_action)
        print("robot_action",robot.action)
        print("final_goal",robot.final_goal)
        ############################################################################################
        #Publish SAFETY SYSTEM MESSAGES        
        # Safety messages are not published only if the safety_system node stopped
        safety_msg.hri_status = hri.status
        safety_msg.audio_message = hri.audio_message
        safety_msg.safety_action = hri.safety_action
        safety_msg.human_command = hri.human_command
        safety_msg.critical_index = hri.critical_index
        safety_msg.critical_dist = hri.critical_dist
        safety_msg.new_goal= hri.new_goal
        safety_msg.operation_mode = hri.operation
        if robot.polytunnel==True:
            safety_msg.action_mode = "polytunnel"
        else:
            safety_msg.action_mode = "footpath"
        pub_safety.publish(safety_msg)
   
        rate.sleep() #to keep fixed the publishing loop rate
