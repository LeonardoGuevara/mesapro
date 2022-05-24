#!/usr/bin/env python

#required packages
import rospy
from tf.transformations import euler_from_quaternion , quaternion_from_euler
from geometry_msgs.msg import Pose, PoseStamped
from math import * #to avoid prefix math.
import numpy as np #to use matrix
import yaml
from std_msgs.msg import String, Bool
from topological_navigation.route_search2 import TopologicalRouteSearch2
from mesapro.msg import human_msg, hri_msg, robot_msg
import threading # Needed for Timer
#######################################################################################################################
#GENERAL PURPUSES VARIABLES
pub_hz=0.01 #main loop frequency
#Setup ROS publiser
pub_safety = rospy.Publisher('human_safety_info', hri_msg,queue_size=1)
safety_msg = hri_msg()
pub_pose = rospy.Publisher("/human/posestamped", PoseStamped, queue_size=1)
pose_msg = PoseStamped()
pub_goal = rospy.Publisher("/goal/posestamped", PoseStamped, queue_size=1)
goal_msg = PoseStamped()
#Importing global parameters from .yaml file
config_direct=rospy.get_param("/hri_safety_system/config_direct") #you have to change /hri_safety_system/ if the node is not named like this
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
collision_risk_dist=parsed_yaml_file.get("human_safety_config").get("collision_risk_distances") #Human to robot distances (m) used to determine the HRI risk during logistics 
uvc_risk_dist=parsed_yaml_file.get("human_safety_config").get("uvc_risk_distances") #Human to robot distances (m) used to determine the HRI risk during uvc treatment
#########################################################################################################################

class robot_class:
    def __init__(self): #It is done only the first iteration
        self.pos_x=0 #X
        self.pos_y=0 #Y
        self.pos_theta=0 #orientation
        self.action=4 #initial value, "waiting for human command" as initial condition
        self.current_goal="Unknown"  #name of current robot goal
        self.current_goal_info="Unknown"   #initial condition
        self.polytunnel=False #to know if robot is moving inside the polytunnel or not, by default is "False"
        self.final_goal="Unknown" #name of final robot goal
        self.goal_coord=[0,0] # (x,y) coordinates of final_goal , by default (0,0)
        self.collision_flag=False #Flag to know if a collision was detected by the safety bumpers, by default is "False"
        self.collection_point=parsed_yaml_file.get("human_safety_config").get("collection_point")  #Name of the node defined as the collection_point
        self.resume_goal=False # Flag to know if the robot can resume the previous goal after been stop or if can move to the collection point if previos goal was already completed
        self.time_without_hri=parsed_yaml_file.get("human_safety_config").get("time_without_hri") # Maximum time without receiving a new command while robot is "waiting for a command"
        self.efficient_timer_running=False #flag to know if the efficient timer is running or not
        self.automatic_reactivation=rospy.get_param("/hri_safety_system/automatic_reactivation") # flag to know if automatic reactivation feature is required or not
        #self.timer_efficient = threading.Timer(self.time_without_hri,self.efficient_timeout) # If "n" seconds elapse, call efficient_timeout()
        #self.timer_efficient.start()
        
    def robot_info_callback(self,robot_info):
        if robot_info.current_node!="Unknown":
            parent=topo_map.rsearch.get_node_from_tmap2(robot_info.current_node)
            polytunnel = False #initial assuption that the robot is outside the polytunnel nodes, then the human commands are not allowed
            for edge in parent["node"]["edges"]:
                if edge["action"]=="row_traversal":
                    polytunnel = True # If at least an edge is connected with the polytunnels nodes, then the human commands are allowed
                    break
            self.polytunnel=polytunnel
            self.current_goal=robot_info.current_node
            self.current_goal_info=parent
            self.final_goal=robot_info.goal_node
            if self.final_goal!="Unknown":
                parent=topo_map.rsearch.get_node_from_tmap2(self.final_goal)
                self.goal_coord=[parent["node"]["pose"]["position"]["x"],parent["node"]["pose"]["position"]["y"]]
        self.action=robot_info.action
        if self.automatic_reactivation==True:
            if self.efficient_timer_running==False:
                if self.action==4 and (human.detection==False or (human.detection==True and hri.critical_dist>collision_risk_dist[0])) and self.final_goal!="Unknown" and self.current_goal!=self.collection_point: 
                    self.timer_efficient = threading.Timer(self.time_without_hri,self.efficient_timeout) # If "n" seconds elapse, call safety_timeout()
                    self.timer_efficient.start()
                    self.efficient_timer_running=True
                    #self.resume_goal=False
            else:    
                if (self.action==4 and ((human.detection==True and hri.critical_dist<collision_risk_dist[0]) or self.final_goal=="Unknown" or self.current_goal==self.collection_point)) or self.action!=4: 
                    self.timer_efficient.cancel()
                    self.efficient_timer_running=False
                    #self.resume_goal=False
             
    def robot_pose_callback(self,pose):
        self.pos_x=pose.position.x
        self.pos_y=pose.position.y
        quat = pose.orientation    
        # From quaternion to Euler
        angles = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        theta = angles[2]
        self.pos_theta=np.unwrap([theta])[0]
        
    def collision_detection_callback(self,collision):
        self.collision_flag=collision.data

    def efficient_timeout(self):
        print("No human command received in a long time")
        #if self.final_goal!="Unknown" and self.current_goal!=self.collection_point: #only if goal is not "Unknown" 
        self.resume_goal=True #to alert that robot can resume previous goal or move to collection point
        #self.efficient_timer_running==False
        
class human_class:
    def __init__(self): 
        self.pos_x=[0]
        self.pos_y=[0]
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
        self.time_without_msg=parsed_yaml_file.get("human_safety_config").get("time_without_msg") # Maximum time without receiving human perception messages
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        
    def human_callback(self,human_info):
        self.pos_x=human_info.position_x
        self.pos_y=human_info.position_y
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
        
    def human_pose_global(self,orientation,area,r_pos_x,r_pos_y,r_pos_theta,h_local_x,h_local_y):
        #assuming the human orientation is aligned to the robot orientation (facing to it or giving the back)
        if area<=4: #if human was detected in front of the robot (areas 0 to 4)
            new_theta=r_pos_theta 
            pos_y=(sin(new_theta)*h_local_x+cos(new_theta)*h_local_y)+r_pos_y
            pos_x=(cos(new_theta)*h_local_x-sin(new_theta)*h_local_y)+r_pos_x
        elif area>=5: #if human was detected at the back of the robot (areas 5 to 9)
            new_theta=r_pos_theta+pi
            pos_y=(-sin(new_theta)*h_local_x-cos(new_theta)*h_local_y)+r_pos_y
            pos_x=(-cos(new_theta)*h_local_x+sin(new_theta)*h_local_y)+r_pos_x
        if orientation==0:
            pos_theta=new_theta+pi
        else:
            pos_theta=new_theta
        return pos_x,pos_y,pos_theta

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
        self.pos_global_x=0
        self.pos_global_y=0
        self.pos_global_theta=0
        self.status=0 #initial condition 
        self.audio_message=0 #initial condition 
        self.safety_action=7 #initial condition     
        self.human_command=0 #initial condition 
        self.critical_index=0 #index of the human considered as critical during interaction (it is not neccesary the same than the closest human or the goal human)
        self.critical_dist=0 #distance of the critical human
        self.safe_cond_occlusion=False #it is true if the detected human is considered in risk during autonomous navigation (occluding the robot path), used to avoid unnecesary stops with human who is not occluding the current robot path
        self.new_goal="none" #name of the new final goal
        self.operation=parsed_yaml_file.get("robot_config").get("operation_mode") #it can be "UVC" or "logistics"
        self.safe_cond_approach=False #True if the human is static and facing the robot, False if not satisfying this safety condition, used to stop the robot during approaching maneuvers inside polytunnels
        self.safe_cond_critic_area=False #True if the human is located at critical areas, False if is not located at critical areas, used to determine if a human is critical to be tracked or not
        self.gesture_control=rospy.get_param("/hri_safety_system/gesture_control",True) # Flag to activate or not gesture control

    def critical_human_selection(self,polytunnel,posture,dist,area,sensor,orientation,motion,final_goal,r_pos_x,r_pos_y,r_pos_theta,h_pos_x,h_pos_y,action):  
        ##################################################################################################################
        #Critical areas distribution: Inside the polytunnel includes areas 2 and 7, and outside polytunnel includes 1,2,3 and 6,7,8
        #critical human selection logic is: 1st human within 1.2m, 2nd human performing gesture, 3rd human centered, 4rd closest human   
        ######################################################################################################################
        n_human=len(dist)
        critical_area_old=False #initial value
        gesture_old=False #initial value
        danger_old=False #initial value
        centered_old=False #initial value
        command_old=0 #initial value
        critical_index=0 #initial value
        
        if ((area[critical_index]==2 or area[critical_index]==7) and polytunnel==True) or (((area[critical_index]>=1 and area[critical_index]<=3) or (area[critical_index]>=6 and area[critical_index]<=8)) and polytunnel==False):
            critical_area_old=True
            if area[critical_index]==2 or area[critical_index]==7:
                centered_old=True
        if dist[critical_index]<=collision_risk_dist[1] and critical_area_old==True:
            danger_old=True 
        if (sensor[critical_index]!=1 and posture[critical_index]==8 and orientation[critical_index]==0 and danger_old==False and critical_area_old==True):  #rigth_forearm_sideways
            gesture_old=True
            if polytunnel==True:
                command_old=1 #approach (approching to picker inside polytunnels)
            else:
                command_old=4 #move forwards (move towards the human at footpaths)
        elif (sensor[critical_index]!=1 and posture[critical_index]==10 and orientation[critical_index]==0  and critical_area_old==True): #both_hands_front
            gesture_old=True
            command_old=3 # stop (make the robot stop and wait for new command inside and outside polytunnels)
        elif (sensor[critical_index]!=1 and posture[critical_index]==4 and orientation[critical_index]==0  and critical_area_old==True): #left_forearm_sideways
            gesture_old=True
            if polytunnel==True:
                command_old=2 #move away (move away from picker inside polytunnel)
            else:
                command_old=5 #move backwards (move away from human at footpaths)
        elif (sensor[critical_index]!=1 and posture[critical_index]==7  and orientation[critical_index]==0 and critical_area_old==True and polytunnel==False): #right_arm_sideways
            gesture_old=True
            command_old=6 #move right (only valid at footpaths)
        elif (sensor[critical_index]!=1 and posture[critical_index]==3 and orientation[critical_index]==0 and critical_area_old==True and polytunnel==False): #left_arm_sideways
            gesture_old=True
            command_old=7 #move left (only valid at footpaths)
        elif (sensor[critical_index]!=1 and posture[critical_index]==5 and orientation[critical_index]==0 and critical_area_old==True and polytunnel==False): #right_arm_up
            gesture_old=True
            command_old=8 #rotate clockwise (only valid at footpaths)
        elif (sensor[critical_index]!=1 and posture[critical_index]==1 and orientation[critical_index]==0 and critical_area_old==True and polytunnel==False): #left_arm_up
            gesture_old=True
            command_old=9 #rotate counterclockwise (only valid at footpaths)
            
            
        for k in range(0,n_human):  
            update=False
            critical_area_new=False #initial value
            gesture_new=False #initial value
            danger_new=False #initial value
            centered_new=False #initial value
            command_new=0 #initial value
            if ((area[k]==2 or area[k]==7) and polytunnel==True) or (((area[k]>=1 and area[k]<=3) or (area[k]>=6 and area[k]<=8)) and polytunnel==False):
                critical_area_new=True
                if area[k]==2 or area[k]==7:
                    centered_new=True
            if dist[k]<=collision_risk_dist[1] and critical_area_new==True:
                danger_new=True 
            if (sensor[k]!=1 and posture[k]==8  and orientation[k]==0  and danger_new==False and critical_area_new==True): #rigth_forearm_sideways
                gesture_new=True
                if polytunnel==True:
                    command_new=1 #approach (approching to picker inside polytunnels)
                else:
                    command_new=4 #move forwards (move towards the human at footpaths)
            elif (sensor[k]!=1 and posture[k]==10  and orientation[k]==0 and critical_area_new==True): #both_hands_front
                gesture_new=True
                command_new=3 # stop (make the robot stop and wait for new command inside and outside polytunnels)
            elif (sensor[k]!=1 and posture[k]==4  and orientation[k]==0 and critical_area_new==True): #left_forearm_sideways
                gesture_new=True
                if polytunnel==True:
                    command_new=2 #move away (move away from picker inside polytunnel)
                else:
                    command_new=5 #move backwards (move away from human at footpaths)
            elif (sensor[k]!=1 and posture[k]==7  and orientation[k]==0 and critical_area_new==True and polytunnel==False): #right_arm_sideways
                gesture_new=True
                command_new=6 #move right (only valid at footpaths)
            elif (sensor[k]!=1 and posture[k]==3  and orientation[k]==0 and critical_area_new==True and polytunnel==False): #left_arm_sideways
                gesture_new=True
                command_new=7 #move left (only valid at footpaths)
            elif (sensor[k]!=1 and posture[k]==5  and orientation[k]==0 and critical_area_new==True and polytunnel==False): #right_arm_up
                gesture_new=True
                command_new=8 #rotate clockwise  (only valid at footpaths)
            elif (sensor[k]!=1 and posture[k]==1  and orientation[k]==0 and critical_area_new==True and polytunnel==False): #left_arm_up
                gesture_new=True
                command_new=9 #rotate counterclockwise  (only valid at footpaths)
            #############################################################
            #THIS IS THE CORE OF THE CRITICAL HUMAN SELECTION
            if danger_old==True:
                if dist[critical_index]>=dist[k]:
                    update=True
            else: #danger_old=False
                if danger_new==False:
                    if critical_area_old==True:
                        if critical_area_new==True:
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
                    else: # critical_area_old=False
                        if critical_area_new==True:
                            update=True
                        else: #critical_area_new=False
                            if dist[critical_index]>=dist[k]:
                                update=True                            
                else:#danger_new=True
                    update=True
            
            
            if update==True:
                critical_index=k
                critical_area_old=critical_area_new
                gesture_old=gesture_new
                danger_old=danger_new
                centered_old=centered_new
                command_old=command_new                
        #######################################################################
        critical_area=critical_area_old
        human_command=command_old
        critical_dist=dist[critical_index]
        h_local_x=h_pos_x[critical_index]
        h_local_y=h_pos_y[critical_index]
        
        #HUMAN GLOBAL POSITION
        [h_global_x,h_global_y,h_global_theta]=human.human_pose_global(orientation[critical_index],area[critical_index],r_pos_x,r_pos_y,r_pos_theta,h_local_x,h_local_y)
        
        #RISK INFERENCE FOR TOPO NAV ACTIONS
        safe_cond_occlusion=False #initial condition
        safe_cond_approach=False #initial condition
        if self.operation=="logistics": #if robot operation is logistics
            if final_goal!="Unknown": #execute only if the robot has a goal selected already
                if action>=0 and action<=3: #If robot is "moving to current goal" or "approaching" or "moving away", or "pause" i.e. only for operations which require topological goals
                    [safe_cond_occlusion,safe_cond_approach]=self.risk_analysis(area[critical_index],sensor[critical_index],orientation[critical_index],motion[critical_index],h_global_y,h_global_x,critical_area,final_goal,r_pos_y,r_pos_x) #risk:=occlusion
                
        self.critical_index=critical_index
        self.critical_dist=critical_dist
        self.human_command=human_command
        self.safe_cond_occlusion=safe_cond_occlusion
        self.safe_cond_approach=safe_cond_approach
        self.safe_cond_critic_area=critical_area
        self.pos_global_x=h_global_x
        self.pos_global_y=h_global_y
        self.pos_global_theta=h_global_theta
    
    def risk_analysis(self,area,sensor,orientation,motion,h_global_y,h_global_x,critical_area,final_goal,r_pos_y,r_pos_x):
        #################################################################################################################
        #When robot is moving to a topological goal, it is assume a risk when he/she is in a critical area and closer to the current robot goal than the robot
        #################################################################################################################
        goal = topo_map.rsearch.get_node_from_tmap2(final_goal)
        g_y=goal["node"]["pose"]["position"]["y"]
        g_x=goal["node"]["pose"]["position"]["x"]
        h_dist=abs(h_global_y-g_y)+abs(h_global_x-g_x)
        r_dist=abs(r_pos_y-g_y)+abs(r_pos_x-g_x)
        if critical_area==True: 
            if h_dist<=r_dist:
                safe_cond_occlusion=True
            else:
                safe_cond_occlusion=False
        else: #if human is on the side of the robot 
            safe_cond_occlusion=False
        if motion==2 or (sensor!=1 and orientation==1): #In case the human is not static, or he/she is not facing the robot
            safe_cond_approach=False #not safe
        else:
            safe_cond_approach=True    
        return safe_cond_occlusion, safe_cond_approach

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
    
    def resume_goal_or_collection_point(self,current_goal,final_goal,collection_point):
        if current_goal==final_goal:
            goal=collection_point
        else:
            goal=final_goal
        return goal
    
    def get_connected_nodes_tmap(self, node):
        children=[]
        for edge in node["node"]["edges"]:
            children.append(edge["node"])
        return children
    
    def decision_making(self):
        #HUMAN INFO
        h_pos_x=human.pos_x
        h_pos_y=human.pos_y
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
        final_goal=robot.final_goal
        polytunnel=robot.polytunnel
        current_goal_info=robot.current_goal_info
        collision=robot.collision_flag
        current_goal=robot.current_goal
        collection_point=robot.collection_point
    
        ## IN CASE OF HUMAN DETECTION 
        if detection==True: #execute only if at least a human is detected   
            #Critical human selection
            self.critical_human_selection(polytunnel,posture,dist,area,sensor,orientation,motion,final_goal,r_pos_x,r_pos_y,r_pos_theta,h_pos_x,h_pos_y,action)
            print("DISTANCE TO CRITICAL HUMAN",self.critical_dist)
            ###UV-C treatment#######################################
            if self.operation!="logistics": #during UV-C treatment
                self.human_command=0 #no human command expected during uv-c treatment
                self.new_goal=final_goal # the current goal is not changed
                self.audio_message=1 #UVC danger message
                if self.critical_dist>uvc_risk_dist[0]: #if human is above 10m from the robot
                    self.status=1 #safety HRI
                    self.safety_action=7 # keep the previous robot action
                elif self.critical_dist>=uvc_risk_dist[1] and self.critical_dist<uvc_risk_dist[0]: #if human is between 7-10m from the robot
                    self.status=2 #risky HRI
                    self.safety_action=7  # keep the previous robot action
                else: #if human is within 7m
                    self.status=3 #dangerous HRI
                    self.safety_action=4 # stop UV-C treatment and wait for a new human command to restart
                if action!=0: #if robot is not moving to goal, i.e. static           
                    self.safety_action=7 # # keep the previous robot action
                
            ###LOGISTICS###############################################
            else:
                ##RISK LEVEL
                if action>=0 and action<=3: #If robot is "moving to current goal" or "approaching" or "moving away", or "pause" i.e. only for operations which require topological goals
                    if self.safe_cond_occlusion==True or final_goal=="Unknown": #if human is occluding the current robot path or if a goal hasn't been selected yet
                        if self.critical_dist>collision_risk_dist[0]: #if human is above 3.6m and robot is on normal operation
                            self.status=1 # safety HRI
                        elif self.critical_dist>collision_risk_dist[1] and self.critical_dist<=collision_risk_dist[0]: #if human is between 1.2-3.6m
                            self.status=2 # risky HRI
                        else: #if human is within 1.2m
                            self.status=3 # dangerous HRI
                        if action==1 and self.safe_cond_approach==False: #if robot is "approaching" and picker is "moving" or "not facing the robot"
                            self.status=3 # dangerous HRI
                    else: #if human is not located at critical areas or is not occluding the robot path
                        self.status=1 # safety HRI
                else: #for operations which do not require topological goals, i.e. "waiting_for_new_human_command","teleoperation","gesture_control"
                    if self.safe_cond_critic_area==True or final_goal=="Unknown": #if human is located at critical areas or if a goal hasn't been selected yet
                        if self.critical_dist>collision_risk_dist[0]: #if human is above 3.6m and robot is on normal operation
                            self.status=1 # safety HRI
                        elif self.critical_dist>collision_risk_dist[1] and self.critical_dist<=collision_risk_dist[0]: #if human is between 1.2-3.6m
                            self.status=2 # risky HRI
                        else: #if human is within 1.2m
                            self.status=3 # dangerous HRI    
                    else: #if human is not located at critical areas 
                         self.status=1 # safety HRI
                ##IN CASE THE HUMAN PERFORMS BODY GESTURES
                if self.gesture_control==True and self.critical_dist<=collision_risk_dist[0]: #excecute only if gesture_control is activated and if human is <3.6m (to avoid false positives)
                    #In case the picker wants the robot to approch to him/her , only valid inside polytunnels and above 1.2m 
                    if self.human_command==1: #sensor[self.critical_index]!=1 and posture[self.critical_index]==1:# and polytunnel==True and self.critical_dist>1.2 and self.critical_area==True: #picker is ordering the robot to approach (using both arms)
                        self.audio_message=4 #alert to make the picker aware of the robot approaching to him/her
                        self.safety_action=1 # to make the robot approach to the picker
                        self.new_goal=self.find_new_goal(self.pos_global_y,r_pos_y,current_goal_info)
                    #In case the picker wants the robot to stop  (polytunnel and footpath)
                    elif self.human_command==3:
                        self.audio_message=2 # start a message to ask the picker for a new order to approach/move away or move to a new goal
                        self.safety_action=4 # make the robot stop and wait for a new command to restart operation
                        self.new_goal=final_goal # the current goal is not changed
                    #In case the picker wants the robot to move away, only valid inside polytunnels
                    elif self.human_command==2:
                        self.audio_message=5 #message moving away
                        self.safety_action=2 # to make the robot move away from the picker
                        self.new_goal=self.find_new_goal(self.pos_global_y,r_pos_y,current_goal_info)
                    #In case the picker wants to control the robot velocities by performing gestures (only valid at footpaths)
                    elif self.human_command>=4 and self.human_command<=9:
                        self.audio_message=10 #alet of gesture control
                        self.safety_action=6 # to make the robot activate the gesture control at footpaths
                        self.new_goal=final_goal # the current goal is not changed
                else:
                    self.human_command=0 #assuming no gesture detected
                   
                if action==0 or action==2: #if robot is moving to an original goal or moving away from the picker
                    if self.status==1: #if human is above 3.6m and robot is on normal operation
                        if self.human_command==0: #if human is not performing any gesture
                            if action==2:
                                self.audio_message=5 # alert robot moving away
                            else: #action=0
                                self.audio_message=6 # alert robot presence
                            self.safety_action=7 # keep the previous robot action
                            self.new_goal=final_goal # the current goal is not changed
                    elif self.status==2: #if human is between 1.2-3.6m
                        if self.human_command==0: #if human is not performing any gesture
                            self.safety_action=3 # pause operation and continue when human has move away 
                            self.audio_message=3 # message to ask the human for free space to continue moving
                            self.new_goal=final_goal # the current goal is not changed                         
                    else: #if human is within 1.2m 
                        if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards 
                            self.safety_action=3 # pause operation and continue when human has move away 
                            self.audio_message=3 # message to ask the human for free space to continue moving
                            self.new_goal=final_goal # the current goal is not changed
                            
                elif action==1: #if robot is approaching to the picker
                    if self.status==1 or self.status==2 : #if human is above 1.2m
                        if self.human_command==0: #if human is not performing any gesture
                            self.audio_message=4 # alert to make the picker aware of the robot approaching to him/her
                            self.safety_action=7 # keep it approaching to the picker
                            self.new_goal=final_goal # the current goal is not changed
                    else: #if human is within 1.2m  or moving or is not facig the robot
                        if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards 
                            self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                            self.audio_message=2 # message to ask the human for new order
                            self.new_goal=final_goal # the current goal is not changed
                            
                elif action==3: #if robot is in pause mode, waiting till the human is occluding to continue moving
                    if self.status==1 : #if human is above 3.6m 
                        if self.human_command==0 : #if human is not performing any gesture
                            self.audio_message=6 # alert robot presence
                            self.safety_action=0 # restart operation making the robot moving to the current goal
                            self.new_goal=final_goal # the current goal is not changed
                    elif self.status==2 : #if human is between 1.2-3.6m
                        if self.human_command==0: #if human is not performing any gesture
                            self.safety_action=3 # still in pause operation and continue when human has move away 
                            self.audio_message=3 # message to ask the human for free space to continue moving
                            self.new_goal=final_goal # the current goal is not changed       
                    else: #if human is within 1.2m
                        if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards 
                            self.safety_action=7 # still in pause operation 
                            self.audio_message=3 # message to ask the human for free space to continue moving
                            self.new_goal=final_goal # the current goal is not changed
                            
                elif action==4: #if robot is waiting for a new human command
                    if robot.resume_goal==True:
                        self.new_goal=self.resume_goal_or_collection_point(current_goal,final_goal,collection_point)    
                        self.safety_action=0 # moving to current goal (resuming the previos goal or going to collection point)
                        self.audio_message=6 # alert robot presence
                        robot.resume_goal=False   
                    else:
                        if self.status==1 or self.status==2: #if human is above 1.2m 
                            if self.human_command==0: #if human is not performing any gesture
                                self.safety_action=7 # still waiting for human order 
                                self.audio_message=2 # message to ask the human for new order
                                self.new_goal=final_goal # the current goal is not changed
                        else: #if human is within 1.2m or moving or is not facig the robot
                            if self.human_command!=2 and self.human_command!=5:  # if picker is not ordering the robot to move away or move backwards 
                                self.safety_action=7 # still waiting for another human order 
                                self.audio_message=2 # message to ask the human for new order
                                self.new_goal=final_goal # the current goal is not changed
                elif action==6: #if robot is in gesture control mode
                    if self.human_command==0 or self.human_command==3: #if human is not longer performing any gesture for "gesture control" or if is ordering to stop
                        self.safety_action=4 # stop operation and wait till human gives the robot a new order 
                        self.audio_message=2 # message to ask the human for new order
                        self.new_goal=final_goal # the current goal is not changed                       
        
        elif thermal_detection==True and self.operation!="logistics": #even if no human was detected but thermal_detection is TRUE, only for UVC treatment
            self.human_command=0 #no human command expected during uv-c treatment
            self.status=2 #risky HRI
            self.safety_action=7 # keep the previous robot action     
            self.audio_message=1 #UVC danger message
            self.new_goal=final_goal # the current goal is not changed 
        
        else: #NO HUMAN DETECTION        
            #Default values
            self.status=0 #no human/human but without risk
            self.critical_index=0 #by default
            self.critical_dist=100 #by default
            self.human_command=0 #no human command
            self.new_goal=final_goal # the current goal is not changed
            if self.operation=="logistics":
                if action==0 or action==2: # robot is moving to an initial goal or moving away from the picker
                    self.safety_action=7 # no safety action / keep the previous robot action
                    if action==2:
                        self.audio_message=5 # alert robot moving away
                    else: #action=0
                        self.audio_message=6 # alert robot presence
                elif action==1 or action==6: # if robot was approaching or it was in gesture control mode at footpaths  
                    self.safety_action=4 # make it stop for safety purposes, waiting for a human order 
                    self.audio_message=2 # message to ask the human for new order
                elif action==3: # if goal was paused while moving to a goal or moving away from the picker
                    self.audio_message=6 # alert robot presence
                    self.safety_action=0 # restart operation making the robot moving to the current goal
                elif action==4: #if robot is waiting for human order
                    if robot.resume_goal==True:
                        self.new_goal=self.resume_goal_or_collection_point(current_goal,final_goal,collection_point)    
                        self.safety_action=0 # moving to current goal (resuming the previos goal or going to collection point)
                        self.audio_message=6 # alert robot presence
                        robot.resume_goal=False                       
                    else:
                        self.safety_action=7 # no safety action / keep the previous robot action
                        self.audio_message=2 # message to ask the human for new order
                
            else: #UVC
                self.safety_action=7 # no safety action / keep the previous robot action
                self.audio_message=0 #no message
        
        #FINALLY, check if a collision was detected, if perception system is ok, if teleoperacion is not activated and then check if first goal was assined already
        if collision==True:
            self.status=3 #Collision detected
            #self.critical_index=0 #by default
            #self.critical_dist=0 #by default 0
            self.human_command=0 #no human command
            #self.new_goal=final_goal # the current goal is not changed
            self.safety_action=3 # to make the robot pause the operation 
            self.audio_message=11 # message for collision detection
        elif perception==False: #If no human perception messages are being published
            self.safety_action=3 # to pause robot operation 
            self.audio_message=8 # to alert that human perception system is not working
            self.human_command=0 #no human command
        elif action==5: #if robot operation is "teleoperation"
            self.safety_action=5 # teleoperation mode 
            self.audio_message=9 # to alert that robot is moving in teleoperation mode
            self.human_command=0 #no human command
        elif final_goal=="Unknown" and self.safety_action!=6: #If the robot doesn't have a goal yet BUT gesture control (footpaths) is not activated
            self.safety_action=7 # no safety action 
            self.audio_message=0 # no message
            self.human_command=0 #no human command
        
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
    rospy.Subscriber('collision_detection', Bool, robot.collision_detection_callback) 
    robot.base_frame = rospy.get_param("row_traversal/base_frame", "base_link")
    #Rate setup
    rate = rospy.Rate(1/pub_hz)  # main loop frecuency in Hz
    while not rospy.is_shutdown():	      
        hri.decision_making()
        #print("status",hri.status)
        print("##############################################################")
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
        #Publish Human position
        pose_msg.header.seq = 1
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = hri.pos_global_x
        pose_msg.pose.position.y = hri.pos_global_y
        pose_msg.pose.position.z = 0.0
        quater = quaternion_from_euler(0, 0, hri.pos_global_theta, 'ryxz')
        pose_msg.pose.orientation.x = quater[0]
        pose_msg.pose.orientation.y = quater[1]
        pose_msg.pose.orientation.z = quater[2]
        pose_msg.pose.orientation.w = quater[3]
        pub_pose.publish(pose_msg)
        #Publish Goal position
        goal_msg.header.seq = 1
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = robot.goal_coord[0]
        goal_msg.pose.position.y = robot.goal_coord[1]
        goal_msg.pose.position.z = 0.0
        quater = quaternion_from_euler(0, 0, 1.7, 'ryxz')
        goal_msg.pose.orientation.x = quater[0]
        goal_msg.pose.orientation.y = quater[1]
        goal_msg.pose.orientation.z = quater[2]
        goal_msg.pose.orientation.w = quater[3]
        pub_goal.publish(goal_msg)
        rate.sleep() #to keep fixed the publishing loop rate
