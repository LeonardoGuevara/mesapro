#!/usr/bin/env python

import rospy
import math
import PyKDL
import sys, yaml, os
import actionlib

import tf
from tf.transformations import euler_from_quaternion

import numpy as np
import matplotlib.path as mplPath

from copy import deepcopy
from datetime import datetime

from dynamic_reconfigure.server import Server
from polytunnel_navigation_actions.cfg import RowTraversalConfig

from polytunnel_navigation_actions.point2line import pnt2line
from polytunnel_navigation_actions.pid import PID
from polytunnel_navigation_actions.low_pass_filter import LowPassFilter

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from std_msgs.msg import String
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Point, PointStamped
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker

from std_srvs.srv import SetBool

from strands_navigation_msgs.msg import TopologicalMap
import polytunnel_navigation_actions.msg
from polytunnel_navigation_actions.msg import Obstacle, ObstacleArray
from polytunnel_navigation_actions.msg import TwoInts
from polytunnel_navigation_actions.msg import RowTraversalHealth
from topological_navigation_msgs.msg import ClosestEdges

#########################################################################################################    
#### CHANGES NEEDED FOR HUMAN AWARE NAVIGATION ##########################################################
#########################################################################################################
from mesapro.msg import hri_msg , robot_msg
#########################################################################################################
#########################################################################################################

class inRowTravServer(object):

    _feedback = polytunnel_navigation_actions.msg.inrownavFeedback()
    _result   = polytunnel_navigation_actions.msg.inrownavResult()
    _controller_freq=20.0
    _controller_rate=1/_controller_freq

    def __init__(self, name, pole_pos_per_row):

        self.pole_pos_per_row = pole_pos_per_row

        self.collision=False
        self._user_controlled=False
        self.goal_overshot=False
        self.prealign_y_axis=True

        self.giveup_timer_active=False
        self.notification_timer_active=False
        self.notified=False

        # Dynamically reconfigurable parameters
        self.constant_forward_speed = False     # Stop when obstacle in safety area only (no slowdown) **WIP**
        self.min_obj_size = 0.1                 # Minimum object radius for slow down
        self.min_dist_to_obj = 0.5              # Distance to object at which the robot should stop
        self.approach_dist_to_obj = 3.0         # Distance to object at which the robot starts to slow down
        self.force_forwards_facing=False        # Force Robot to face forwards during traversal
        self.force_backwards_facing=False       # Force Robot to face backwards during traversal
        self.prealign_y_axis=True               # Align laterally before going forwards
        self.maximum_dev_dist=3.0               # Maximum overall distance the robot can drift away from the line before cancelling goal
        self.maximum_dev_dist_y=0.2             # Maximum distance in y the robot can drift away from the line before cancelling goal
        self.maximum_dev_dist_theta=0.17        # Maximum angle [rad] the robot can drift away from the line before cancelling goal
        self.initial_heading_tolerance = 0.005  # Initial heading tolerance [rads]
        self.initial_alignment_tolerance = 0.05 # Initial alingment tolerance [m], how far to the side it is acceptable for the robot to be before moving forwards
        ## PID Controller ######################################################
        self.kp_ang= 0.5            # Proportional gain for heading corrections
        self.ki_ang= 0.0            # Integral gain for heading corrections
        self.kd_ang=0.0             # Derivative gain for heading corrections
        self.kp_y= 0.75             # Integral gain for sideways corrections
        self.ki_y= 0.0              # Proportional gain for sideways corrections
        self.kd_y= 0.0              # Derivative gain for sideways corrections
        ## end PID Controller ##################################################
        self.granularity= 0.5                   # Distance between minigoals along path (carrot points)
        self.y_row_detection_bias = 0.7         # Weight given to the reference given by row detection
        self.y_path_following_bias = 0.3        # Weight given to the original path following
        self.ang_row_detection_bias = 0.2       # Weight given to the angular reference given by row detection
        self.ang_path_following_bias = 0.8      # Weight given to the angular refernce given by path following
        self.minimum_turning_speed = 0.01       # Minimum turning speed
        self.emergency_clearance_x = 0.22       # Clearance from corner frames to trigger emergency stop in x
        self.emergency_clearance_y = 0.22       # Clearance from corner frames to trigger emergency stop in y
        self.goal_tolerance_radius = 0.2        # Goal tolerance Radius in metres
        self.forward_speed= 0.8                 # Forward moving speed
        self.quit_on_timeout=False              # SHould the robot cancel when it meets an obstacle?
        self.time_to_quit=10.0                  # Time until the action is cancelled since collision detected
        self.simultaneous_alignment=False       # Wether to align heading and Y axis simultaneusly or not
        self.y_speed_limit=0.3                  # Maximum speed on the Y axis
        self.turning_speed_limit=0.1            # Maximum turning speed
        self.use_row_entry_function = True      # Enter the row using row_entry function or just go forwards
        self.row_entry_distance = 1.5           # Distance in meters to use row entry function
        self.row_entry_min_speed = 0.15         # Minimum row entry speed
        self.row_entry_kp = 0.3                 # Row entry forward speed gain based on distance travelled
        self.tf_buffer_size = 1.0               # Size of the tf buffer in seconds
        self.offset_row_detector = False        # Whether to offset the row detector output
        self.follow_poles_from_file = False     # Whether to define the path by poles in the pole file
        self.use_low_pass_filter = True         # Whether to apply a low pass filter to references
        self.cutoff_freq_ang=1.5                # Low pass filter cut off frequency for angle error
        self.cutoff_freq_y=1.5                  # Low pass filter cut off frequency for y error
        # This dictionary defines which function should be called when a variable changes via dynamic reconfigure
        self._reconf_functions={'variables':['emergency_clearance_x', 'emergency_clearance_y', 'tf_buffer_size',
                                             'kp_ang', 'ki_ang', 'kd_ang', 'kp_y', 'ki_y', 'kd_y'],
                                'functions':[self.define_safety_zone, self.define_safety_zone, self.reconf_tf_listener,
                                             self.setup_controllers, self.setup_controllers, self.setup_controllers, self.setup_controllers, self.setup_controllers, self.setup_controllers]}
        
        #########################################################################################################    
        #### CHANGES NEEDED FOR HUMAN AWARE NAVIGATION ##########################################################
        #########################################################################################################
        self.hri_dist=100                       # distance to the critical human computed by the safety_system
        self.hri_safety_action=5                # new safety action determined by the safety_system
        self.robot_action=4                     # current robot action, initially waiting for human command,
        self.han_start_dist=3.6                 # Human to robot Distance at which the robot starts to slow down
        self.han_final_dist=1                   # Human to robot Distance at which the robot must stop
        ########################################################################################################### 
        ###########################################################################################################
        # Useful variables
        self.object_detected = False
        self.curr_distance_to_object=-1.0
        self.laser_emergency_regions=[]
        self.redefine_laser_regions=False
        self.limits=[]
        self.emergency_base_points=[]  # Corners of Emergency Areas
        self.y_ref=None
        self.ang_ref=None
        self.config={}
        self.cancelled = False
        self.paused = False
        self.dev_pause = False
        self.tf_pause = False
        self._action_name = name
        self._got_top_map=False
        self.lnodes=None
        self.action_edges=None
        self.action_nodes=None
        self.backwards_mode=False
        self.safety_marker=None
        self.active=False
        self.row_traversal_status = "OFF"
        self.y_err_path = 0.0
        self.ang_err_path = 0.0
        self.y_err_filt = 0.0
        self.ang_err_filt = 0.0
        self.latest_row_trav_time=None
        self.y_ref_ts = rospy.Time.now()
        self.ang_ref_ts = rospy.Time.now()
        self.final_node=None
        self.initial_node=None
        self.execute_callback_ts = rospy.Time.now()
        self.row_detector_timeout_s = 0.3  # Age cut-off for row detector input in seconds
        self.final_pose=None
        self.initial_pose=None
        self.offset = None
        self.row_poles = []
        self.use_closest_edges = False

        self.enable_laser_safety_zone = rospy.get_param("row_traversal/enable_laser_safety_zone", True)
        self.stop_on_overshoot = rospy.get_param("row_traversal/stop_on_overshoot", False)
        self.use_row_detector = rospy.get_param("row_traversal/use_row_detector", True)
        self.base_frame = rospy.get_param("row_traversal/base_frame", "base_link")
        self.corner_frames = rospy.get_param("row_traversal/corner_frames", ["top0", "top1", "top2", "top3"])
        self.wheel_steer_frames = rospy.get_param("row_traversal/wheel_steer_frames", ["leg0", "leg1", "leg2", "leg3"])

        rospy.loginfo("Setting up PID controllers.")
        self.setup_controllers()

        while not self.lnodes and not self.cancelled:
            rospy.loginfo("Waiting for topological map")
            rospy.Subscriber('/topological_map', TopologicalMap, self.topological_map_cb)
            if not self.lnodes:
                rospy.sleep(0.2)

        rospy.Subscriber('/scan', LaserScan, self.laser_cb, queue_size=1)
        rospy.Subscriber('/robot_pose', Pose, self.robot_pose_cb)
        rospy.Subscriber('/closest_node', String, self.closest_node_cb)
        rospy.Subscriber('/current_node', String, self.current_node_cb)
        rospy.Subscriber('/row_detector/path_error',Pose2D, self.row_correction_cb)
        rospy.Subscriber('/row_detector/obstacles', ObstacleArray, self.obstacles_callback,  queue_size=1)
        rospy.Subscriber('/teleop_joy/joy_priority', Bool, self.joy_lock_cb)
        rospy.Subscriber('/closest_edges', ClosestEdges, self.closest_edges_cb)
        #########################################################################################################    
        #### CHANGES NEEDED FOR HUMAN AWARE NAVIGATION ##########################################################
        #########################################################################################################
        rospy.Subscriber('robot_info',robot_msg,self.robot_callback)  
        rospy.Subscriber('human_safety_info',hri_msg,self.safety_callback)  
        #########################################################################################################
        #########################################################################################################
        self._tf_listerner = tf.TransformListener(cache_time=rospy.Duration(self.tf_buffer_size))

        self.ppub = rospy.Publisher('/row_traversal/row_line', Path, queue_size=1)
        self.cmd_pub = rospy.Publisher('/nav_vel', Twist, queue_size=1)
        self.safety_zone_vis_pub = rospy.Publisher('/row_traversal/safety_zone', Marker, queue_size=1)
        self.not_pub = rospy.Publisher('/row_traversal/notification', String, queue_size=1)
        self.health_pub = rospy.Publisher('/row_traversal/health', RowTraversalHealth, queue_size=1)

        health_publisher_timer = rospy.Timer(rospy.Duration(0.2), self.publish_health_timed_caller, oneshot=False)

        self.dyn_reconf_srv = Server(RowTraversalConfig, self.dyn_reconf_callback)

        rospy.loginfo("Creating safety zone.")
        self.define_safety_zone()

        rospy.loginfo("Creating action server.")
        self._as = actionlib.SimpleActionServer(self._action_name, polytunnel_navigation_actions.msg.inrownavAction, execute_cb = self.executeCallback, auto_start = False)
        self._as.register_preempt_callback(self.preemptCallback)
        rospy.loginfo(" ...starting")
        self._as.start()
        rospy.loginfo(" ...done")

        self.safety_zone_vis_pub.publish(self.safety_marker)


    def dyn_reconf_callback(self, config, level):
        """
           Dynamic reconfigure of parameters
           
           This function changes the internal variables modified through the dynamic reconfigure interface
        """
        rospy.loginfo("reconfigure request")
        print(config, level)

        if self.config:
            changed_dict = {x: self.config[x] != config[x] for x in self.config if x in config}
            lk = [key  for (key, value) in changed_dict.items() if value]
            if lk:
                print("config changed ", lk[0], config[lk[0]])
                if hasattr(self, lk[0]):
                    setattr(self, lk[0], config[lk[0]])
                    print(lk[0], getattr(self, lk[0]))
                    if lk[0] in self._reconf_functions['variables']:
                        self._reconf_functions['functions'][self._reconf_functions['variables'].index(lk[0])]()
            else:
                print("nothing changed!")
                print(self.force_forwards_facing, self.force_backwards_facing)

            #self.set_config(lk[0], config[lk[0]])
            self.config = config
        else:
            print("First config: ")#, config.items()
            self.config = config
            for i in config.items():
                if hasattr(self, i[0]):
                    setattr(self, i[0], i[1])
                    print(i[0], getattr(self, i[0]))
                    if i[0] in self._reconf_functions['variables']:
                        self._reconf_functions['functions'][self._reconf_functions['variables'].index(i[0])]()
        return config


    def reconf_tf_listener(self):
        print("Setting new tf buffer size to: ", self.tf_buffer_size)
        self._tf_listerner = tf.TransformListener(cache_time=rospy.Duration(self.tf_buffer_size))


    def define_safety_zone(self):
        """ Defines the Safety zone around the robot

        Arguments:
        clearance     -- The outwards distance trom the corner_frames at which the vertices of the safety zone are defined
        corner_frames -- The name of the frames that define the extremes of the robot
        """
        print("Defining Safety Zone")
        self.limits=[]
        self.emergency_base_points=[]

        for i in self.corner_frames:
            d={}
            self._tf_listerner.waitForTransform(self.base_frame,i,rospy.Time.now(), rospy.Duration(1.0))
            (trans,rot) = self._tf_listerner.lookupTransform(self.base_frame,i,rospy.Time.now())
            i_x, i_y, i_z = trans
            cpi=PointStamped()
            cpi.header.frame_id=self.base_frame
            cpi.point.x= i_x+self.emergency_clearance_x if i_x > 0 else  i_x-self.emergency_clearance_x
            cpi.point.y= i_y+self.emergency_clearance_y if i_y > 0 else  i_y-self.emergency_clearance_y
            cpi.point.z=-0.3
            d['point']=cpi
            d['angle']=math.atan2(cpi.point.y,cpi.point.x)
            self.emergency_base_points.append(d)

        # We sort in angle from move base
        self.emergency_base_points = sorted( self.emergency_base_points, key=lambda k: k['angle'])
        self.redefine_laser_regions = True
        print("visualise safety zone")
        self.safety_zones_visualisation()



    def publish_health_timed_caller(self, timer):
        self.publish_health()


    def publish_health(self):
## COMMENTED OUT TO FIX ACTION NOT CANCELLING WHEN TOPONAV GOAL IS CANCELLED 
#        if self.cancelled and rospy.get_time() - self.execute_callback_ts.to_sec() > 5:
#            self.cancelled = False
#            self.reset_controllers()
#            self.paused = False

        health_data = RowTraversalHealth()
        hd = Header()
        hd.stamp = rospy.Time.now()
        health_data.header = hd
        health_data.status.data = self.row_traversal_status                  # Status of row traversal, what are we currently doing OFF, ALIGN, FOLLOW, COLLISION, CANCELLED
        health_data.y_err_path.data = self.y_err_path                       # Error in Y to reference line
        if self.y_ref:                                     # Error in Y to detected row If a detected row reference is available
            health_data.y_err_row_detector.data = self.y_ref
        else:
            health_data.y_err_row_detector.data = 0.0
        health_data.ang_err_path.data =  self.ang_err_path                 # Orientation error to reference line
        if self.ang_ref:                                   # Angular error to detected row If a detected row reference is available
            health_data.ang_err_row_detector.data = self.ang_ref
        else:
            health_data.ang_err_row_detector.data = 0.0
        health_data.y_err_filt.data = self.y_err_filt
        health_data.ang_err_filt.data = self.ang_err_filt
        health_data.collision.data = self.collision
        health_data.cancelled.data = self.cancelled
        health_data.paused.data = self.paused

        self.health_pub.publish(health_data)



    def laser_cb(self, msg):
        if self.redefine_laser_regions:
            self.safety_zones_find_laser_regions(msg)
        elif self.laser_emergency_regions and self.active and self.enable_laser_safety_zone :
            min_range = min(x for x in msg.ranges if x > msg.range_min) # Necessary in case there are -1 in data
            self.collision=False
            points_inside = []
            angles_inside = []
            #print "min range: ", min_range, " -> ", self.max_emergency_dist#, " of ", len(minslist)
            if min_range<=self.max_emergency_dist:
                minslist = [(x, msg.ranges.index(x)) for x in msg.ranges if x <= self.max_emergency_dist]
                #print "min range: ", min_range, " -> ", self.max_emergency_dist, " of ", len(minslist)
                for i in minslist:
                    angle = (i[1]*msg.angle_increment)-msg.angle_min
                    p1=[(i[0]*np.cos(angle)),(i[0]*np.sin(angle))]
                    path = mplPath.Path(self.emergency_poly)
                    inside2 = path.contains_points([p1])
                    if inside2:
                        self.collision=True
                        self._send_velocity_commands(0.0, 0.0, 0.0)
                        degang = np.rad2deg(angle)
                        if degang>=360.0:
                            degang=degang-360.0
                        points_inside.append(p1)
                        angles_inside.append(angle)
                        #colstr = "HELP!: collision "+ str(degang) +" "+ str(i[0])+" "+ str(rospy.Time.now().secs)
                        #print colstr
                        if self.quit_on_timeout and not self.giveup_timer_active:
                            self.timer = rospy.Timer(rospy.Duration(self.time_to_quit), self.giveup, oneshot=True)
                            self.giveup_timer_active=True

                        if not self.notified and not self.notification_timer_active:
                            self.timer = rospy.Timer(rospy.Duration(3.0), self.nottim, oneshot=True)
                            self.notification_timer_active=True
                        #self.not_pub.publish(colstr)
                        break
            
            if len(points_inside) == 0:
                self.collision=False

            if not self.collision:
                self.notified=False
    
    
    def joy_lock_cb(self, msg):
        self.reset_controllers()
        if msg.data:
            self._user_controlled=True
            if self.goal_overshot:
                self.goal_overshot=False
        else:
            self._user_controlled=False
        
        
    def closest_edges_cb(self, msg):
        self.closest_edge = msg
        self.use_closest_edges = True
    

    def obstacles_callback(self, msg):
        min_obs_dist=1000
        for obs in msg.obstacles:
            if obs.radius > self.min_obj_size:
                obs_pose = self._transform_to_pose_stamped(obs)
                cobs_dist = np.hypot(obs_pose.pose.position.x, obs_pose.pose.position.y)
                if cobs_dist < min_obs_dist:
                    obs_dist = cobs_dist
                    min_obs_dist = obs_dist
                    obs_ang = math.atan2(obs_pose.pose.position.y, obs_pose.pose.position.x)
                    if obs_ang > np.pi:
                        obs_ang = obs_ang - 2*np.pi

                    #print "Obstacle Size: ", obs.radius, " detected at ", obs_ang, " degrees "#, obs_dist," meters away",  self.backwards_mode


        if min_obs_dist <= self.approach_dist_to_obj:
            if (np.abs(obs_ang) <= (np.pi/25.0)) or (np.abs(obs_ang) >= (np.pi*24/25.0)):
                if np.abs(obs_ang) < np.pi/2.0 :#and self.backwards_mode:
                    #print "Obstacle Size: ", obs.radius, " detected at ", obs_ang, " degrees ", obs_dist," meters away",  self.backwards_mode
                    self.object_detected = True
                    self.curr_distance_to_object=obs_dist
                elif np.abs(obs_ang) > np.pi/2.0 :#and not self.backwards_mode:
                    #print "Obstacle Size: ", obs.radius, " detected at ", obs_ang, " degrees ", obs_dist," meters away",  self.backwards_mode
                    self.object_detected = True
                    self.curr_distance_to_object=obs_dist
                else:
                    self.object_detected = False
                    self.curr_distance_to_object=-1.0
        else:
            self.object_detected = False
            self.curr_distance_to_object=-1.0



    def safety_zones_find_laser_regions(self, msg):
        self.laser_emergency_regions=[]
        for i in range(len(self.emergency_base_points)-1):
            d = {}
            d['range'] = []
            d['range'].append(int(np.floor((self.emergency_base_points[i]['angle']-msg.angle_min)/msg.angle_increment)))
            d['range'].append(int(np.floor((self.emergency_base_points[i+1]['angle']-msg.angle_min)/msg.angle_increment)))
            midx=(self.emergency_base_points[i]['point'].point.x + self.emergency_base_points[i+1]['point'].point.x)/2.0
            midy=(self.emergency_base_points[i]['point'].point.y + self.emergency_base_points[i+1]['point'].point.y)/2.0
            d['dist']= math.hypot(self.emergency_base_points[i]['point'].point.x, self.emergency_base_points[i+1]['point'].point.y)
            d['mean_dist']=math.hypot(midx, midy)
            self.laser_emergency_regions.append(d)

        d = {}
        d['range'] = []
        d['range'].append(int(np.floor((self.emergency_base_points[-1]['angle']-msg.angle_min)/msg.angle_increment)))
        d['range'].append(int(np.floor((self.emergency_base_points[0]['angle']-msg.angle_min)/msg.angle_increment)))
        midx=(self.emergency_base_points[0]['point'].point.x + self.emergency_base_points[-1]['point'].point.x)/2.0
        midy=(self.emergency_base_points[0]['point'].point.y + self.emergency_base_points[-1]['point'].point.y)/2.0
        d['dist']= d['dist']= math.hypot(self.emergency_base_points[0]['point'].point.x, self.emergency_base_points[-1]['point'].point.y)
        d['mean_dist']=math.hypot(midx, midy)
        self.laser_emergency_regions.append(d)

        self.emergency_poly=[]
        for i in self.laser_emergency_regions:
            print(i)

        for i in self.emergency_base_points:
            r=(i['point'].point.x, i['point'].point.y)
            self.emergency_poly.append(r)

        self.emergency_poly=np.asarray(self.emergency_poly)
        self.redefine_laser_regions=False

        self.max_emergency_dist= 0.0
        for i in self.laser_emergency_regions:
            print(i['dist'])
            self.max_emergency_dist=np.max([self.max_emergency_dist, i['dist']])
#        laser_angles.append(msg.angle_max)


    def safety_zones_visualisation(self):

        base_pose = Pose()
        base_pose.orientation.w=1.0

        amarker = Marker()
        amarker.header.frame_id = self.base_frame
        #amarker.header.stamp = rospy.Time.now()
        amarker.type = 4
        amarker.pose = Pose()
        amarker.pose.position.z = 0.51
        amarker.scale.x = 0.05
        amarker.color.a = 0.5
        amarker.color.r = 0.9
        amarker.color.g = 0.1
        amarker.color.b = 0.1
        amarker.lifetime = rospy.Duration(0.0)
        amarker.frame_locked = True

        for i in self.emergency_base_points:
            amarker.points.append(i['point'].point)
        amarker.points.append(self.emergency_base_points[0]['point'].point)

        self.safety_marker=amarker


    def row_correction_cb(self, msg):
        if np.isnan((msg.y)):
            self.y_ref=None
        else:
            self.y_ref_ts = rospy.Time.now()
            self.y_ref=msg.y

        if np.isnan((msg.theta)):
            self.ang_ref=None
        else:
            self.ang_ref_ts = rospy.Time.now()
            self.ang_ref=msg.theta


    def topological_map_cb(self, msg):
        self.lnodes=msg

        # Get all row trav edges in topomap
        action_edges = []
        action_nodes = {}
        for node in self.lnodes.nodes:
                for edge in node.edges:
                    if edge.action == self._action_name.strip("/").split("/")[-1]:
                        action_edges.append(edge.edge_id.split("_"))
                        action_nodes[node.name] = node.pose

        self.action_edges = action_edges
        self.action_nodes = action_nodes

    def robot_pose_cb(self, msg):
        self.robot_pose = msg


    def closest_node_cb(self, msg):
        if self.closest_node_cb != msg.data:
            self.closest_node=msg.data
            print(self.closest_node)
                
                
    def current_node_cb(self, msg):
        if self.current_node_cb != msg.data:
            self.current_node=msg.data
            print(self.current_node)


    def get_node_position(self, node):
        pose=None
        for i in self.lnodes.nodes:
            if i.name == node:
                pose = i.pose
                break
        return pose


    def get_node_from_pose(self, pose):
        """ Get the name of the goal node """
        name=None
        for i in self.lnodes.nodes:
            if i.pose == pose:
                name = i.name
                break
        return name


    def get_current_edge(self, goal_node):
        """ Get the row traversal edge we are closest too """
        
        if self.current_node != "none":
            current_edge = [self.current_node, goal_node]
            return current_edge, 0.0
        
        if self.use_closest_edges:
            min_dist = self.closest_edge.distances[0]
            current_edge = None
            
            print(min_dist, self.closest_edge)
            
            for node in self.lnodes.nodes:
                for edge in node.edges:
                    if edge.action == self._action_name.strip("/").split("/")[-1] and (edge.edge_id == self.closest_edge.edge_ids[0] or edge.edge_id == self.closest_edge.edge_ids[1]):
                        print(edge.node, node.name, goal_node)
                        if goal_node == node.name:
                            current_edge = [edge.node, node.name]
                        elif goal_node == edge.node:
                        #action_edges.append(edge.edge_id.split("_"))
                            current_edge = [node.name, edge.node]
                    #elif edge.action == self._action_name.strip("/").split("/")[-1] and edge.edge_id == self.closest_edge.edge_ids[1]:
            if current_edge:
                print("Got current edge")
                return current_edge, min_dist

        print("closest edges do not match recalculating!!")
        self.closest_node = rospy.wait_for_message('/closest_node', String)
        min_dist = None
        current_edge = None
        for edge in self.action_edges:
            if self.closest_node.data in edge and goal_node == edge[1]:
                dist, _ = pnt2line([self.robot_pose.position.x, self.robot_pose.position.y, 0],  # Current pose
                                   [self.action_nodes[edge[0]].position.x, self.action_nodes[edge[0]].position.y, 0],  # Start
                                   [self.action_nodes[edge[1]].position.x, self.action_nodes[edge[1]].position.y, 0])  # End
                if not min_dist or dist < min_dist:
                    min_dist = dist
                    current_edge = edge

        return current_edge, min_dist


    def _distance_between_poses(self, posea, poseb):
        ang = math.atan2(poseb.position.y-posea.position.y, poseb.position.x-posea.position.x)
        the_quat = Quaternion(*tf.transformations.quaternion_from_euler(0.0, 0.0, ang))
        return math.hypot(poseb.position.x-posea.position.x, poseb.position.y-posea.position.y), the_quat, ang


    def _transform_to_pose_stamped(self, pose_2d):
        the_pose = PoseStamped()
        the_pose.header.frame_id = self.base_frame
        the_pose.pose.position.x = pose_2d.pose.x#-1.0*pose_2d.pose.x
        the_pose.pose.position.y = pose_2d.pose.y
        the_pose.pose.position.z = 0.5
        the_pose.pose.orientation.w = 1.0
        return the_pose


    def _get_angle_between_quats(self, ori1, ori2):
        q1 = PyKDL.Rotation.Quaternion(ori1.x, ori1.y, ori1.z, ori1.w)
        q2 = PyKDL.Rotation.Quaternion(ori2.x, ori2.y, ori2.z, ori2.w)

        ang1 = q1.GetRPY()
        ang2 = q2.GetRPY()

        return ang2[2]-ang1[2]


    def get_current_row(self):
        """ Finds the current row and sorts the poles by distance from start """
        if len(self.pole_pos_per_row) == 0:
            rospy.logwarn("No pole file so cant get current row")
            return None

        self.closest_pole_behind = 0
        self.closest_pole_ahead = 1
        self.pole_array = []

        min_dist = None
        min_inds = None
        min_ind = 0
        i=0
        inds = []
        for row_poles in self.pole_pos_per_row:
            xs = row_poles[0][:, 0]
            ys = row_poles[0][:, 1]
            dists = []
            ind = 0
            inds = []
            for dx, dy in zip(xs, ys):
                if dx != self.initial_node_original.position.x and dx != self.final_node_original.position.x and dy != self.initial_node_original.position.y and dy != self.final_node_original.position.y:
                    # print "Inodep: ", self.initial_node_original.position.x, self.initial_node_original.position.y, "Finodep: ", self.final_node_original.position.x, self.final_node_original.position.y
                    # print "pole x y : ", dx, dy
                    dist, nearest = pnt2line([dx, dy, 0], [self.initial_node_original.position.x, self.initial_node_original.position.y, 0], [self.final_node_original.position.x, self.final_node_original.position.y, 0])

                    if dist < 0.6:
                        dists.append(dist)
                        inds.append(ind)
                        obstacle = Obstacle()
                        obstacle = self.fill_obstacle_msg(obstacle,dx,dy)
                        self.pole_array.append(obstacle)
                        if dist > 0.1:
                            rospy.logwarn("There is at least 1 pole offset more than 10cm from this row.")
                ind += 1
            mean_dist = np.mean(dists)
            if len(dists) > 1:
                if not min_dist or (mean_dist < min_dist):
                    min_dist = mean_dist
                    min_ind = i
                    min_inds = inds
            i+=1

        if not min_inds:
            row_poles = np.array([[self.initial_node.position.x, self.initial_node.position.y],
                                  [self.final_node.position.x, self.final_node.position.y]])
        else:
            row_poles = self.pole_pos_per_row[min_ind][0]
            row_poles = row_poles[min_inds]
            row_poles = np.append(row_poles, [[self.initial_node.position.x, self.initial_node.position.y],
                                              [self.final_node.position.x, self.final_node.position.y]],
                                              axis=0)
        obstacle = Obstacle()
        obstacle = self.fill_obstacle_msg(obstacle,self.initial_node.position.x, self.initial_node.position.y)
        self.pole_array.append(obstacle)
        obstacle = Obstacle()
        obstacle = self.fill_obstacle_msg(obstacle,self.final_node.position.x, self.final_node.position.y)
        self.pole_array.append(obstacle)

        # print "added nodes:", row_poles
        xs = row_poles[:, 0]
        ys = row_poles[:, 1]
        dxs = self.initial_node.position.x - xs
        dys = self.initial_node.position.y - ys
        dists = np.sqrt(dxs**2 + dys**2)
        inds = np.argsort(dists)
        # print "inds: ", inds
        sorted_dists = dists[inds]
        self.row_poles = row_poles[inds]
        self.pole_array = [self.pole_array[i] for i in inds]
        # print "initial dists:", sorted_dists
        corrected=True
        while corrected:
            corrected = False
            rm = []
            for i in np.arange(len(sorted_dists)-1):
                if np.sqrt((sorted_dists[i] - sorted_dists[i+1])**2) < 1.0:
                    rm.append(i)
                    corrected = True
            if corrected:
                sorted_dists = np.delete(sorted_dists, rm)
                self.row_poles = np.delete(self.row_poles, rm, axis=0)
                for i in sorted(rm, reverse=True):
                    del self.pole_array[i]

        self.row_dists = sorted_dists
        # print "sorted poles:", self.row_poles
        # print "sorted dists:", self.row_dists


    def get_row_offset(self):

        if self.row_poles == []:
            self.get_current_row()
        if self.row_poles == []:
            rospy.logwarn("Haven't found row so cant offset row detector")
            return None

        min_dist = None
        xs = self.row_poles[:, 0]
        ys = self.row_poles[:, 1]
        dxs = self.robot_pose.position.x - xs
        dys = self.robot_pose.position.y - ys
        ind = np.argmin(np.sqrt(dxs**2 + dys**2))
        dists = []
        for dx, dy in zip(xs, ys):
            dist, nearest = pnt2line([dx, dy, 0], [self.initial_node_original.position.x, self.initial_node_original.position.y, 0], [self.final_node_original.position.x, self.final_node_original.position.y, 0])
            if dist < 0.6:
                dists.append(dist)
        mean_dist = np.mean(dists)
        if len(dists) > 1:
            min_dist = mean_dist
            closest_pole_x = xs[1]
            closest_pole_y = ys[1]

        if not min_dist or min_dist > 0.6:
            return None

        dist, nearest = pnt2line([closest_pole_x, closest_pole_y, 0], [self.initial_node_original.position.x, self.initial_node_original.position.y, 0], [self.final_node_original.position.x, self.final_node_original.position.y, 0])

        closest_pose = PoseStamped(pose=Pose(position=Point(x=nearest[0], y=nearest[1])),
                                   header=Header(frame_id='map'))
        closest_pole = PoseStamped(pose=Pose(position=Point(x=closest_pole_x, y=closest_pole_y)),
                                   header=Header(frame_id='map'))

        base_to_line = self._tf_listerner.transformPose(self.base_frame,closest_pose)
        base_to_pole = self._tf_listerner.transformPose(self.base_frame,closest_pole)

        offset = min_dist*np.sign(base_to_pole.pose.position.y - base_to_line.pose.position.y)

        # This is used to shift the start and end positions back in line with
        # the poles, so we can then apply the offset to them in the same way.
        self.offset_vector =[closest_pole_x-closest_pose.pose.position.x, closest_pole_y-closest_pose.pose.position.y]

        return offset


    def apply_offset_to_nodes(self):
        self.initial_node.position.x = self.initial_node_original.position.x + self.offset_vector[0]
        self.initial_node.position.y = self.initial_node_original.position.y + self.offset_vector[1]
        self.final_node.position.x = self.final_node_original.position.x + self.offset_vector[0]
        self.final_node.position.y = self.final_node_original.position.y + self.offset_vector[1]

        # Below is added to stop refilling in self.get_vector_to_pose()
        radius, the_quat, ang = self._distance_between_poses(self.initial_node, self.final_node)
        self.goal_pose = PoseStamped(pose=Pose(position=Point(x=self.final_node.position.x,
                                                              y=self.final_node.position.y),
                                               orientation=the_quat),
                                     header=Header(frame_id='map'))

        self.get_current_row()


    def fill_obstacle_msg(self, msg, x, y):
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.x = x
        msg.pose.y = y
        msg.pose.theta = np.nan
        msg.radius = 0.15

        return msg


    def get_line_from_pole_file(self):
        dist_along_row = math.hypot(self.initial_node.position.x-self.robot_pose.position.x, self.initial_node.position.y-self.robot_pose.position.y)
        dist_diff = self.row_dists-dist_along_row

        while self.closest_pole_behind >= -1 and self.closest_pole_behind < len(self.row_poles)-1:
            self.closest_pole_behind = 0 if self.closest_pole_behind < 0 else self.closest_pole_behind;
            if dist_diff[self.closest_pole_behind] >= 0:
                if self.closest_pole_behind <= 0:
                    self.closest_pole_behind = -1
                    break
                self.closest_pole_behind -= 1
            elif dist_diff[self.closest_pole_behind] < -0:
                if dist_diff[self.closest_pole_behind+1] > 0:
                    break
                self.closest_pole_behind += 1

        self.closest_pole_ahead = self.closest_pole_behind + 1

        while self.closest_pole_ahead < len(self.row_poles) and self.closest_pole_ahead >= 0:
            if dist_diff[self.closest_pole_ahead] <= 0:
                if self.closest_pole_ahead >= len(self.row_poles-1):
                    self.closest_pole_ahead = len(self.row_poles)
                    break
                self.closest_pole_ahead += 1
            elif dist_diff[self.closest_pole_ahead] > 0:
                if dist_diff[self.closest_pole_ahead-1] < 0 or self.closest_pole_ahead <= self.closest_pole_behind+1:
                    break
                self.closest_pole_ahead -= 1

        # print "behind pole: ", self.closest_pole_behind, "ahead pole: ", self.closest_pole_ahead

        if self.closest_pole_behind >= 0:
            self.initial_pose.position.x = self.row_poles[self.closest_pole_behind,0]
            self.initial_pose.position.y = self.row_poles[self.closest_pole_behind,1]
        else:
            self.initial_pose = deepcopy(self.initial_node)

        if self.closest_pole_ahead < len(self.row_poles):
            self.final_pose.position.x = self.row_poles[self.closest_pole_ahead,0]
            self.final_pose.position.y = self.row_poles[self.closest_pole_ahead,1]
        else:
            self.final_pose = deepcopy(self.final_node)

        ba_poles = TwoInts(a=Int32(data=self.closest_pole_behind),
                           b=Int32(data=self.closest_pole_ahead))
        poles = ObstacleArray(obstacles=self.pole_array)
        self.behind_ahead_poles_pub.publish(ba_poles)
        self.poles_pub.publish(poles)

    def check_direction(self):

        if self.force_forwards_facing or self.force_backwards_facing:
            if self.force_forwards_facing:
                self.backwards_mode=False
                print("we MUST be going forwards")
            elif  self.force_backwards_facing:
                self.backwards_mode=True
                print("we MUST be going backwards")
        else:
            ang_diff = self.get_angle()
            if ang_diff > (math.pi/2.0) or ang_diff < -(math.pi/2.0):
                self.backwards_mode=True
                print("we should be going backwards")
            else:
                self.backwards_mode=False
                print("forwards heading")


    def align_orientation(self):
        corrected = False
        self.row_traversal_status = "ALIGN"

        dist, y_err, ang_diff = self._get_vector_to_pose()
        if self.prealign_y_axis and np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
            corrected = True
            print("Aligning ang based on path")
            while np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
                self._send_velocity_commands(0.0, 0.0, self.controller_ang.update(ang_diff), consider_minimum_rot_vel=True)
                rospy.sleep(self._controller_rate)
                dist, y_err, ang_diff = self._get_vector_to_pose()
                # print "Aligning ang path", ang_diff
                self.row_traversal_status = "ALIGN"

        dist, y_err, ang_diff = self._get_vector_to_pose()
        if self.prealign_y_axis and np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
            print("Aligning y based on path")
            corrected = True
            self.check_wheel_direction(direction='y')
            while np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                self._send_velocity_commands(0.0, self.controller_y.update(y_err), 0.0, consider_minimum_rot_vel=True)
                rospy.sleep(self._controller_rate)
                dist, y_err, ang_diff = self._get_vector_to_pose()
                # print "Aligning Y path", y_err
                self.row_traversal_status = "ALIGN"

        if self.use_row_detector:
            self.activate_row_detector(True)

        dist, y_err, ang_diff = self._get_references()
        if self.prealign_y_axis and np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
            print("re-aligning ang based on detection")
            corrected = True
            while np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
                self._send_velocity_commands(0.0, 0.0, self.controller_ang.update(ang_diff), consider_minimum_rot_vel=True)
                rospy.sleep(self._controller_rate)
                dist, y_err, ang_diff = self._get_references()
                # print "Aligning ang detection", ang_diff
                self.row_traversal_status = "ALIGN"


        dist, y_err, ang_diff = self._get_references()
        if self.prealign_y_axis and np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
            print("re-aligning y based on detection")
            corrected = True
            self.check_wheel_direction(direction='y')
            while np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                self._send_velocity_commands(0.0, self.controller_y.update(y_err), 0.0, consider_minimum_rot_vel=True)
                rospy.sleep(self._controller_rate)
                dist, y_err, ang_diff = self._get_references()
                # print "Aligning path detection", y_err
                self.row_traversal_status = "ALIGN"


    def enter_row(self):
        print("Entering row")
        self.row_traversal_status = "FOLLOW"
        overall_goal_distance = math.hypot(self.final_node.position.x-self.robot_pose.position.x, self.final_node.position.y-self.robot_pose.position.y)
        # print 'overall goal distance =', overall_goal_distance
        
        dist, y_err, ang_diff = self._get_references(publish_health=True)
        pre_gdist=dist
        
        if not self.cancelled:
            self.check_wheel_direction(direction='x')

        distance_travelled = 0
        remaining_goal_distance = overall_goal_distance
        self.reset_controllers()
        while distance_travelled < self.row_entry_distance and not self.cancelled and np.abs(dist) > self.goal_tolerance_radius and not self.goal_overshot:
            self.latest_row_trav_time = rospy.Time.now()

            if self.cancelled:
                print("leaveing row entry as self.cancelled = ", self.cancelled)
                break 
            dist, y_err, ang_diff = self._get_references(publish_health=True)
            speed=self.get_row_entry_speed(distance_travelled)

            self._send_velocity_commands(speed, self.controller_y.update(y_err), self.controller_ang.update(ang_diff))
            rospy.sleep(self._controller_rate)

            self.row_traversal_status = "FOLLOW"
            remaining_goal_distance = math.hypot(self.final_node.position.x-self.robot_pose.position.x, self.final_node.position.y-self.robot_pose.position.y)
            distance_travelled = np.abs(overall_goal_distance - remaining_goal_distance)

            # check for overshoot
            if not self._user_controlled:
                if np.sign(dist) != np.sign(pre_gdist):
                    self.goal_overshot= True
                    nottext="Row traversal has overshoot, previous distance "+str(np.abs(pre_gdist))+" current distance "+str(np.abs(dist))
                    self.not_pub.publish(nottext)
                    rospy.logwarn(nottext)


        if remaining_goal_distance > overall_goal_distance:
            print("WE HAVE GONE THE WRONG WAY!!!!")
            self.cancelled = True


    def get_row_entry_speed(self, distance_travelled):
        #########################################################################################################    
        #### CHANGES NEEDED FOR HUMAN AWARE NAVIGATION ##########################################################
        #########################################################################################################
        if self.hri_safety_action==0 or self.hri_safety_action==2 or self.hri_safety_action==5: #execute if safety action is "going to goal" or "moving away" or "no safety action"
            if not self.constant_forward_speed:
                if not self.object_detected and self.curr_distance_to_object <= self.approach_dist_to_obj:
                    speed = self.row_entry_min_speed + max(0.0, (self.row_entry_kp*distance_travelled))
                else:
                    slowdown_delta = self.approach_dist_to_obj - self.min_dist_to_obj
                    current_percent = (self.curr_distance_to_object - self.min_dist_to_obj) / slowdown_delta
                    if current_percent >0:
                        speed = min((current_percent*self.forward_speed), (self.row_entry_min_speed + max(0.0, (self.row_entry_kp*distance_travelled))))
                    else:
                        speed = 0.0
            else:
                speed = self.row_entry_min_speed + max(0.0, (self.row_entry_kp*distance_travelled))
        
        else: #execute if safety action is "reduce speed" or "stop" or "pause"
            if self.hri_safety_action==1: #if safety action is "reduce speed" while approaching
                dist=self.hri_dist  
                if dist <= self.han_start_dist:
                    slowdown_delta = self.han_start_dist - self.han_final_dist
                    current_percent = (dist - self.han_final_dist) / slowdown_delta
                    if current_percent >0:
                        #print("Limiting speed")
                        speed = (current_percent*self.forward_speed)
                    else:
                        #print("stop")
                        speed = 0.0
                else:
                    speed = self.row_entry_min_speed + max(0.0, (self.row_entry_kp*distance_travelled))
            elif self.hri_safety_action==3 or self.hri_safety_action==4: #if a safety action is "stop" or if the robot is "waiting for a human command" 
                #print("stop")
                speed = 0.0
        if self.backwards_mode:
            speed = -speed
        #################################################################################################
        ###################################################################################################
        return speed
    

    def check_wheel_direction(self, direction='x'):
        # Check that the wheels are all facing the right way before we proceed
        # TODO: Make this work for an arbritrary number of wheels, not just 4
        print('Checking wheel direction:'),
        wheels_aligned = False
        while not wheels_aligned and not self.cancelled:
            now=rospy.Time(0)
            try:
                self._tf_listerner.waitForTransform(self.base_frame, self.wheel_steer_frames[0], now, rospy.Duration(4.0))
                _, wheel_rot0 = self._tf_listerner.lookupTransform(self.base_frame, self.wheel_steer_frames[0], now)
                self._tf_listerner.waitForTransform(self.base_frame, self.wheel_steer_frames[1], now, rospy.Duration(4.0))
                _, wheel_rot1 = self._tf_listerner.lookupTransform(self.base_frame, self.wheel_steer_frames[1], now)
                self._tf_listerner.waitForTransform(self.base_frame, self.wheel_steer_frames[2], now, rospy.Duration(4.0))
                _, wheel_rot2 = self._tf_listerner.lookupTransform(self.base_frame, self.wheel_steer_frames[2], now)
                self._tf_listerner.waitForTransform(self.base_frame, self.wheel_steer_frames[3], now, rospy.Duration(4.0))
                _, wheel_rot3 = self._tf_listerner.lookupTransform(self.base_frame, self.wheel_steer_frames[3], now)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                wheels_aligned = False
                continue

            _, _, wheel_yaw0 =  euler_from_quaternion(wheel_rot0)
            _, _, wheel_yaw1 =  euler_from_quaternion(wheel_rot1)
            _, _, wheel_yaw2 =  euler_from_quaternion(wheel_rot2)
            _, _, wheel_yaw3 =  euler_from_quaternion(wheel_rot3)
            # print(wheel_yaw0,wheel_yaw1,wheel_yaw2,wheel_yaw3)

            if direction == 'x':
                if abs(wheel_yaw0) < 0.08 and abs(wheel_yaw1) < 0.08 and abs(wheel_yaw2) < 0.08 and abs(wheel_yaw3) < 0.08:
                    wheels_aligned = True
                else:
                    self._send_velocity_commands(0,0,0)

            if direction == 'y':
                if abs(abs(wheel_yaw0)-math.pi/2) < 0.08 and abs(abs(wheel_yaw1)-math.pi/2) < 0.08 and abs(abs(wheel_yaw2)-math.pi/2) < 0.08 and abs(abs(wheel_yaw3)-math.pi/2) < 0.08:
                    wheels_aligned = True
                else:
                    self._send_velocity_commands(0,0.000001,0)

            if self.paused:
                self._get_references()
            rospy.sleep(self._controller_rate)
        
        if self.cancelled:
            print('Cancelled during wheel check')
        else:
            print('wheel check passed')



    def get_forward_speed(self):
        #########################################################################################################    
        #### CHANGES NEEDED FOR HUMAN AWARE NAVIGATION ##########################################################
        #########################################################################################################
        if self.hri_safety_action==0 or self.hri_safety_action==2 or self.hri_safety_action==5: #execute if safety action is "going to goal" or "moving away" or "no safety action"
            if not self.constant_forward_speed:
                if not self.object_detected and self.curr_distance_to_object <= self.approach_dist_to_obj:
                    #print "not limiting"
                    if self.backwards_mode:
                        speed = -self.forward_speed
                    else:
                        speed = self.forward_speed
                else:
                    slowdown_delta=self.approach_dist_to_obj-self.min_dist_to_obj
                    current_percent = (self.curr_distance_to_object-self.min_dist_to_obj)/slowdown_delta
                    if current_percent >0:
                        speed = current_percent*self.forward_speed
                    else:
                        speed = 0.0
                    if self.backwards_mode:
                        speed = -speed
            else:
                if self.backwards_mode:
                    speed = -self.forward_speed
                else:
                    speed = self.forward_speed
        else:  #execute if safety action is "reduce speed" or "stop" or "pause"
            if self.hri_safety_action==1: #if safety action is "reduce speed" while approaching
                dist=self.hri_dist  
                if dist <= self.han_start_dist:
                    slowdown_delta = self.han_start_dist - self.han_final_dist
                    current_percent = (dist - self.han_final_dist) / slowdown_delta
                    if current_percent >0:
                        #print("Limiting speed")
                        speed = (current_percent*self.forward_speed)
                    else:
                        #print("stop")
                        speed = 0.0
                    if self.backwards_mode:
                            speed = -speed
                else:
                    if self.backwards_mode:
                        speed = -self.forward_speed
                    else:
                        speed = self.forward_speed
            elif self.hri_safety_action==3 or self.hri_safety_action==4: #if a safety action is "stop" or if the robot is "waiting for a human command" 
                print("stop")
                speed = 0.0
        #############################################################################################################
        #############################################################################################################
        return speed


    def go_forwards(self):
        self.row_traversal_status = "FOLLOW"
        if self.backwards_mode:
            speed = -self.forward_speed
        else:
            speed = self.forward_speed

        if not self.cancelled:
            print("Going forwards")
            self.check_wheel_direction(direction='x')

        dist, y_err, ang_diff = self._get_references(publish_health=True)

        # Hack to stop the robot if it overshot
        self.goal_overshot=False
        pre_gdist=dist

        self.reset_controllers()
        while not self.cancelled and not self.goal_overshot:
            # print "DIST: ", dist
            self.latest_row_trav_time = rospy.Time.now()
            dist, y_err, ang_diff = self._get_references(publish_health=True)

            if not dist or not y_err or not ang_diff:
                continue

            if self.cancelled or self.goal_overshot or np.abs(dist) < self.goal_tolerance_radius:
                break
            else:
                speed=self.get_forward_speed()
                self._send_velocity_commands(speed, self.controller_y.update(y_err), self.controller_ang.update(ang_diff))

            rospy.sleep(self._controller_rate)


            progress_to_goal=np.abs(pre_gdist)-np.abs(dist)
            if not self._user_controlled:
                if np.sign(dist) != np.sign(pre_gdist):
                    self.goal_overshot= True
                    nottext="Row traversal has overshoot, previous distance "+str(np.abs(pre_gdist))+" current distance "+str(np.abs(dist))
                    print(nottext)
                    print(progress_to_goal, dist, pre_gdist)
                    self.not_pub.publish(nottext)
                    rospy.logwarn(nottext)

            self.row_traversal_status = "FOLLOW"

        self._send_velocity_commands(0.0, 0.0, 0.0)
        self.active=False
        self.row_traversal_status = "OFF"

        if not self.cancelled:
            return True
        else:
            return False



    def follow_path(self, continue_row_trav):

        self.check_direction()

        self.reset_controllers()
        self.setup_low_pass_filters()

        if not continue_row_trav:
            # print "Align Orientation first"
            self.align_orientation()
            if self.use_row_entry_function:
                # print "Enter row"
                self.enter_row()
        elif self.use_row_detector:
            self.activate_row_detector(True)

        # print "Now go forwards!!"
        row_trav_complete = self.go_forwards()

        return row_trav_complete


    def setup_controllers(self):
        self.controller_ang = PID(P=self.kp_ang, D=self.kd_ang, I=self.ki_ang)
        self.controller_y = PID(P=self.kp_y, D=self.kd_y, I=self.ki_y)
        print("PID Controller ang: ", self.controller_ang.Kp, self.controller_ang.Ki, self.controller_ang.Kd)
        print("PID Controller y: ", self.controller_y.Kp, self.controller_y.Ki, self.controller_y.Kd)

    def setup_low_pass_filters(self):
        self.lpf_ang = LowPassFilter(fc=self.cutoff_freq_ang, fs=self._controller_freq)
        self.lpf_y = LowPassFilter(fc=self.cutoff_freq_y, fs=self._controller_freq)


    def reset_controllers(self):
        self.controller_ang.reset()
        self.controller_y.reset()


    def _x_speed_limiter(self, xvel):
        '''
        Function to limit the speed in the X axis according with the x_speed_limit parameter
        '''
        if abs(xvel) > self.forward_speed:
            if xvel < 0:
                return (-1.0*self.forward_speed)
            else:
                return self.forward_speed
        else:
            return xvel


    def _y_speed_limiter(self, yvel):
        '''
        Function to limit the speed in the Y axis according with the y_speed_limit parameter
        '''
        if abs(yvel) > self.y_speed_limit:
            if yvel < 0:
                return (-1.0*self.y_speed_limit)
            else:
                return self.y_speed_limit
        else:
            return yvel


    def _ang_speed_limiter(self, angvel):
        '''
        Function to limit the rotation speed according with the turning_speed_limit parameter
        '''
        if abs(angvel) > self.turning_speed_limit:
            if angvel < 0:
                return (-1.0*self.turning_speed_limit)
            else:
                return self.turning_speed_limit
        else:
            return angvel


    def _send_velocity_commands(self, xvel, yvel, angvel, consider_minimum_rot_vel=False):
        #print self.collision
        if not self.collision and not self.paused:
            cmd_vel = Twist()
            cmd_vel.linear.x = self._x_speed_limiter(xvel)
            cmd_vel.linear.y = self._y_speed_limiter(yvel)
            if consider_minimum_rot_vel:
                if np.isclose(angvel, 0.0):
                    cmd_vel.angular.z = 0.0
                elif np.abs(angvel) >= self.minimum_turning_speed:
                    cmd_vel.angular.z = self._ang_speed_limiter(angvel)
                else:
                    if angvel > 0.0:
                        cmd_vel.angular.z = self.minimum_turning_speed
                    elif angvel < 0.0:
                        cmd_vel.angular.z = -1.0 * self.minimum_turning_speed
            else:
                cmd_vel.angular.z = self._ang_speed_limiter(angvel)

    #        print 'ANg: ', cmd_vel.angular.z
            self.cmd_pub.publish(cmd_vel)
        else:
            if self.active:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                cmd_vel.angular.z = 0.0
                self.cmd_pub.publish(cmd_vel)



    def _get_references(self, publish_health=False):
        using_row_yref = False
        using_row_angref = False
        use_row_detector = self.use_row_detector

        if not self.offset_row_detector:
            self.offset = None
        elif self.offset_row_detector and self.offset == None:
            self.offset = self.get_row_offset()
            self.apply_offset_to_nodes()
            if self.offset == None:
                use_row_detector = False
            else:
                use_row_detector = self.use_row_detector
            print("Row detector offset = ", self.offset)

        dist, y_path_err, ang_path_diff = self._get_vector_to_pose()

        y_ref = self.y_ref
        if use_row_detector and y_ref and y_path_err and rospy.get_time() - self.y_ref_ts.to_sec() < self.row_detector_timeout_s:
    #            if self.backwards_mode:
#                self.y_ref = -1.0*self.y_ref
            using_row_detector_yref = True
            y_ref = y_ref if not self.offset else y_ref-self.offset
            y_err = np.average([y_ref, y_path_err], weights=[self.y_row_detection_bias, self.y_path_following_bias])
        elif y_path_err:
            y_err = y_path_err
        elif use_row_detector and y_ref and rospy.get_time() - self.y_ref_ts.to_sec() < self.row_detector_timeout_s:
            y_err = y_ref if not self.offset else y_ref-self.offset
        else:
            y_err = None

        ang_ref = self.ang_ref
        if ang_ref and ang_path_diff and rospy.get_time() - self.ang_ref_ts.to_sec() < self.row_detector_timeout_s:
            using_row_detector_angref = True
            ang_diff = np.average([ang_ref, ang_path_diff], weights=[self.ang_row_detection_bias, self.ang_path_following_bias])
        elif ang_path_diff:
            ang_diff = ang_path_diff
        elif ang_ref and rospy.get_time() - self.ang_ref_ts.to_sec() < self.row_detector_timeout_s:
            ang_diff = ang_ref
        else:
            ang_diff = None

        if not dist or not ang_diff or not y_err:
            return None, None, None

        current_max_angle_error = max( abs(ang_diff), abs(ang_path_diff), abs(ang_diff-ang_path_diff) )
        current_max_path_error = max( abs(y_err), abs(y_path_err), abs(ang_diff-ang_path_diff) )

        if self.active and not self.cancelled:
            if (current_max_angle_error > self.maximum_dev_dist_theta or current_max_path_error > self.maximum_dev_dist_y):
                rospy.logerr("Line deviation is too high, y_error is %f angle error is %f. \n Move the robot a little and put back in auto", current_max_path_error, current_max_angle_error )
                self.paused = True
                self.dev_pause = True
            elif self.dev_pause and not self.tf_pause:
                rospy.logwarn("Line deviation is okay now, y_error is %f angle error is %f", current_max_path_error, current_max_angle_error )
                self.reset_controllers()
                self.paused = False
                self.dev_pause = False

        # Apply low pass filter to the y and ang errors
        if self.use_low_pass_filter:
            if self.lpf_ang.init:
                self.ang_err_filt = self.lpf_ang.update(ang_diff)
            else:
                self.ang_err_filt = self.lpf_ang.filter_init(ang_diff)

            if self.lpf_y.init:
                self.y_err_filt = self.lpf_y.update(y_err)
            else:
                self.y_err_filt = self.lpf_y.filter_init(y_err)
            y_err = self.y_err_filt
            ang_diff =  self.ang_err_filt

#        print dist, y_err, ang_diff
        return dist, y_err, ang_diff



    def _get_vector_to_pose(self):
        if self.follow_poles_from_file:
            if self.row_poles == []:
                self.get_current_row()
                self.initialise_pole_following()
            else:
                self.get_line_from_pole_file()
        elif len(self.row_poles) > 0:
            #  Must have been reconfigued, set original poses
            self.row_poles = []
            self.initial_pose = deepcopy(self.initial_node)
            self.final_pose = deepcopy(self.final_node)


        # Get the closest point to the row
        pose = self._get_line_pose()
        if not pose:
            return None, None, None

        # Get tf from base to closest point on line for y ref
        try:
            tf_local = self._tf_listerner.transformPose(self.base_frame,pose)

            # Get tf from base to the end of row for dist
            tf_goal = self._tf_listerner.transformPose(self.base_frame,self.goal_pose)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None, None, None
        # Since goal_pose contains the correct heading, the tf is the ang error
        _, _, ang_diff = euler_from_quaternion([tf_goal.pose.orientation.x,
                                                tf_goal.pose.orientation.y,
                                                tf_goal.pose.orientation.z,
                                                tf_goal.pose.orientation.w])

        if self.backwards_mode:
            if ang_diff>0:
                ang_diff=-(math.pi-ang_diff)
            else:
                ang_diff=-(-math.pi-ang_diff)

        if self.offset != None:
            y_path_err = tf_local.pose.position.y - self.offset
        else:
            y_path_err = tf_local.pose.position.y

        self.y_err_path = y_path_err
        self.ang_err_path = ang_diff

        return tf_goal.pose.position.x, y_path_err, ang_diff


    def get_angle(self):
        # Get tf from base to the end of row for dist
        tf_goal = self._tf_listerner.transformPose(self.base_frame,self.goal_pose)

        # Since goal_pose contains the correct heading, the tf is the ang error
        _, _, ang_diff = euler_from_quaternion([tf_goal.pose.orientation.x,
                                                tf_goal.pose.orientation.y,
                                                tf_goal.pose.orientation.z,
                                                tf_goal.pose.orientation.w])
        return ang_diff


    def _get_line_pose(self):
        """ Returns a PoseStamped of the pose on a line nearest the robot """
        now = rospy.Time(0)
        try:
            self._tf_listerner.waitForTransform('map', self.base_frame, now, rospy.Duration(self.tf_buffer_size))
            (trans,rot) = self._tf_listerner.lookupTransform('map',self.base_frame,now)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            self.paused = True
            self.tf_pause = True
            rospy.logerr("TF exception - Could not get map -> base_link tf for %f seconds!", self.tf_buffer_size)
            return None

        self.tf_pause = False
        if not self.dev_pause:
            self.reset_controllers()
            self.paused = False

        dist, nearest = pnt2line([trans[0], trans[1], 0], [self.initial_pose.position.x, self.initial_pose.position.y, 0], [self.final_pose.position.x, self.final_pose.position.y, 0])

        if dist >= self.maximum_dev_dist:
            rospy.logerr("Robot has drift too far from line \n Row traversal has been cancelled! \n Finish the row manually")
            if not self.cancelled:
                self.cancelled=True
                self.active=False
                self._as.set_aborted(self._result)

        nearest_pose = PoseStamped(pose=Pose(position=Point(x=nearest[0],
                                                            y=nearest[1])),
                                   header=Header(frame_id='map'))

        return nearest_pose


    def activate_row_detector(self, onoff):
        rospy.wait_for_service('/row_detector/activate_detection')
        try:
            activate_service = rospy.ServiceProxy('/row_detector/activate_detection', SetBool)
            resp1 = activate_service(onoff)
            print(resp1.message)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


    def check_for_continued_row_traversal(self):
        continue_row_trav = False

        if self.latest_row_trav_time:
            print("Time since last row trav: ", rospy.get_time() - self.latest_row_trav_time.to_sec())
            if ( (rospy.get_time() - self.latest_row_trav_time.to_sec()) < 0.5 ):
                print("continue row traversal")
                continue_row_trav = True

        return continue_row_trav



    def get_start_and_end_poses(self, goal):
        """ Get the start and end positions of the current edge for pnt2line function """
        goal_node = self.get_node_from_pose(goal.target_pose.pose)

        final_pose=goal.target_pose.pose

        current_edge, min_dist_to_edge = self.get_current_edge(goal_node)

        print("Current edge: ", current_edge)
        print(min_dist_to_edge)

        if min_dist_to_edge == None:
            min_dist_to_edge = 10000.0

        if min_dist_to_edge > 0.7:
            rospy.logerr("We are too far from a row traversal edge (%f)" %min_dist_to_edge)
            return False

        initial_node = current_edge[0]
        initial_pose = self.action_nodes[initial_node]

        # TODO: Should be able to reduce how many copies we need
        # Extra one was for setting a new node position if we offset the row
        self.initial_pose = deepcopy(initial_pose)
        self.final_pose = deepcopy(final_pose)
        self.initial_node = deepcopy(initial_pose)
        self.final_node = deepcopy(final_pose)
        self.initial_node_original = deepcopy(initial_pose)
        self.final_node_original = deepcopy(final_pose)

        # Below is added to stop refilling in self.get_vector_to_pose()
        radius, the_quat, ang = self._distance_between_poses(self.initial_node, self.final_node)
        self.goal_pose = PoseStamped(pose=Pose(position=Point(x=self.final_node.position.x,
                                                              y=self.final_node.position.y),
                                               orientation=the_quat),
                                     header=Header(frame_id='map'))
        return True


    def initialise_pole_following(self):
        self.poles_pub = rospy.Publisher("/row_traversal/row_poles", ObstacleArray, queue_size=10)
        self.behind_ahead_poles_pub = rospy.Publisher("/row_traversal/behind_ahead_poles", TwoInts, queue_size=10)


    def executeCallback(self, goal):
        rospy.loginfo("Activating row taversal action")
        self.execute_callback_ts = rospy.Time.now()

        continue_row_trav = self.check_for_continued_row_traversal()

        # self.backwards_mode=False
        self.paused = False
        self.active=True
        self.offset = None
        self.row_poles = []
        print("GETTING GOAL NODE:")

        on_row = self.get_start_and_end_poses(goal)
        if on_row:
            self.cancelled = False

            self.get_current_row()

            if self.follow_poles_from_file:
                self.initialise_pole_following()

            if self.offset_row_detector and self.offset == None:
                self.offset = self.get_row_offset()
                self.apply_offset_to_nodes()

            success = self.follow_path(continue_row_trav)

            if success:
                rospy.loginfo('%s: Succeeded' % self._action_name)
                self._as.set_succeeded(self._result)
            else:
                self._as.set_preempted()
        else:
            self.cancelled = True
            self._as.set_aborted(self._result)

        self.row_traversal_status = "OFF"

        if self.use_row_detector:
            self.activate_row_detector(False)



    def giveup(self, timer):
        if self.collision:
            rospy.loginfo("Row Traversal Cancelled")
            self.not_pub.publish("Row Traversal timedout after collision")
            self.cancelled = True
            self._as.set_aborted(self._result)
            # self.backwards_mode=False
        self.giveup_timer_active=False



    def nottim(self, timer):
        if self.collision and self.active:
            now = datetime.now()
            s2 = now.strftime("%Y/%m/%d-%H:%M:%S")
            colstr = "HELP!: too close to obstacle near "+ str(self.closest_node) +" time: "+s2
            self.not_pub.publish(colstr)
        self.notification_timer_active=False
        self.notified=True



    def preemptCallback(self):
        rospy.loginfo('%s: Preempted' % self._action_name)
        self.cancelled = True
        # self.backwards_mode=False
        self.active = False

    #########################################################################################################    
    #### CHANGES NEEDED FOR HUMAN AWARE NAVIGATION ##########################################################
    #########################################################################################################
    def robot_callback(self,robot_info):
        self.robot_action=robot_info.action
        
    def safety_callback(self,safety_info):
        if safety_info.safety_action!=self.hri_safety_action and safety_info.safety_action!=5: #if safety action is required
            self.hri_safety_action=safety_info.safety_action 
        elif safety_info.safety_action==5: #if no safety action is required
            self.hri_safety_action=self.robot_action          
        self.hri_dist=safety_info.critical_dist   
    ########################################################################################################
    ########################################################################################################
def load_data_from_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)



if __name__ == '__main__':
    rospy.init_node('row_traversal')

    pole_positions_path = rospy.get_param("~pole_positions_path", "")

    pole_pos_per_row = []
    if not pole_positions_path.endswith('.yaml'):
        rospy.logwarn("Cant offset row detector - no pole file")
    else:
        pp = load_data_from_yaml(pole_positions_path)
        pole_positions = []
        pole_pos_per_row = []
        for tunnel in pp:
            for row in tunnel[list(tunnel.keys())[0]]:
                coordinates = row[list(row.keys())[0]]["coordinates"]
                orientation = row[list(row.keys())[0]]["orientation"]

                coordinates = [[pos["x"], pos["y"]] for pos in coordinates]

                pole_positions.extend(coordinates)
                pole_pos_per_row.append([np.array(coordinates), orientation])

    server = inRowTravServer(rospy.get_name(),pole_pos_per_row)
