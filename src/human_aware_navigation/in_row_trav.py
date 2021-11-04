#!/usr/bin/env python

import rospy
import math
import PyKDL

import numpy as np
import matplotlib.path as mplPath

import tf

import actionlib
import polytunnel_navigation_actions.msg
import std_msgs.msg

from datetime import datetime
from std_srvs.srv import SetBool

from dynamic_reconfigure.server import Server
from polytunnel_navigation_actions.cfg import RowTraversalConfig

from polytunnel_navigation_actions.point2line import pnt2line

from std_msgs.msg import Header
from std_msgs.msg import String
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist

#import strands_navigation_msgs.srv
from strands_navigation_msgs.msg import TopologicalMap


from sensor_msgs.msg import LaserScan
from polytunnel_navigation_actions.msg import ObstacleArray
from polytunnel_navigation_actions.msg import RowTraversalHealth

from visualization_msgs.msg import Marker
#from visualization_msgs.msg import MarkerArray

######################################################################
from mesapro.msg import human_msg, hri_msg, robot_msg
#######################################################################
class inRowTravServer(object):

    _feedback = polytunnel_navigation_actions.msg.inrownavFeedback()
    _result   = polytunnel_navigation_actions.msg.inrownavResult()
    _controller_freq=1/20.0

    def __init__(self, name):
        self.collision=False
        self._user_controlled=False
        self.goal_overshot=False
        self.prealign_y_axis=True

        self.giveup_timer_active=False
        self.notification_timer_active=False
        self.notified=False


        # Dynamically reconfigurable parameters
        self.kp_ang_ro= 0.6                     # Proportional gain for initial orientation target
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
        self.kp_ang= 0.2                        # Proportional gain for heading correction
        self.kp_y= 0.1                          # Proportional gain for sideways corrections
        self.granularity= 0.5                   # Distance between minigoals along path (carrot points)
        self.y_row_detection_bias = 0.7         # Weight given to the reference given by row detection
        self.y_path_following_bias = 0.3        # Weight given to the original path following
        self.ang_row_detection_bias = 0.2       # Weight given to the angular reference given by row detection
        self.ang_path_following_bias = 0.8      # Weight given to the angular refernce given by path following
        self.minimum_turning_speed = 0.01       # Minimum turning speed
        self.emergency_clearance_x = 0.22       # Clearance from corner frames to trigger emergency stop in x
        self.emergency_clearance_y = 0.22       # Clearance from corner frames to trigger emergency stop in y
        self.goal_tolerance_radius = 0.1        # Goal tolerance Radius in metres
        self.forward_speed= 0.8                 # Forward moving speed
        self.quit_on_timeout=False              # SHould the robot cancel when it meets an obstacle?
        self.time_to_quit=10.0                  # Time until the action is cancelled since collision detected
        self.simultaneous_alignment=False       # Wether to align heading and Y axis simultaneusly or not
        self.y_speed_limit=0.3                  # Maximum speed on the Y axis
        self.turning_speed_limit=0.1            # Maximum turning speed
        self.use_row_entry_function = False     # Enter the row using row_entry function or just go forwards
        self.row_entry_distance = 1.5           # Distance in meters to use row entry function
        self.row_entry_min_speed = 0.15         # Minimum row entry speed
        self.row_entry_kp = 0.3                 # Row entry forward speed gain based on distance travelled
        self.tf_buffer_size = 1.0               # Size of the tf buffer in seconds
        ############################################################################################
        self.human_position_x=[0]               # vector with the human positions X (local frame)
        self.human_position_y=[0]               # vector with the human positions Y (local frame)
        self.human_distance=[0]                 # distance from the robot to each human detected
        self.human_sensor=0                     # which sensor was used to detect the human
        self.human_sensor_t0=[0]                # time counter for the latest data from lidar
        self.human_sensor_t1=[0]                # time counter for the latest data from camera
        self.hri_critical_index=0               # index of the most critical human detected
        self.hri_status= 0                      # human aware navigation flag
        self.hri_safety_action=0                # safety action from the safety system
        self.hri_human_command=0                # human command activated by gesture recognition
        self.han_start_dist=3.6                 # Distance to human at which the robot starts to slow down
        self.han_final_dist=0                   # Distance to human at which the robot must stop
        ###########################################################################################################3
        # This dictionary defines which function should be called when a variable changes via dynamic reconfigure
        self._reconf_functions={'variables':['emergency_clearance_x', 'emergency_clearance_y', 'tf_buffer_size'],
                                'functions':[self.define_safety_zone, self.define_safety_zone, self.reconf_tf_listener]}


        self.object_detected = False
        self.curr_distance_to_object=-1.0
        self.laser_emergency_regions=[]
        self.redefine_laser_regions=False
        self.limits=[]
        self.emergency_base_points=[]           # Corners of Emergency Areas
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
        self.backwards_mode=False
        self.safety_marker=None
        self.active=False
        self.row_traversal_status = "OFF"
        self.y_err_path = 0.0
        self.ang_err_path = 0.0
        self.latest_row_trav_time=None
        self.final_pose=None
        self.initial_pose=None

        self.enable_laser_safety_zone = rospy.get_param("row_traversal/enable_laser_safety_zone", True)
        self.stop_on_overshoot = rospy.get_param("row_traversal/stop_on_overshoot", False)

        # Wether to use or not row detector, basically making it a gps based only action, useful for non polytunnel based actions
        self.use_row_detector = rospy.get_param("row_traversal/use_row_detector",True)
        
        self.base_frame = rospy.get_param("row_traversal/base_frame", "base_link")
        self.corner_frames = rospy.get_param("row_traversal/corner_frames", ["top0", "top1", "top2", "top3"])
        self.wheel_steer_frames = rospy.get_param("row_traversal/wheel_steer_frames", ["leg0", "leg1", "leg2", "leg3"])
 

        while not self.lnodes and not self.cancelled:
            rospy.loginfo("Waiting for topological map")
            rospy.Subscriber('/topological_map', TopologicalMap, self.topological_map_cb)
            if not self.lnodes:
                rospy.sleep(1.0)

        rospy.Subscriber('/scan', LaserScan, self.laser_cb, queue_size=1)
        rospy.Subscriber('/robot_pose', Pose, self.robot_pose_cb)
        rospy.Subscriber('/closest_node', std_msgs.msg.String, self.closest_node_cb)
        rospy.Subscriber('/row_detector/path_error',Pose2D, self.row_correction_cb)
        rospy.Subscriber('/row_detector/obstacles', ObstacleArray, self.obstacles_callback,  queue_size=1)
        rospy.Subscriber('/teleop_joy/joy_priority', Bool, self.joy_lock_cb)
        ##########################################################################################
        rospy.Subscriber('human_info',human_msg,self.human_callback)  
        rospy.Subscriber('human_safety_info',hri_msg,self.safety_callback)  
        #############################################################################################
        self._tf_listerner = tf.TransformListener(cache_time=rospy.Duration(self.tf_buffer_size))
        #self._activate_srv = rospy.ServiceProxy('/row_detector/activate_detection', SetBool)

        self.ppub = rospy.Publisher('/row_traversal/row_line', Path, queue_size=1)
        self.cmd_pub = rospy.Publisher('/nav_vel', Twist, queue_size=1)
        self.ref_pub = rospy.Publisher('/row_traversal/goal_reference', PoseStamped, queue_size=1)
        self.safety_zone_vis_pub = rospy.Publisher('/row_traversal/safety_zone', Marker, queue_size=1)
        self.not_pub = rospy.Publisher('/row_traversal/notification', String, queue_size=1)
        self.health_pub = rospy.Publisher('/row_traversal/health', RowTraversalHealth, queue_size=1)

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

        rospy.spin()


    def dyn_reconf_callback(self, config, level):
        """ 
           Dynamic reconfigure of parameters
           
           This function changes the internal variables modified through the dynamic reconfigure interface
        """
        rospy.loginfo("reconfigure request")
        print config, level

        if self.config:
            changed_dict = {x: self.config[x] != config[x] for x in self.config if x in config}
            lk = [key  for (key, value) in changed_dict.items() if value]
            if lk:
                print "config changed ", lk[0], config[lk[0]]
                if hasattr(self, lk[0]):
                    setattr(self, lk[0], config[lk[0]])
                    print lk[0], getattr(self, lk[0])
                    if lk[0] in self._reconf_functions['variables']:
                        self._reconf_functions['functions'][self._reconf_functions['variables'].index(lk[0])]()
            else:
                print "nothing changed!"
                print self.force_forwards_facing, self.force_backwards_facing

            #self.set_config(lk[0], config[lk[0]])
            self.config = config
        else:
            print "First config: "#, config.items()
            self.config = config
            for i in config.items():
                if hasattr(self, i[0]):
                    setattr(self, i[0], i[1])
                    print i[0], getattr(self, i[0])

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
        print "Defining Safety Zone"
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
        print "visualise safety zone"
        self.safety_zones_visualisation()



    def publish_health_timed_caller(self, timer):
        self.publish_health()


    def publish_health(self):
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
        if msg.data:
            self._user_controlled=True
            if self.goal_overshot:
                self.goal_overshot=False
        else:
            self._user_controlled=False
    

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
            print i

        for i in self.emergency_base_points:
            r=(i['point'].point.x, i['point'].point.y)
            self.emergency_poly.append(r)

        self.emergency_poly=np.asarray(self.emergency_poly)
        self.redefine_laser_regions=False

        self.max_emergency_dist= 0.0
        for i in self.laser_emergency_regions:
            print i['dist']
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
            self.y_ref=msg.y

        if np.isnan((msg.theta)):
            self.ang_ref=None
        else:
            self.ang_ref=msg.theta



    def topological_map_cb(self, msg):
        self.lnodes=msg


    def robot_pose_cb(self, msg):
        self.robot_pose = msg


    def closest_node_cb(self, msg):
        if self.closest_node_cb != msg.data:
            self.closest_node=msg.data
            print self.closest_node
            #for some stupid reason I need to republish this here
            if self.safety_marker:
                self.safety_zone_vis_pub.publish(self.safety_marker)



    def get_node_position(self, node):
        pose=None
        for i in self.lnodes.nodes:
            if i.name == node:
                pose = i.pose
                break
        return pose


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
        the_pose.pose.orientation.w = 1.0#tf.transformations.quaternion_from_euler(0.0,0.0,pose_2d.pose.theta)
        #map_pose = self.listener.transformPose('/map', the_pose)
        return the_pose


    def _get_angle_between_quats(self, ori1, ori2):
        q1 = PyKDL.Rotation.Quaternion(ori1.x, ori1.y, ori1.z, ori1.w)
        q2 = PyKDL.Rotation.Quaternion(ori2.x, ori2.y, ori2.z, ori2.w)

        ang1 = q1.GetRPY()
        ang2 = q2.GetRPY()

        return ang2[2]-ang1[2]



    def check_direction(self):
        dist, y_err, ang_diff = self._get_vector_to_pose()
        print ang_diff, math.degrees(ang_diff)

        if self.force_forwards_facing or self.force_backwards_facing:
            if self.force_forwards_facing:
                self.backwards_mode=False
                print("we MUST be going forwards")
            elif  self.force_backwards_facing:
                self.backwards_mode=True
                print("we MUST be going backwards")
        else:
            if ang_diff > (math.pi/2.0) or ang_diff< -(math.pi/2.0):
                self.backwards_mode=True
                print("we should be going backwards")
            else:
                self.backwards_mode=False
                print("forwards heading")


    def align_orientation(self):
        corrected = False
        self.row_traversal_status = "ALIGN"
        # This if allows us to switch between types of alignment of pre-alignments methods
        if self.simultaneous_alignment:
            # Newer Method
            if self.use_row_detector:
                self.activate_row_detector(True)
                
            print "Aligning based on path first"
            dist, y_err, ang_diff = self._get_vector_to_pose()
            if np.abs(ang_diff) >= self.initial_heading_tolerance or np.abs(y_err) >= self.initial_alignment_tolerance:
                corrected = True
                print "aligning based on path Y dev", y_err, " INITIAL ANg DIFF: ", np.abs(ang_diff)
                while np.abs(ang_diff) >= self.initial_heading_tolerance or np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                    self._send_velocity_commands(0.0, (self.kp_y/2.0)*y_err, self.kp_ang_ro*ang_diff, consider_minimum_rot_vel=True)
                    rospy.sleep(0.05)
                    dist, y_err, ang_diff = self._get_vector_to_pose()
                    self.row_traversal_status = "ALIGN"
            print "re-aligning based on detection"  
            dist, y_err, ang_diff = self._get_references()
            if np.abs(ang_diff) >= self.initial_heading_tolerance or np.abs(y_err) >= self.initial_alignment_tolerance:
                corrected = True
                print "aligning based on path Y dev", y_err, " INITIAL ANg DIFF: ", np.abs(ang_diff)
                while np.abs(ang_diff) >= self.initial_heading_tolerance or np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                    self._send_velocity_commands(0.0, (self.kp_y/2.0)*y_err, self.kp_ang_ro*ang_diff, consider_minimum_rot_vel=True)
                    rospy.sleep(0.05)
                    dist, y_err, ang_diff = self._get_references()
                    self.row_traversal_status = "ALIGN"      
        else:
            # Original Method
            dist, y_err, ang_diff = self._get_vector_to_pose()
            if self.prealign_y_axis and np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
                corrected = True
                print "Aligning ang based on path"
                while np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
                    self._send_velocity_commands(0.0, 0.0, self.kp_ang_ro*ang_diff, consider_minimum_rot_vel=True)
                    rospy.sleep(0.05)
                    dist, y_err, ang_diff = self._get_vector_to_pose()
                    # print "Aligning ang path", ang_diff
                    self.row_traversal_status = "ALIGN"

            
            dist, y_err, ang_diff = self._get_vector_to_pose()
            if self.prealign_y_axis and np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                print "Aligning y based on path"
                corrected = True
                self.check_wheel_direction(direction='y')
                while np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                    self._send_velocity_commands(0.0, (self.kp_y/2.0)*y_err, 0.0, consider_minimum_rot_vel=True)
                    rospy.sleep(0.05)
                    dist, y_err, ang_diff = self._get_vector_to_pose()
                    # print "Aligning Y path", y_err
                    self.row_traversal_status = "ALIGN"


            if self.use_row_detector:
                self.activate_row_detector(True)

            dist, y_err, ang_diff = self._get_references()
            if self.prealign_y_axis and np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
                print "re-aligning ang based on detection"
                corrected = True
                while np.abs(ang_diff) >= self.initial_heading_tolerance and not self.cancelled:
                    self._send_velocity_commands(0.0, 0.0, self.kp_ang_ro*ang_diff, consider_minimum_rot_vel=True)
                    rospy.sleep(0.05)
                    dist, y_err, ang_diff = self._get_references()
                    # print "Aligning ang detection", ang_diff
                    self.row_traversal_status = "ALIGN"


            dist, y_err, ang_diff = self._get_references()
            if self.prealign_y_axis and np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                print "re-aligning y based on detection"
                corrected = True
                self.check_wheel_direction(direction='y')
                while np.abs(y_err) >= self.initial_alignment_tolerance and not self.cancelled:
                    self._send_velocity_commands(0.0, (self.kp_y/2.0)*y_err, 0.0, consider_minimum_rot_vel=True)
                    rospy.sleep(0.05)
                    dist, y_err, ang_diff = self._get_references()
                    # print "Aligning path detection", y_err
                    self.row_traversal_status = "ALIGN"




    def enter_row(self):
        print "Entering row"
        self.row_traversal_status = "FOLLOW"
        overall_goal_distance = np.abs(math.hypot(self.final_pose.position.x-self.robot_pose.position.x, self.final_pose.position.y-self.robot_pose.position.y))
        # print 'overall goal distance =', overall_goal_distance
        
        dist, y_err, ang_diff = self._get_references(publish_health=True)
        pre_gdist=dist
        
        if not self.cancelled:
            self.check_wheel_direction(direction='x')

        distance_travelled = 0
        remaining_goal_distance = overall_goal_distance
        while distance_travelled < self.row_entry_distance and not self.cancelled and np.abs(dist) > self.goal_tolerance_radius and not self.goal_overshot:
            self.latest_row_trav_time = rospy.Time.now()
            if self.cancelled:
                print "leaveing row entry as self.cancelled = ", self.cancelled
                break 
            dist, y_err, ang_diff = self._get_references(publish_health=True)
            speed=self.get_row_entry_speed(distance_travelled)
            # print "Enter row, xvel ", format(speed, '.3f'), " dist done ", format(distance_travelled, '.3f'), " dist_tot: ", format(remaining_goal_distance, '.3f')
            # print 'Cancelled? : ', self.cancelled, ' Collision? : ', self.collision
            self._send_velocity_commands(speed, self.kp_y*y_err, self.kp_ang*ang_diff)

            rospy.sleep(self._controller_freq)
            self.row_traversal_status = "FOLLOW"
            remaining_goal_distance = np.abs(math.hypot(self.final_pose.position.x-self.robot_pose.position.x, self.final_pose.position.y-self.robot_pose.position.y))
            distance_travelled = np.abs(overall_goal_distance - remaining_goal_distance)

            # check for overshoot
            if not self._user_controlled:
                if np.sign(dist) != np.sign(pre_gdist):
                    self.goal_overshot= True
                    nottext="Row traversal has overshoot, previous distance "+str(np.abs(pre_gdist))+" current distance "+str(np.abs(dist))
                    print nottext
                    self.not_pub.publish(nottext)
                    rospy.logwarn(nottext)


        if remaining_goal_distance > overall_goal_distance:
            print "WE HAVE GONE THE WRONG WAY!!!!"
            self.cancelled = True
   
    def get_row_entry_speed(self, distance_travelled):
        ##########################################################################################
        if self.hri_safety_action==0: #if hri is safety, i.e normal operation
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

        else: #if a safety action is required
            if self.hri_safety_action==1: #if the speed is reduced while approaching
                index=self.hri_critical_index
                if self.human_sensor[index]==1: #if sensor was lidar 
                    dist=sqrt(self.human_position_x[index]**2+self.human_position_y[index]**2)
                elif self.human_sensor[index]==1: #if sensor was camera
                    dist=self.human_distance[index]  
                else: #if sensor was camera+lidar
                    if self.human_sensor_t0[index]>=self.human_sensor_t1[index]: #if data from lidar is newer than from camera
                        dist=sqrt(self.human_position_x[index]**2+self.human_position_y[index]**2)
                    else: #if data from camera is newer than from lidar
                        dist=self.human_distance[index]     
                if dist <= self.han_start_dist:
                    slowdown_delta = self.han_start_dist - self.han_final_dist
                    current_percent = (dist - self.han_final_dist) / slowdown_delta
                    if current_percent >0:
                        print("Limiting speed")
                        speed = min((current_percent*self.forward_speed), (self.row_entry_min_speed + max(0.0, (self.row_entry_kp*distance_travelled))))
                    else:
                        print("stop")
                        speed = 0.0
                else:
                    speed = self.row_entry_min_speed + max(0.0, (self.row_entry_kp*distance_travelled))
            elif self.hri_safety_action>=2: #if a safety stop is required or the robot needs a human command
                print("stop")
                speed = 0.0
        ##############################################################################
        
        if self.backwards_mode:
            speed = -speed

        return speed
    

    def check_wheel_direction(self, direction='x'):
        # Check that the wheels are all facing the right way before we proceed
        print('Checking wheel direction')
        wheels_aligned = False
        while not wheels_aligned:
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
            
            _, _, wheel_yaw0 =  tf.transformations.euler_from_quaternion(wheel_rot0)
            _, _, wheel_yaw1 =  tf.transformations.euler_from_quaternion(wheel_rot1)
            _, _, wheel_yaw2 =  tf.transformations.euler_from_quaternion(wheel_rot2)
            _, _, wheel_yaw3 =  tf.transformations.euler_from_quaternion(wheel_rot3)

            # print(wheel_yaw0,wheel_yaw1,wheel_yaw2,wheel_yaw3)
        
            if direction is 'x':
                if abs(wheel_yaw0) < 0.08 and abs(wheel_yaw1) < 0.08 and abs(wheel_yaw2) < 0.08 and abs(wheel_yaw3) < 0.08:
                    wheels_aligned = True
                else:
                    self._send_velocity_commands(0,0,0)
            
            if direction is 'y':
                if abs(abs(wheel_yaw0)-math.pi/2) < 0.08 and abs(abs(wheel_yaw1)-math.pi/2) < 0.08 and abs(abs(wheel_yaw2)-math.pi/2) < 0.08 and abs(abs(wheel_yaw3)-math.pi/2) < 0.08:
                    wheels_aligned = True
                else:
                    self._send_velocity_commands(0,0.000001,0)
                        
            rospy.sleep(self._controller_freq)
        
        print('Wheel check passed')

                

    def get_forward_speed(self):
        ##########################################################################################
        if self.hri_safety_action==0: #if hri is safety, i.e normal operation
            if not self.constant_forward_speed:
                if not self.object_detected and self.curr_distance_to_object <= self.approach_dist_to_obj:
                    #print "not limiting"
                    if self.backwards_mode:
                        speed = -self.forward_speed
                    else:
                        speed = self.forward_speed
                else:
                    #print "limiting"
                    slowdown_delta=self.approach_dist_to_obj-self.min_dist_to_obj
                    current_percent = (self.curr_distance_to_object-self.min_dist_to_obj)/slowdown_delta
                    if current_percent >0:
                        speed = current_percent*self.forward_speed
                    else:
                        speed = 0.0
                    if self.backwards_mode:
                        speed = -speed
            else:
                #print "not limiting"
                if self.backwards_mode:
                    speed = -self.forward_speed
                else:
                    speed = self.forward_speed
        else: #if a safety action is required
            if self.hri_safety_action==1: #if the speed is reduced while approaching
                index=self.hri_critical_index
                if self.human_sensor[index]==1: #if sensor was lidar 
                    dist=sqrt(self.human_position_x[index]**2+self.human_position_y[index]**2)
                elif self.human_sensor[index]==1: #if sensor was camera
                    dist=self.human_distance[index]  
                else: #if sensor was camera+lidar
                    if self.human_sensor_t0[index]>=self.human_sensor_t1[index]: #if data from lidar is newer than from camera
                        dist=sqrt(self.human_position_x[index]**2+self.human_position_y[index]**2)
                    else: #if data from camera is newer than from lidar
                        dist=self.human_distance[index]     
                if dist <= self.han_start_dist:
                    slowdown_delta = self.han_start_dist - self.han_final_dist
                    current_percent = (dist - self.han_final_dist) / slowdown_delta
                    if current_percent >0:
                        print("Limiting speed")
                        speed = min((current_percent*self.forward_speed), (self.row_entry_min_speed + max(0.0, (self.row_entry_kp*distance_travelled))))
                    else:
                        print("stop")
                        speed = 0.0
                    if self.backwards_mode:
                            speed = -speed
                else:
                    if self.backwards_mode:
                        speed = -self.forward_speed
                    else:
                        speed = self.forward_speed
            elif self.hri_safety_action>=2: #if a safety stop is required or the robot needs a human command
                print("stop")
                speed = 0.0
        ##############################################################################
        return speed



    def go_forwards(self):#, start_goal):
        self.row_traversal_status = "FOLLOW"
        if self.backwards_mode:
            speed = -self.forward_speed
        else:
            speed = self.forward_speed
        
        if not self.cancelled:
            print "Going forwards"
            self.check_wheel_direction(direction='x')

        
        dist, y_err, ang_diff = self._get_references()
#        goal_overshot=False
        gdist, gy_err, gang_diff = self._get_references()
        self.goal_overshot=False
        pre_gdist=gdist     #Hack to stop the robot if it overshot

        dist, y_err, ang_diff = self._get_references(publish_health=True)
        #print "1-> ", dist, " ", self.cancelled
        #self.goal_overshot=False
        while np.abs(dist)>self.goal_tolerance_radius and not self.cancelled and not self.goal_overshot:
            self.latest_row_trav_time = rospy.Time.now()
            if self.cancelled:
                break
            if not self.goal_overshot:
                speed=self.get_forward_speed()
                self._send_velocity_commands(speed, self.kp_y*y_err, self.kp_ang*ang_diff)
            else:
                self._send_velocity_commands(0.0, 0.0, 0.0)
                break
            rospy.sleep(self._controller_freq)

            dist, y_err, ang_diff = self._get_references(publish_health=True)

            if not dist:
                dist=pre_gdist
            # Hack to stop the robot if it overshot:
            # To see if the robot has overshot we check if the distance to goal has actually increased 
            # or has changed much faster than expected (massive misslocalisation) 4 times the forward speed 
            # times the control period (0.05 seconds) 
            # and that is not being controlled (helped) by the user.
            pre_gdist=dist     
            progress_to_goal=np.abs(pre_gdist)-np.abs(gdist)
            #print progress_to_goal, gdist, pre_gdist, self._user_controlled, self.stop_on_overshoot
            if not self._user_controlled:
                if np.sign(dist) != np.sign(gdist):#progress_to_goal < -0.1:# or np.abs(progress_to_goal)>=((4*self._controller_freq)*self.forward_speed):
                    self.goal_overshot= True
                    nottext="Row traversal has overshoot, previous distance "+str(np.abs(pre_gdist))+" current distance "+str(np.abs(gdist))
                    print nottext
                    print progress_to_goal, gdist, pre_gdist
                    self.not_pub.publish(nottext)
                    rospy.logwarn(nottext)
#                        if not self.stop_on_overshoot:
                    #self.cancelled=True
                    #self._send_velocity_commands(0.0, 0.0, 0.0)
                    #break

            #print "- ", dist, " ", self.cancelled
            self.row_traversal_status = "FOLLOW"


        self._send_velocity_commands(0.0, 0.0, 0.0)
        self.active=False
        self.row_traversal_status = "OFF"
            
        if not self.cancelled or self.goal_overshot:
            return True
        else:
            return False
        


    def follow_path(self, continue_row_trav):

        self.check_direction()

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
        
        if row_trav_complete:
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)
            self.row_change_status = "OFF"
        else:
            self.row_change_status = "OFF"



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
                cmd_vel.angular.z = angvel

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
        dist, y_path_err, ang_path_diff = self._get_vector_to_pose()
        using_row_yref = False
        using_row_angref = False

        if self.y_ref and y_path_err:
#            if self.backwards_mode:
#                self.y_ref = -1.0*self.y_ref
            using_row_detector_yref = True
            y_err = np.average([self.y_ref, y_path_err], weights=[self.y_row_detection_bias, self.y_path_following_bias])
        elif y_path_err:
            y_err = y_path_err
        elif self.y_ref:
            y_err = self.y_ref
        else:
            y_err = None


        if self.ang_ref and ang_path_diff:
            using_row_detector_angref = True
            ang_diff = np.average([self.ang_ref, ang_path_diff], weights=[self.ang_row_detection_bias, self.ang_path_following_bias])
        elif ang_path_diff:
            ang_diff = ang_path_diff
        elif self.ang_ref:
            ang_diff = self.ang_ref
        else:
            ang_diff = None

        current_max_angle_error = max( abs(ang_diff), abs(ang_path_diff), abs(ang_diff-ang_path_diff) )
        current_max_path_error = max( abs(y_err), abs(y_path_err), abs(ang_diff-ang_path_diff) )

        if self.active and not self.cancelled:
            if (current_max_angle_error > self.maximum_dev_dist_theta or current_max_path_error > self.maximum_dev_dist_y):
                rospy.logerr("Line deviation is too high, y_error is %f angle error is %f. \n Move the robot a little and put back in auto", current_max_path_error, current_max_angle_error )
                self.paused = True
                self.dev_pause = True
            elif self.dev_pause and not self.tf_pause:
                rospy.logwarn("Line deviation is okay now, y_error is %f angle error is %f", current_max_path_error, current_max_angle_error )
                self.paused = False
                self.dev_pause = False

            
#        print dist, y_err, ang_diff
        return dist, y_err, ang_diff



    def _get_vector_to_pose(self):
        
        pose=self._get_line_pose()
        if not pose:
            return None, None, None

        transform_local = self._tf_listerner.transformPose(self.base_frame,pose)
         
        dist, the_quat, b = self._distance_between_poses(pose.pose,self.final_pose)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id='map'
        goal_pose.pose.position.x = self.final_pose.position.x
        goal_pose.pose.position.y = self.final_pose.position.y
        goal_pose.pose.orientation = the_quat
        transform_goal = self._tf_listerner.transformPose(self.base_frame,goal_pose)
        #print 'goal dist in x: ',transform_goal.pose.position.x, 'goal dist in y: ', transform_local.pose.position.y
        orientation_list = [transform_local.pose.orientation.x, transform_local.pose.orientation.y, transform_local.pose.orientation.z, transform_local.pose.orientation.w]
        euls = tf.transformations.euler_from_quaternion(orientation_list)
        #print transform.pose.position.x, transform.pose.position.y, math.degrees(euls[2]), euls[2]
        self.ref_pub.publish(transform_local)

        ang_diff = euls[2]

        if self.backwards_mode:
            if ang_diff>0:
                ang_diff=-(math.pi-ang_diff)
            else:
                ang_diff=-(-math.pi-ang_diff)


        self.y_err_path = transform_local.pose.position.y
        self.ang_err_path = ang_diff



        return transform_goal.pose.position.x, transform_local.pose.position.y, ang_diff



    def _get_line_pose(self):
        now=rospy.Time(0)
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
            self.paused = False

        #pose error
        dist, nearest = pnt2line([trans[0], trans[1], 0], [self.initial_pose.position.x, self.initial_pose.position.y, 0], [self.final_pose.position.x, self.final_pose.position.y, 0])
        #print 'dist = ', dist, ' nearest = ', nearest
        if dist >= self.maximum_dev_dist:# and self._inside_row_mode:
            rospy.logerr("Robot has drift too far from line \n Row traversal has been cancelled! \n Finish the row manually")
            if not self.cancelled:
                self.cancelled=True
                self.active=False
                self._as.set_aborted(self._result) 
        
        nearest_pose=PoseStamped()
        nearest_pose.header.frame_id='map'
        nearest_pose.pose.position.x = nearest[0]
        nearest_pose.pose.position.y = nearest[1]
        
        radius, the_quat, ang = self._distance_between_poses(self.initial_pose, self.final_pose)        
        nearest_pose.pose.orientation = the_quat

        
        return nearest_pose
        
            
    def activate_row_detector(self, onoff):
        rospy.wait_for_service('/row_detector/activate_detection')
        try:
            activate_service = rospy.ServiceProxy('/row_detector/activate_detection', SetBool)
            resp1 = activate_service(onoff)
            print resp1.message
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    def check_for_continued_row_traversal(self):
        
        continue_row_trav = False

        if self.latest_row_trav_time:
            print "Time since last row trav: ", rospy.get_time() - self.latest_row_trav_time.to_sec()
            if ( (rospy.get_time() - self.latest_row_trav_time.to_sec()) < 0.5 ):
                print "continue row traversal"
                continue_row_trav = True

        return continue_row_trav



    def get_start_and_end_poses(self, goal):

        initial_pose=self.get_node_position(self.closest_node)
        final_pose=goal.target_pose.pose

        if self.final_pose and self.initial_pose:
            if initial_pose == final_pose:
                initial_pose = self.robot_pose
                print 'closest node is current node'
            elif self.final_pose.position.x == final_pose.position.x and self.final_pose.position.y == final_pose.position.y: # if goal is repeated continue as usual
                initial_pose == self.initial_pose
                print 'repeated goal'


        self.initial_pose=initial_pose
        self.final_pose=final_pose



    def executeCallback(self, goal):
        rospy.loginfo("Activating row taversal action")
        
        health_publisher_timer = rospy.Timer(rospy.Duration(0.2), self.publish_health_timed_caller, oneshot=False)

        continue_row_trav = self.check_for_continued_row_traversal()

        self.backwards_mode=False
        self.cancelled = False
        self.paused = False
        self.active=True

        print "GETTING GOAL NODE:"


        self.get_start_and_end_poses(goal)
        self.follow_path(continue_row_trav)


        if self.use_row_detector:
            self.activate_row_detector(False)

        self.publish_health()
        health_publisher_timer.shutdown()



    def giveup(self, timer):
        if self.collision:
            rospy.loginfo("Row Traversal Cancelled")
            self.not_pub.publish("Row Traversal timedout after collision")
            self.cancelled = True
            self._as.set_aborted(self._result)
            self.backwards_mode=False
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
        self._as.set_preempted()
        self.cancelled = True
        self.backwards_mode=False
        self.active = False
    #######################################################################################
    def human_callback(self,human_info):
        self.human_position_x=human_info.position_x
        self.human_position_y=human_info.position_y
        self.human_distance=human_info.distance
        self.human_sensor=human_info.sensor
        self.human_sensor_t0=human_info.sensor_t0
        self.human_sensor_t1=human_info.sensor_t1
    
    def safety_callback(self,safety_info):
        self.hri_status=safety_info.hri_status
        self.hri_safety_action=safety_info.safety_action
        self.hri_human_command=safety_info.human_command
        self.hri_critical_index=safety_info.critical_index   
    ##############################################################################################    
if __name__ == '__main__':
    rospy.init_node('row_traversal')
    server = inRowTravServer(rospy.get_name())
