#!/usr/bin/env python

import rospy
import actionlib
import yaml, json

import dynamic_reconfigure.client

import topological_navigation_msgs.msg
from topological_navigation_msgs.msg import NavStatistics
from topological_navigation_msgs.msg import CurrentEdge
from topological_navigation_msgs.msg import ClosestEdges
from topological_navigation_msgs.srv import EvaluateEdge, EvaluateEdgeRequest, EvaluateNode, EvaluateNodeRequest

from std_msgs.msg import String
from actionlib_msgs.msg import GoalStatus

from topological_navigation.route_search2 import RouteChecker, TopologicalRouteSearch2, get_route_distance
from topological_navigation.navigation_stats import nav_stats
from topological_navigation.tmap_utils import *

from topological_navigation.edge_action_manager import EdgeActionManager
from topological_navigation.edge_reconfigure_manager import EdgeReconfigureManager

from copy import deepcopy
from threading import Lock

######################################################################
from mesapro.msg import human_msg, hri_msg, robot_msg
from sensor_msgs.msg import Joy
import numpy as np
#######################################################################


# A list of parameters topo nav is allowed to change and their mapping from dwa speak.
# If not listed then the param is not sent, e.g. TrajectoryPlannerROS doesn't have tolerances.
DYNPARAM_MAPPING = {
    "DWAPlannerROS": {
        "yaw_goal_tolerance": "yaw_goal_tolerance",
        "xy_goal_tolerance": "xy_goal_tolerance",
    },
    "TebLocalPlannerROS": {
        "yaw_goal_tolerance": "yaw_goal_tolerance",
        "xy_goal_tolerance": "xy_goal_tolerance",
    },
}
    
status_mapping = {}
status_mapping[0] = "PENDING"
status_mapping[1] = "ACTIVE"
status_mapping[2] = "PREEMPTED"
status_mapping[3] = "SUCCEEDED"
status_mapping[4] = "ABORTED"
status_mapping[5] = "REJECTED"
status_mapping[6] = "PREEMPTING"
status_mapping[7] = "RECALLING"
status_mapping[8] = "RECALLED"
status_mapping[9] = "LOST"
###################################################################################################################


###################################################################################################################
class TopologicalNavServer(object):
    
    _feedback = topological_navigation_msgs.msg.GotoNodeFeedback()
    _result = topological_navigation_msgs.msg.GotoNodeResult()

    _feedback_exec_policy = topological_navigation_msgs.msg.ExecutePolicyModeFeedback()
    _result_exec_policy = topological_navigation_msgs.msg.ExecutePolicyModeResult()

    def __init__(self, name, mode):
        
        rospy.on_shutdown(self._on_node_shutdown)
        
        ############################################################################################
        self.robot_operation=6                  # robot operation used for the safety system
        self.past_operation=100                   # past robot operation
        self.prev_status = None                  # status of the route following
        self.goal= "None"                        # Current goal to reach
        self.goal_move_away="WayPoint139"
        self.goal_approach="WayPoint83"
        self.hri_critical_index=0               # index of the most critical human detected
        self.hri_status= 0                      # human aware navigation flag
        self.hri_safety_action=0                # safety action from the safety system
        self.hri_human_command=0                # human command activated by gesture recognition
        ###########################################################################################
        
        self.node_by_node = False
        self.cancelled = False
        self.preempted = False
        self.stat = None
        self.no_orientation = False
        self._target = "None"
        self.current_action = "none"
        self.next_action = "none"
        self.nav_from_closest_edge = False
        self.fluid_navigation = True

        self.current_node = "Unknown"
        self.closest_node = "Unknown"
        self.closest_edges = ClosestEdges()
        self.nfails = 0
        
        self.navigation_activated = False
        self.navigation_lock = Lock()

        move_base_actions = [
            "move_base",
            "human_aware_navigation",
            "han_adapt_speed",
            "han_vc_corridor",
            "han_vc_junction",
        ]

        self.needed_actions = []
        self.move_base_actions = rospy.get_param("~move_base_actions", move_base_actions)

        # what service are we using as move_base?
        self.move_base_name = rospy.get_param("~move_base_name", "move_base")
        if not self.move_base_name in self.move_base_actions:
            self.move_base_actions.append(self.move_base_name)
        
        self.stats_pub = rospy.Publisher("topological_navigation/Statistics", NavStatistics, queue_size=10)
        self.edge_pub = rospy.Publisher("topological_navigation/Edge", CurrentEdge, queue_size=10)
        self.route_pub = rospy.Publisher("topological_navigation/Route", topological_navigation_msgs.msg.TopologicalRoute, queue_size=10)
        self.cur_edge = rospy.Publisher("current_edge", String, queue_size=10)
        self.move_act_pub = rospy.Publisher("topological_navigation/move_action_status", String, latch=True, queue_size=1)

        self._map_received = False
        rospy.Subscriber("/topological_map_2", String, self.MapCallback)
        rospy.loginfo("Navigation waiting for the Topological Map...")

        while not self._map_received:
            rospy.sleep(rospy.Duration.from_sec(0.05))
        rospy.loginfo("Navigation received the Topological Map")
        
        self.make_move_base_edge()
        self.edge_action_manager = EdgeActionManager()

        # Creating Action Server for navigation
        rospy.loginfo("Creating GO-TO-NODE action server...")
        self._as = actionlib.SimpleActionServer(name, topological_navigation_msgs.msg.GotoNodeAction,
                                                execute_cb=self.executeCallback, auto_start=False)
        self._as.register_preempt_callback(self.preemptCallback)
        self._as.start()
        rospy.loginfo("...done")

        # Creating Action Server for execute policy
        rospy.loginfo("Creating EXECUTE_POLICY_MODE action server...")
        self._as_exec_policy = actionlib.SimpleActionServer("topological_navigation/execute_policy_mode", topological_navigation_msgs.msg.ExecutePolicyModeAction, 
                                                            execute_cb=self.executeCallbackexecpolicy, auto_start=False)
        self._as_exec_policy.register_preempt_callback(self.preemptCallbackexecpolicy)
        self._as_exec_policy.start()
        rospy.loginfo("...done")

        rospy.loginfo("Subscribing to Localisation Topics...")
        rospy.Subscriber("closest_node", String, self.closestNodeCallback)
        rospy.Subscriber("closest_edges", ClosestEdges, self.closestEdgesCallback)
        rospy.Subscriber("current_node", String, self.currentNodeCallback)
        rospy.loginfo("...done")
        ##########################################################################################
        #rospy.Subscriber('human_info',human_msg,self.human_callback)  
        rospy.Subscriber('human_safety_info',hri_msg,self.safety_callback)  
        #rospy.Subscriber('joy',Joy,self.joy_callback)  
        #############################################################################################
        try:
            rospy.wait_for_service('restrictions_manager/evaluate_edge', timeout=3.0)
            
            self.evaluate_edge_srv = rospy.ServiceProxy(
                'restrictions_manager/evaluate_edge', EvaluateEdge)
            self.evaluate_node_srv = rospy.ServiceProxy(
                'restrictions_manager/evaluate_node', EvaluateNode)
            
            self.using_restrictions = True
        except:
            rospy.logwarn("Restrictions Unavailable")
            self.using_restrictions = False

        self.edge_reconfigure = rospy.get_param("~reconfigure_edges", True)
        self.srv_edge_reconfigure = rospy.get_param("~reconfigure_edges_srv", False)
        if self.edge_reconfigure:
            self.edgeReconfigureManager = EdgeReconfigureManager()
        else:
            rospy.logwarn("Edge Reconfigure Unavailable")

        rospy.loginfo("All Done.")
        ###########################################################################################
        #rospy.spin()
        ########################################################################################
        
    def _on_node_shutdown(self):
        self.cancel_current_action(timeout_secs=2)
        
        
    def make_move_base_edge(self):
        
        self.move_base_edge = {}
        self.move_base_edge["action"] = self.move_base_name
        self.move_base_edge["edge_id"] = "move_base_edge"
        self.move_base_edge["action_type"] = "move_base_msgs/MoveBaseGoal"
        
        self.move_base_edge["goal"] = {}
        self.move_base_edge["goal"]["target_pose"] = {}
        self.move_base_edge["goal"]["target_pose"]["pose"] = "$node.pose"
        self.move_base_edge["goal"]["target_pose"]["header"] = {}
        self.move_base_edge["goal"]["target_pose"]["header"]["frame_id"] = "$node.parent_frame"
    

    def init_reconfigure(self):
        
        self.move_base_planner = rospy.get_param("~move_base_planner", "move_base/DWAPlannerROS")
        rospy.loginfo("Creating reconfigure client for {}".format(self.move_base_planner))
        self.rcnfclient = dynamic_reconfigure.client.Client(self.move_base_planner)
        self.init_dynparams = self.rcnfclient.get_configuration()
        

    def reconf_movebase(self, cedg, cnode, intermediate):

        if cnode["node"]["properties"]["xy_goal_tolerance"] <= 0.1:
            cxygtol = 0.1
        else:
            cxygtol = cnode["node"]["properties"]["xy_goal_tolerance"]
        if not intermediate:
            if cnode["node"]["properties"]["yaw_goal_tolerance"] <= 0.087266:
                cytol = 0.087266
            else:
                cytol = cnode["node"]["properties"]["yaw_goal_tolerance"]
        else:
            cytol = 6.283

        params = {"yaw_goal_tolerance": cytol, "xy_goal_tolerance": cxygtol}
        rospy.loginfo("Reconfiguring %s with %s" % (self.move_base_name, params))
        print("Intermediate: {}".format(intermediate))
        self.reconfigure_movebase_params(params)
        

    def reconfigure_movebase_params(self, params):

        self.init_dynparams = self.rcnfclient.get_configuration()
        
        key = self.move_base_planner[self.move_base_planner.rfind("/") + 1 :]
        translation = DYNPARAM_MAPPING[key]
        
        translated_params = {}
        for k, v in params.iteritems():
            if k in translation:
                if rospy.has_param(self.move_base_planner + "/" + translation[k]):
                    translated_params[translation[k]] = v
                else:
                    rospy.logwarn("%s has no parameter %s" % (self.move_base_planner, translation[k]))
            else:
                rospy.logwarn("%s has no dynparam translation for %s" % (self.move_base_planner, k))
                
        self._do_movebase_reconf(translated_params)
        

    def _do_movebase_reconf(self, params):
        
        try:
            self.rcnfclient.update_configuration(params)
        except rospy.ServiceException as exc:
            rospy.logwarn("Could not reconfigure move_base parameters. Caught service exception: %s. Will continue with previous parameters", exc)
            

    def reset_reconf(self):
        self._do_movebase_reconf(self.init_dynparams)


    def MapCallback(self, msg):
        """
         This Function updates the Topological Map everytime it is called
        """
        self.lnodes = yaml.safe_load(msg.data)
        self.topol_map = self.lnodes["pointset"]
        self.curr_tmap = deepcopy(self.lnodes)
        self.rsearch = TopologicalRouteSearch2(self.lnodes)
        self.route_checker = RouteChecker(self.lnodes)

        for node in self.lnodes["nodes"]:
            for edge in node["node"]["edges"]:
                if edge["action"] not in self.needed_actions:
                    self.needed_actions.append(edge["action"])
        
        self._map_received = True


    def executeCallback(self, goal):
        
        """
        This Functions is called when the topo nav Action Server is called
        """
        print("\n####################################################################################################")
        rospy.loginfo("Processing GO-TO-NODE goal (No Orientation = {})".format(goal.no_orientation))
        can_start = False

        with self.navigation_lock:
            if self.cancel_current_action(timeout_secs=10):
                # we successfully stopped the previous action, claim the title to activate navigation
                self.navigation_activated = True
                can_start = True
        
        if can_start:

            self.cancelled = False
            self.preempted = False
            self.no_orientation = goal.no_orientation
            
            self._feedback.route = "Starting..."
            self._as.publish_feedback(self._feedback)
            ###################################################################
            self.robot_operation=1 # operation when the robot is moving to the goal
            self.past_operation=self.robot_operation #to give priority to a new human command goal
            self.goal=goal.target
            ###################################################################
            self.navigate(goal.target)
            ###################################################################
            self.robot_operation=6 # robot operation is "wait for new goal"
            #self.past_operation=self.robot_operation
            ###################################################################
        else:
            rospy.logwarn("Could not cancel current navigation action, GO-TO-NODE goal aborted")
            self._as.set_aborted()

        self.navigation_activated = False


    def executeCallbackexecpolicy(self, goal):
        
        """
        This Function is called when the execute policy Action Server is called
        """
        print("\n####################################################################################################")
        rospy.loginfo("Processing EXECUTE POLICY MODE goal")
        can_start = False

        with self.navigation_lock:
            if self.cancel_current_action(timeout_secs=10):
                # we successfully stopped the previous action, claim the title to activate navigation
                self.navigation_activated = True
                can_start = True
       
        if can_start:

            self.cancelled = False
            self.preempted = False
            self.nav_from_closest_edge = False
            
            route = goal.route
            valid_route = self.route_checker.check_route(route)
            
            if valid_route:
                final_edge = get_edge_from_id_tmap2(self.lnodes, route.source[-1], route.edge_id[-1])
                target = final_edge["node"]
                result = self.execute_policy(route, target)
            else:
                result = False
                self.cancelled = True
                rospy.logerr("Invalid route in execute policy mode goal")

            if not self.cancelled and not self.preempted:
                self._result_exec_policy.success = result
                if result:
                    self._as_exec_policy.set_succeeded(self._result_exec_policy)
                else:
                    self._as_exec_policy.set_aborted(self._result_exec_policy)
            else:
                if not self.preempted:
                    self._result_exec_policy.success = result
                    self._as_exec_policy.set_aborted(self._result_exec_policy)
                else:
                    self._result_exec_policy.success = False
                    self._as_exec_policy.set_preempted(self._result_exec_policy)

        else: 
            rospy.logwarn("Could not cancel current navigation action, EXECUTE POLICY MODE goal aborted.")
            self._as_exec_policy.set_aborted()

        self.navigation_activated = False


    def preemptCallback(self):
        
        rospy.logwarn("Preempting GO-TO-NODE goal")
        self.preempted = True
        self.cancel_current_action(timeout_secs=2)


    def preemptCallbackexecpolicy(self):
        
        rospy.logwarn("Preempting EXECUTE POLICY MODE goal")
        self.preempted = True
        self.cancel_current_action(timeout_secs=2)


    def closestNodeCallback(self, msg):
        self.closest_node = msg.data


    def closestEdgesCallback(self, msg):
        self.closest_edges = msg


    def currentNodeCallback(self, msg):
        
        if self.current_node != msg.data:  # is there any change on this topic?
            self.current_node = msg.data  # we are at this new node
            if msg.data != "none":  # are we at a node?
                rospy.loginfo("New node reached: {}".format(self.current_node))
                if self.navigation_activated:  # is the action server active?
                    if self.stat:
                        self.stat.set_at_node()
                    # if the robot reached and intermediate node and the next action is move base goal has been reached
                    if (
                        self.current_node == self.current_target
                        and self._target != self.current_target
                        and self.next_action in self.move_base_actions
                        and self.current_action in self.move_base_actions
                        and self.fluid_navigation
                    ):
                        rospy.loginfo("Intermediate node reached: %s", self.current_node)
                        self.goal_reached = True


    def navigate(self, target):
        """
        This function takes the target node and plans the actions that are required
        to reach it.
        """
        result = False

        if not self.cancelled:

            g_node = self.rsearch.get_node_from_tmap2(target)
            self.max_dist_to_closest_edge = rospy.get_param("~max_dist_to_closest_edge", 1.0)
            
            if self.closest_edges.distances[0] > self.max_dist_to_closest_edge or self.current_node != "none":
                self.nav_from_closest_edge = False
                o_node = self.rsearch.get_node_from_tmap2(self.closest_node)
                rospy.loginfo("Planning from the closest NODE: {}".format(self.closest_node))
            else:
                self.nav_from_closest_edge = True
                o_node, the_edge = self.orig_node_from_closest_edge(g_node)
                rospy.loginfo("Planning from the closest EDGE: {}".format(the_edge["edge_id"]))
                
            rospy.loginfo("Navigating From Origin %s to Target %s", o_node["node"]["name"], target)
             
            # Everything is Awesome!!!
            # Target and Origin are not None
            if (g_node is not None) and (o_node is not None):
                if g_node["node"]["name"] != o_node["node"]["name"]:
                    route = self.rsearch.search_route(o_node["node"]["name"], target)
                    route = self.enforce_navigable_route(route, target)
                    if route.source:
                        rospy.loginfo("Navigating Case 1: Following route")
                        self.publish_route(route, target)
                        result, inc = self.followRoute(route, target, 0)
                        rospy.loginfo("Navigating Case 1 -> res: %d", inc)
                    else:
                        rospy.logwarn("Navigating Case 1a: There is no route from {} to {}. Check your edges.".format(o_node["node"]["name"], target))
                        self.cancelled = True
                        result = False
                        inc = 1
                        rospy.loginfo("Navigating Case 1a -> res: %d", inc)
                else:      
                    if self.nav_from_closest_edge:
                        result, inc = self.to_goal_node(g_node, the_edge)
                    else:
                        result, inc = self.to_goal_node(g_node)
            else:
                rospy.logwarn("Navigating Case 3: Target or Origin Nodes were not found on Map")
                self.cancelled = True
                result = False
                inc = 1
                rospy.loginfo("Navigating Case 3 -> res: %d", inc)
        
        if (not self.cancelled) and (not self.preempted):
            self._result.success = result
            if result:
                self._feedback.route = target
                self._as.publish_feedback(self._feedback)
                self._as.set_succeeded(self._result)
            else:
                self._feedback.route = self.current_node
                self._as.publish_feedback(self._feedback)
                self._as.set_aborted(self._result)
        else:
            if not self.preempted:
                self._feedback.route = self.current_node
                self._as.publish_feedback(self._feedback)
                self._result.success = result
                self._as.set_aborted(self._result)
            else:
                self._result.success = False
                self._as.set_preempted(self._result)
 

    def execute_policy(self, route, target):
        
        succeeded, inc = self.followRoute(route, target, 1)
        if succeeded:
            rospy.loginfo("Navigation Finished Successfully")
            self.publish_feedback_exec_policy(GoalStatus.SUCCEEDED)
        else:
            if self.cancelled and self.preempted:
                rospy.logwarn("Fatal Fail")
                self.publish_feedback_exec_policy(GoalStatus.PREEMPTED)
            elif self.cancelled:
                rospy.logwarn("Navigation Failed")
                self.publish_feedback_exec_policy(GoalStatus.ABORTED)

        return succeeded
    

    def publish_feedback_exec_policy(self, nav_outcome=None):
        
        if self.current_node == "none":  # Happens due to lag in fetch system
            rospy.sleep(0.5)
            if self.current_node == "none":
                self._feedback_exec_policy.current_wp = self.closest_node
            else:
                self._feedback_exec_policy.current_wp = self.current_node
        else:
            self._feedback_exec_policy.current_wp = self.current_node
        if nav_outcome is not None:
            self._feedback_exec_policy.status = nav_outcome
        self._as_exec_policy.publish_feedback(self._feedback_exec_policy)
        
        
    def orig_node_from_closest_edge(self, g_node):
        
        name_1, _ = get_node_names_from_edge_id_2(self.lnodes, self.closest_edges.edge_ids[0])
        name_2, _ = get_node_names_from_edge_id_2(self.lnodes, self.closest_edges.edge_ids[1])
        
        # Navigate from the closest edge instead of the closest node. First get the closest edges.
        edge_1 = get_edge_from_id_tmap2(self.lnodes, name_1, self.closest_edges.edge_ids[0])
        edge_2 = get_edge_from_id_tmap2(self.lnodes, name_2, self.closest_edges.edge_ids[1])

        # Then get their destination nodes.
        o_node_1 = self.rsearch.get_node_from_tmap2(edge_1["node"])
        o_node_2 = self.rsearch.get_node_from_tmap2(edge_2["node"])

        # If the closest edges are of equal distance (usually a bidirectional edge) 
        # then use the destination node that results in a shorter route to the goal.
        if self.closest_edges.distances[0] == self.closest_edges.distances[1]:
            d1 = get_route_distance(self.lnodes, o_node_1, g_node)
            d2 = get_route_distance(self.lnodes, o_node_2, g_node)
        else: # Use the destination node of the closest edge.
            d1 = 0; d2 = 1
        
        if d1 <= d2:
            return o_node_1, edge_1
        else:
            return o_node_2, edge_2
        
        
    def to_goal_node(self, g_node, the_edge=None):
        
        rospy.loginfo("Target and Origin Nodes are the same")
        self.current_target = g_node["node"]["name"]
        
        if the_edge is None:
            # Check if there is a move_base action in the edges of this node and choose the earliest one in the 
            # list of move_base actions. If not is dangerous to move.
            act_ind = 100
            for i in g_node["node"]["edges"]:
                c_action_server = i["action"]
                if c_action_server in self.move_base_actions:
                    c_ind = self.move_base_actions.index(c_action_server)
                    if c_ind < act_ind:
                        act_ind = c_ind
                        the_edge = i

        if the_edge is None:
            rospy.logwarn("Navigating Case 2: Could not find a move base action in the edges of target {}. Unsafe to move".format(g_node["node"]["name"]))
            rospy.loginfo("Action not taken, outputting success")
            result=True
            inc=0
            rospy.loginfo("Navigating Case 2 -> res: %d", inc)
        else:
            rospy.loginfo("Navigating Case 2a: Getting to the exact pose of target {}".format(g_node["node"]["name"]))
            self.current_target = g_node["node"]["name"]
            origin_node,_ = get_node_names_from_edge_id_2(self.lnodes, the_edge)
            result, inc = self.execute_action(the_edge, g_node, origin_node)
            if not result:
                rospy.logwarn("Navigation Failed")
                inc=1
            else:
                rospy.loginfo("Navigation Finished Successfully")
            rospy.loginfo("Navigating Case 2a -> res: %d", inc)
            
        return result, inc


    def enforce_navigable_route(self, route, target_node):
        """
        Enforces the route to always contain the initial edge that leads the robot to the first node in the given route.
        In other words, avoid that the route contains an initial edge that is too far from the robot pose. 
        """
        if self.nav_from_closest_edge:
            if not(self.closest_edges.edge_ids[0] in route.edge_id or self.closest_edges.edge_ids[1] in route.edge_id):
                first_node = route.source[0] if len(route.source) > 0 else target_node
                
                for edge_id in self.closest_edges.edge_ids:
                    origin, destination = get_node_names_from_edge_id_2(self.lnodes, edge_id)
                    
                    if destination == first_node:
                        route.source.insert(0, origin)
                        route.edge_id.insert(0, edge_id)
                        break
        return route


    def followRoute(self, route, target, exec_policy):
        """
        This function follows the chosen route to reach the goal.
        """
        self.navigation_activated = True
        nnodes = len(route.source)
        Orig = route.source[0]
        Targ = target
        self._target = Targ

        self.init_reconfigure()

        rospy.loginfo("%d Nodes on route" % nnodes)

        inc = 1
        rindex = 0
        nav_ok = True
        route_len = len(route.edge_id)
        self.fluid_navigation = True

        o_node = self.rsearch.get_node_from_tmap2(Orig)
        edge_from_id = get_edge_from_id_tmap2(self.lnodes, route.source[0], route.edge_id[0])
        if edge_from_id:
            a = edge_from_id["action"]
            rospy.loginfo("First action: %s" % a)
        else:
            rospy.logerr("Failed to get edge from id {}. Invalid route".format(route.edge_id[0]))
            return False, inc
        
        if not self.nav_from_closest_edge:        
            # If the robot is not on a node or the first action is not move base type
            # navigate to closest node waypoint (only when first action is not move base)
            if a not in self.move_base_actions:
                rospy.loginfo("The action of the first edge in the route is not a move base action")
                rospy.loginfo("Current node is {}".format(self.current_node))
                
            if self.current_node == "none" and a not in self.move_base_actions:
                self.next_action = a
                rospy.loginfo("Do %s to origin %s" % (self.move_base_name, o_node["node"]["name"]))
    
                # 5 degrees tolerance
                params = {"yaw_goal_tolerance": 0.087266}
                self.reconfigure_movebase_params(params)

                self.current_target = Orig
                nav_ok, inc = self.execute_action(self.move_base_edge, o_node)
                rospy.loginfo("Navigation Finished Successfully") if nav_ok else rospy.logwarn("Navigation Failed")
                
            elif a not in self.move_base_actions:
                move_base_act = False
                for i in o_node["node"]["edges"]:
                    # Check if there is a move_base action in the edages of this node
                    # if not is dangerous to move
                    if i["action"] in self.move_base_actions:
                        move_base_act = True
                
                if not move_base_act:
                    rospy.logwarn("Could not find a move base action in the edges of origin {}. Unsafe to move".format(o_node["node"]["name"]))
                    rospy.loginfo("Action not taken, outputing success")
                    nav_ok = True
                    inc = 0
                else:
                    rospy.loginfo("Getting to the exact pose of origin {}".format(o_node["node"]["name"]))
                    self.current_target = Orig
                    nav_ok, inc = self.execute_action(self.move_base_edge, o_node)
                    rospy.loginfo("Navigation Finished Successfully") if nav_ok else rospy.logwarn("Navigation Failed")
                

        while rindex < (len(route.edge_id)) and not self.cancelled and nav_ok:
            
            cedg = get_edge_from_id_tmap2(self.lnodes, route.source[rindex], route.edge_id[rindex])
            a = cedg["action"]
            
            if rindex < (route_len - 1):
                nedge = get_edge_from_id_tmap2(self.lnodes, route.source[rindex + 1], route.edge_id[rindex + 1])
                a1 = nedge["action"]
                self.fluid_navigation = nedge["fluid_navigation"]
            else:
                nedge = None
                a1 = "none"
                self.fluid_navigation = False

            self.current_action = a
            self.next_action = a1

            rospy.loginfo("From %s do (%s) to %s" % (route.source[rindex], a, cedg["node"]))

            current_edge = "%s--%s" % (cedg["edge_id"], self.topol_map)
            rospy.loginfo("Current edge: %s" % current_edge)
            self.cur_edge.publish(current_edge)

            if not exec_policy:
                self._feedback.route = "%s to %s using %s" % (route.source[rindex], cedg["node"], a)
                self._as.publish_feedback(self._feedback)
            else:
                self.publish_feedback_exec_policy()

            cnode = self.rsearch.get_node_from_tmap2(cedg["node"])
            onode = self.rsearch.get_node_from_tmap2(route.source[rindex])

            # do not care for the orientation of the waypoint if is not the last waypoint AND
            # the current and following action are move_base or human_aware_navigation
            # and when the fuild_navigation is true
            if rindex < route_len - 1 and a1 in self.move_base_actions and a in self.move_base_actions and self.fluid_navigation:
                self.reconf_movebase(cedg, cnode, True)
            else:
                if self.no_orientation:
                    self.reconf_movebase(cedg, cnode, True)
                else:
                    self.reconf_movebase(cedg, cnode, False)

            self.current_target = cedg["node"]

            self.stat = nav_stats(route.source[rindex], cedg["node"], self.topol_map, cedg["edge_id"])
            dt_text = self.stat.get_start_time_str()

            if self.edge_reconfigure:
                if not self.srv_edge_reconfigure:
                    self.edgeReconfigureManager.register_edge(cedg)
                    self.edgeReconfigureManager.initialise()
                    self.edgeReconfigureManager.reconfigure()
                else:
                    self.edgeReconfigureManager.srv_reconfigure(cedg["edge_id"])
            
            nav_ok, inc = self.execute_action(cedg, cnode, onode)
            if self.edge_reconfigure and not self.srv_edge_reconfigure and self.edgeReconfigureManager.active:
                self.edgeReconfigureManager._reset()
                rospy.sleep(rospy.Duration.from_sec(0.3))

            params = {"yaw_goal_tolerance": 0.087266, "xy_goal_tolerance": 0.1}
            self.reconfigure_movebase_params(params)

            not_fatal = nav_ok
            if self.cancelled:
                nav_ok = True
            if self.preempted:
                not_fatal = False
                nav_ok = False

            self.stat.set_ended(self.current_node)
            dt_text=self.stat.get_finish_time_str()
            operation_time = self.stat.operation_time
            time_to_wp = self.stat.time_to_wp

            if nav_ok:
                self.stat.status = "success"
                rospy.loginfo("Navigation Finished on %s (%d/%d)" % (dt_text, operation_time, time_to_wp))
            else:
                if not_fatal:
                    rospy.logwarn("Navigation Failed on %s (%d/%d)" % (dt_text, operation_time, time_to_wp))
                    self.stat.status = "failed"
                else:
                    rospy.logwarn("Fatal Fail on %s (%d/%d)" % (dt_text, operation_time, time_to_wp))
                    self.stat.status = "fatal"

            self.publish_stats()

            current_edge = "none"
            self.cur_edge.publish(current_edge)

            self.current_action = "none"
            self.next_action = "none"
            rindex = rindex + 1
            
           

        self.reset_reconf()

        self.navigation_activated = False

        result = nav_ok
        return result, inc


    def cancel_current_action(self, timeout_secs=-1):
        """
        Cancels the action currently in execution. Returns True if the current goal is correctly ended.
        """
        rospy.loginfo("Cancelling current navigation goal, timeout_secs = {}...".format(timeout_secs))
        
        self.edge_action_manager.preempt()
        self.cancelled = True

        if timeout_secs > 0:
            stime = rospy.get_rostime()
            timeout = rospy.Duration().from_sec(timeout_secs)
            while self.navigation_activated:
                if (rospy.get_rostime() - stime) > timeout:
                    rospy.loginfo("\t[timeout called]")
                    break
                rospy.sleep(0.2)

        rospy.loginfo("Navigation active: " + str(self.navigation_activated))
        return not self.navigation_activated


    def publish_route(self, route, target):
        
        stroute = topological_navigation_msgs.msg.TopologicalRoute()
        for i in route.source:
            stroute.nodes.append(i)
        stroute.nodes.append(target)
        self.route_pub.publish(stroute)
        

    def publish_stats(self):
        
        pubst = NavStatistics()
        pubst.edge_id = self.stat.edge_id
        pubst.status = self.stat.status
        pubst.origin = self.stat.origin
        pubst.target = self.stat.target
        pubst.topological_map = self.stat.topological_map
        pubst.final_node = self.stat.final_node
        pubst.time_to_waypoint = self.stat.time_to_wp
        pubst.operation_time = self.stat.operation_time
        pubst.date_started = self.stat.get_start_time_str()
        pubst.date_at_node = self.stat.date_at_node.strftime("%A, %B %d %Y, at %H:%M:%S hours")
        
        pubst.date_finished = self.stat.get_finish_time_str()
        self.stats_pub.publish(pubst)
        self.stat = None
        

    def execute_action(self, edge, destination_node, origin_node=None):
        ####################################################################
        ################################################################
        if self.past_operation!=self.robot_operation and self.robot_operation!=1: #if a new goal (humand command) is required for the safety system
            status=GoalStatus.SUCCEEDED
            self.prev_status=status
            self.cancelled=True
            self.goal_reached=True
            result = True
            inc = 0    
        #######################################################################
        else: #normal operation
            inc = 0
            result = True
            self.goal_reached = False
            self.prev_status = None
            
            if self.using_restrictions and edge["edge_id"] != "move_base_edge":
                ## check restrictions for the edge
                rospy.loginfo("Evaluating restrictions on edge {}".format(edge["edge_id"]))
                ev_edge_msg = EvaluateEdgeRequest()
                ev_edge_msg.edge = edge["edge_id"]
                ev_edge_msg.runtime = True
                resp = self.evaluate_edge_srv.call(ev_edge_msg)
                if resp.success and resp.evaluation:
                    #the edge is restricted
                    rospy.logwarn("The edge is restricted, stopping navigation")
                    result = False
                    inc = 1
                    return result, inc
        
                ## check restrictions for the node
                rospy.loginfo("Evaluating restrictions on node {}".format(destination_node["node"]["name"]))
                ev_node_msg = EvaluateNodeRequest()
                ev_node_msg.node = destination_node["node"]["name"]
                ev_node_msg.runtime = True
                resp = self.evaluate_node_srv.call(ev_node_msg)
                if resp.success and resp.evaluation:
                    #the node is restricted
                    rospy.logwarn("The node is restricted, stopping navigation")
                    result = False
                    inc = 1
                    return result, inc
    
            
            self.edge_action_manager.initialise(edge, destination_node, origin_node)
            self.edge_action_manager.execute()
            
            status = self.edge_action_manager.client.get_state()
            self.pub_status(status)
            while (
                (status == GoalStatus.ACTIVE or status == GoalStatus.PENDING)
                and not self.cancelled
                and not self.goal_reached
            ):
                status = self.edge_action_manager.client.get_state()
                self.pub_status(status)
                rospy.sleep(rospy.Duration.from_sec(0.01))
            
            res = self.edge_action_manager.client.get_result()
    
            if status != GoalStatus.SUCCEEDED:
                if not self.goal_reached:
                    result = False
                    if status is GoalStatus.PREEMPTED:
                        self.preempted = True
                else:
                    result = True
    
            if not res:
                if not result:
                    inc = 1
                else:
                    inc = 0
    
            rospy.sleep(rospy.Duration.from_sec(0.5))
            status = self.edge_action_manager.client.get_state()
        ################################################################
        self.pub_status(status)
        
        return result, inc
    
    
    def pub_status(self, status):
        
        if status != self.prev_status:
            d = {}
            d["goal"] = self.edge_action_manager.destination_node["node"]["name"]
            d["action"] = self.edge_action_manager.current_action.upper()
            d["status"] = status_mapping[status]
            
            self.move_act_pub.publish(String(json.dumps(d)))
        self.prev_status = status
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
        self.robot_safety_operation()
        
        
    def robot_safety_operation(self):
        status=self.prev_status
        #TO UPDATE THE ROBOT OPERATION
        if self.hri_status!=0: #if a human is detected            
            if self.robot_operation==1: #if robot is moving to picker location 
                if self.hri_status==2: #if distance <=3.6m
                    self.robot_operation=2 #wait for command to approach
                if self.hri_human_command==1: #if human command is "approach"
                    self.robot_operation=3 
                if self.hri_human_command==2: #if human command is "move_away"
                    self.robot_operation=5
                if self.hri_human_command==3: #if human command is "stop"
                    self.robot_operation=2 #wait for command to approach              
            elif self.robot_operation==2: #if robot is waiting for a human command to approach
                if self.hri_human_command==1: #if human command is "approach"
                    self.robot_operation=3 
                if self.hri_human_command==2: #if human command is "move_away"
                    self.robot_operation=5
            elif self.robot_operation==3: #if robot is approaching to picker (already identified)
                if self.hri_status==3: #if distance <=1.2m
                    self.robot_operation=4 #wait for command to move away
                if self.hri_human_command==2: #if human command is "move_away"
                    self.robot_operation=5
                if self.hri_human_command==3: #if human command is "stop"
                    self.robot_operation=4 #wait for command to move away
                if status==GoalStatus.SUCCEEDED and self.current_node==self.goal: #if robot reached the goal and it is static
                    self.robot_operation=4 #make the robot wait for a command to move away
            elif self.robot_operation==4: #if robot is waiting for a human command to move away
                if self.hri_human_command==1: #if human command is "approach"
                    self.robot_operation=3 
                if self.hri_human_command==2: #if human command is "move_away"
                    self.robot_operation=5
            elif self.robot_operation==5: #if robot is moving away from the picker (it can be after collecting tray or because of human command)
                if self.hri_human_command==1: #if human command is "approach"
                    self.robot_operation=3 
                if self.hri_human_command==3: #if human command is "stop"
                    self.robot_operation=2 #wait for command to approach
                if status==GoalStatus.SUCCEEDED and self.current_node==self.goal: #if robot reached the goal and it is static
                    self.robot_operation=6 #make the robot wait for a new goal
            elif self.robot_operation==6: #if robot is waiting for new goal
                if status==GoalStatus.ACTIVE: #if robot is moving
                    if self.hri_status==1: #if distance is 3.6m to 7m
                        self.robot_operation=1 #robot operation is "moving to goal"
                if self.hri_human_command==1: #if human command is "approach"
                    self.robot_operation=3 
                if self.hri_human_command==2: #if human command is "move_away"
                    self.robot_operation=5
            else: #uv-c treatment
                self.robot_operation=0  
                
        else: #if none human is detected
            if status==GoalStatus.ACTIVE: #if robot is moving
                self.robot_operation=1 #robot operation is "moving to goal"
            if self.current_node==self.goal: #if robot reached the goal
                self.robot_operation=6 #robot operation is "wait for new goal"
        #print("ROBOT OPERATION CALLBACK", self.robot_operation)
        if self.past_operation!=self.robot_operation and self.robot_operation!=1:
            with self.navigation_lock:
                if self.cancel_current_action(timeout_secs=10):
                    # we successfully stopped the previous action, claim the title to activate navigation
                    self.navigation_activated = False
        #Publish current robot operation after successfully stopped the previos action (if required)
        pub_robot = rospy.Publisher('robot_info', robot_msg)
        rob_msg = robot_msg()
        rob_msg.operation=self.robot_operation
        pub_robot.publish(rob_msg)  
        #self.robot_goal_update()
        
    def robot_goal_update(self):
        #status=self.nav_status
        #TO CHANGE THE GOAL ACCORDING TO NEW ROBOT OPERATION
        print("PAST OPERATION",self.past_operation)
        print("NEW OPERATION",self.robot_operation)
                
        if (self.robot_operation==2 or self.robot_operation==4 or self.robot_operation==6):# and status==GoalStatus.ACTIVE: #to make the robot stop
            if self.past_operation!=2 and self.past_operation!=4 and self.past_operation!=6: #a new goal is activated only if it was not activated before
                print("OPERATION CHANGED 1")
                self.past_operation=self.robot_operation
                with self.navigation_lock:
                    if self.cancel_current_action(timeout_secs=10):
                        # we successfully stopped the previous action, claim the title to activate navigation
                        self.navigation_activated = False
                
        elif self.robot_operation==3 or self.robot_operation==5: #to approch/move away after being waiting
            if self.past_operation!=self.robot_operation: #a new goal is activated only if it was not activated before
                print("OPERATION CHANGED 2")
                if self.robot_operation==3: #approach
                    self.goal=self.goal_approach
                else: #robot_operation=5, move away
                    self.goal=self.goal_move_away
                
                can_start = False
                self.past_operation=self.robot_operation
                #self.new_goal= True
                with self.navigation_lock:
                    if self.cancel_current_action(timeout_secs=10):
                        # we successfully stopped the previous action, claim the title to activate navigation
                        self.navigation_activated = True
                        can_start = True
                
                if can_start:
                    self.cancelled = False
                    self.preempted = False
                    self.no_orientation = False
                    self._feedback.route = "Starting..."
                    self._as.publish_feedback(self._feedback)
                    self.navigate(self.goal)

                else:
                    rospy.logwarn("Could not cancel current navigation action, GO-TO-NODE goal aborted")
                    self._as.set_aborted()
                self.navigation_activated = False
    
    def joy_callback(self,data):
        #print("JOY NEW DATA")
        buttons=data.buttons
        if np.shape(buttons)[0]>0:
            if buttons[9]>0: #option to order to move up
                print("UP##############################")
                #goal="WayPoint83"
                #self.hri_safety_action=0
                
            elif buttons[8]>0: #start order to move down
                print("DOWN##############################")
                goal="WayPoint139"
                with self.navigation_lock:
                    if self.cancel_current_action(timeout_secs=10):
                        # we successfully stopped the previous action, claim the title to activate navigation
                        self.navigation_activated = False
                        #can_start = True
                '''
                if can_start:
        
                    self.cancelled = False
                    self.preempted = False
                    self.no_orientation = False
                    
                    self._feedback.route = "Starting..."
                    self._as.publish_feedback(self._feedback)
                    #if self.hri_safety_action==2:
                    #    goal.target="WayPoint39"
                    #print("GOAL TARGET ",goal.target)
                    self.navigate(goal)
        
                else:
                    rospy.logwarn("Could not cancel current navigation action, GO-TO-NODE goal aborted")
                    self._as.set_aborted()
                '''
                self.navigation_activated = False
        
    ##############################################################################################       

if __name__ == "__main__":
    rospy.init_node("topological_navigation")
    mode = "normal"
    server = TopologicalNavServer(rospy.get_name(), mode)
    rate = rospy.Rate(1/0.01) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        #print("ROBOT OPERATION MAIN",server.robot_operation)
        server.robot_goal_update()
        rate.sleep() #to keep fixed the publishing loop rate
        
    rospy.loginfo("Exiting.")
###################################################################################################################