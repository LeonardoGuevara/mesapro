#!/usr/bin/env python

import rospy 
import time
from mesapro.msg import hri_msg
from std_msgs.msg import String
      
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=1
        self.critical_dist=0        
        self.current_alert="none"
        self.new_goal="Unknown"
        
    def safety_callback(self,safety_info):
        self.status=safety_info.hri_status
        self.critical_dist=safety_info.critical_dist
        self.new_goal=safety_info.new_goal
        
    def activate_alerts(self):
        if self.status==1:
            self.current_alert="green"
        elif self.status==2:
            self.current_alert="yellow"
        elif self.status==3:
            self.current_alert="red"
        else: #no human detected
            self.current_alert="yellow_blink"
        

###############################################################################################
# Main Script

if __name__ == '__main__':
    time_init=time.time()  
    # Initialize our node       
    hri=hri_class()
    rospy.init_node('visual_alerts',anonymous=True)
    # Setup publisher and subscription
    pub = rospy.Publisher('visual_alerts', String, queue_size=1)
    rospy.Subscriber('human_safety_info',hri_msg,hri.safety_callback)
    #Rate setup
    rate = rospy.Rate(1/0.1) # ROS loop rate in Hz
    #aux=1
    while not rospy.is_shutdown():
        hri.activate_alerts()
        pub.publish(hri.current_alert)
        rate.sleep() 
        
        
