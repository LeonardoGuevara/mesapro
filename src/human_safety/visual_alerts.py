#!/usr/bin/env python

import rospy 
import time
import threading # Needed for Timer

from mesapro.msg import hri_msg
from std_msgs.msg import String
      
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.critical_dist=0
        self.audio_message=0        
        self.current_alert="none"
        self.new_goal="Unknown"
        self.time_without_msg=5                 # Maximum time without receiving safety messages
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        self.problem=False #Flag to know if safety system or human perception have problems
        
    def safety_callback(self,safety_info):
        self.status=safety_info.hri_status
        self.critical_dist=safety_info.critical_dist
        self.new_goal=safety_info.new_goal   
        self.audio_message=safety_info.audio_message
        if self.audio_message==8:
             self.problem=True #to alert that human perception system is not publishing 
        print("Safety message received")
        self.timer_safety.cancel()
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        
    def safety_timeout(self):
        print("No safety message received in a long time")
        self.problem=True #to alert that safety system is not publishing 
        
    def activate_alerts(self):
        if self.status==1:
            self.current_alert="green" #means safety interaction
        elif self.status==2:
            self.current_alert="yellow" #means critical interaction
        elif self.status==3:
            self.current_alert="red" #means dangerous interaction
        elif self.status==0: 
            self.current_alert="yellow_blink" #means that the safety system and human perception are running, but no human is detected
        if self.problem==True: 
            self.current_alert="red_blink" # means that there are problems with the safety system or human perception system
        if self.new_goal=="Unknown":
            self.current_alert="none" #means robot is not moving autonomously yet
        
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
        
        
