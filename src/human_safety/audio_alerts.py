#!/usr/bin/env python

import rospy 
import multiprocessing
from playsound import playsound
import time
import yaml
import threading # Needed for Timer
from mesapro.msg import hri_msg

#Importing global parameters from .yaml file
config_direct=rospy.get_param("/hri_audio_alerts/config_direct")
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
audio_direct=parsed_yaml_file.get("directories_config").get("audio_direct") #directory with the voice messages
intervals_long=parsed_yaml_file.get("audio_config").get("intervals_long") # time in which the first version of a message is repeated, in seconds
intervals_short=parsed_yaml_file.get("audio_config").get("intervals_short") # time between two versions of the same message, in seconds
n_languages=rospy.get_param("/hri_audio_alerts/n_languages",1) # how many version of the same message to be reproduced in a loop
#General purposes variables
version=0 #to know which language its been used, initially is English
pub_hz=0.01 #main loop frequency
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.safety_message=0
        self.critical_dist=0        
        self.current_audio=0
        self.new_goal="Unknown"
        self.change_audio=False #flag to know if current message has to me changed
        self.repeat_audio=False #flag to know if current message should be reproduced again
        self.time_audio=(time.time()-time_init) #time when last message was activated
        self.time_without_msg=parsed_yaml_file.get("human_safety_config").get("time_without_msg") # Maximum time without receiving safety messages
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        
    def safety_callback(self,safety_info):
        self.safety_message=safety_info.audio_message
        self.critical_dist=safety_info.critical_dist
        self.new_goal=safety_info.new_goal
        [self.current_audio,self.change_audio]=self.activate_audio()
        #print("Safety message received")
        self.timer_safety.cancel()
        self.timer_safety = threading.Timer(self.time_without_msg,self.safety_timeout) # If "n" seconds elapse, call safety_timeout()
        self.timer_safety.start()
        
    def safety_timeout(self):
        print("No safety message received in a long time")
        self.safety_message=7 #to alert that safety system is not publishing 
        self.current_audio=self.safety_message
        self.change_audio=True
    
    def activate_audio(self):
        new_audio=self.safety_message
        current=self.current_audio
        change=self.change_audio
        if current!=new_audio:
            #self.past_audio=self.current_audio
            #print("OLD",current)
            current=new_audio
            change=True
            #print("NEW",current)
            print("Update audio alert")
        return current,change
        
    def select_message(self,audio_index):
        if version==0:
            folder=audio_direct+"/english/"
            #version=1
        elif version==1:
            folder=audio_direct+"/polish/"
            #version=0
        if audio_index==1:
            audio="warning_uvc_ligth_activated.mp3"
        if audio_index==2:
            audio="paused_waiting_for_a_new_command.mp3"
        if audio_index==3:
            audio="paused_human_is_occluding_my_path.mp3"
        if audio_index==4:
            audio="warning_getting_closer_to_human.mp3" 
        if audio_index==5:
            audio="moving_away_from_human.mp3"
        if audio_index==6:
            audio="moving_to_current_goal.mp3"
        if audio_index==7:
            audio="safety_system_not_working.mp3"
        if audio_index==8:
            audio="human_perception_not_working.mp3"
        if audio_index==9:
            audio="teleoperation_activated.mp3"
        if audio_index==10:
            audio="gesture_control_activated.mp3"
        if audio_index==11:
            audio="paused_collision_detected.mp3"
        message=folder+audio
        return message


###############################################################################################
# Main Script

if __name__ == '__main__':
    time_init=time.time()  
    # Initialize our node       
    hri=hri_class()
    rospy.init_node('audio_alerts',anonymous=True)
    # Setup and call subscription
    rospy.Subscriber('human_safety_info',hri_msg,hri.safety_callback)
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # main loop frecuency in Hz
    #aux=1
    while not rospy.is_shutdown():
        audio_index=hri.current_audio
        if (audio_index==0) or (audio_index==2 and hri.new_goal=="Unknown"): #if there is not a message to be reproduced, or if the first robot goal was assigned
            print("No audio alert")
        else:
             if hri.change_audio==True or hri.repeat_audio==True:
                if hri.change_audio==True:
                    version=0 #always start with the english version
                message=hri.select_message(audio_index)
                p = multiprocessing.Process(target=playsound, args=(message,))  
                p.start()
                hri.time_audio=time.time()
                hri.repeat_audio=False
                hri.change_audio=False
                print("Audio alert is played in version", version)
                while hri.change_audio==False and hri.repeat_audio==False:   
                    #if hri.change_audio==False:
                    if n_languages!=1: #if more than one language
                        if (time.time()-hri.time_audio>=intervals_long[audio_index] and version==1) or (time.time()-hri.time_audio>=intervals_short[audio_index] and version==0):
                            hri.repeat_audio=True
                            #To change the version in the next iteration
                            if version==1:
                                version=0
                            elif version==0:
                                version=1        
                    else: #only english version
                        if time.time()-hri.time_audio>=intervals_long[audio_index]:
                            hri.repeat_audio=True
                    #print("Audio alert is repeated")
                p.join() #to make sure the new message is not overlapping the past message
        rate.sleep() 
        
        
