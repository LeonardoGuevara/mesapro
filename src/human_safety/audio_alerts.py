#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:14:54 2022

@author: leo
"""
import rospy 
import multiprocessing
from playsound import playsound
import time
from mesapro.msg import hri_msg

intervals=[3,3,3,3,3,3,3] #time between two consecutive voice messages, in seconds, it depedns of each message
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.safety_message=0
        self.critical_dist=0        
        self.current_audio=0
        self.change_audio=False #flag to know if current message has to me changed
        self.repeat_audio=False #flag to know if current message should be reproduced again
        self.time_audio=(time.time()-time_init) #time when last message was activated
        
    def safety_callback(self,safety_info):
        self.status=safety_info.hri_status
        self.safety_message=safety_info.audio_message
        self.critical_dist=safety_info.critical_dist
        [self.current_audio,self.change_audio]=self.activate_audio()
    
    def activate_audio(self):
        new_audio=self.audio_message
        current=self.current_audio
        change=self.change_audio
        if current!=new_audio:
            #self.past_audio=self.current_audio
            current=new_audio
            change=True
        return current,change
        
    def select_message(self,audio_index):
        folder="/home/leo/rasberry_ws/src/mesapro/audio/"
        if audio_index==1:
            audio="warning_uvc_ligth_activated.mp3"
        if audio_index==2:
            audio="waiting_for_a_new_command.mp3"
        if audio_index==3:
            audio="waiting_for_free_space_to_continue.mp3"
        if audio_index==4:
            audio="approaching_to_human.mp3" 
        if audio_index==5:
            audio="moving_away_from_human.mp3"
        if audio_index==6:
            audio="moving_to_a_goal.mp3"
        message=folder+audio
        return message
#p = multiprocessing.Process(target=playsound, args=("/home/leo/rasberry_ws/src/mesapro/audio/waiting_for_free_space_to_continue.mp3",))
#p.start()
#count=0
#while count<=1000000:
        
#    print(count)
#    count=count+1
#input("press ENTER to stop playback")
#p.join()

#p.terminate()

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
    rate = rospy.Rate(1/0.01) # ROS loop rate in Hz
    while not rospy.is_shutdown():	
        if hri.current_audio!=0: #if there is a message to be reprodiced
            if hri.change_audio==True or hri.repeat_audio==True:
                audio_index=hri.current_audio
                message=hri.select_message(audio_index)
                p = multiprocessing.Process(target=playsound, args=(message,))    
                p.start()
                hri.time_audio=time.time()
                hri.repeat_audio=False
                hri.change_audio=False
                while hri.change_audio==False and hri.repeat_audio==False:
                    if time.time()-hri.time_audio>=intervals[audio_index]:
                        hri.repeat_audio=True
                    print("Audio alert is still activated")
                p.terminate()
            else:
                if time.time()-hri.time_audio>=intervals[audio_index]:
                    hri.repeat_audio=True
        rate.sleep() 
        
        
