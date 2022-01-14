#!/usr/bin/env python

import rospy 
import multiprocessing
from playsound import playsound
import time
from mesapro.msg import hri_msg

intervals=[10,10,10,10,10,10,10] #time between two consecutive voice messages, in seconds, it depedns of each message
        
class hri_class:
    def __init__(self): #It is done only the first iteration
        self.status=0
        self.safety_message=0
        self.critical_dist=0        
        self.current_audio=0
        self.new_goal="Unknown"
        self.change_audio=False #flag to know if current message has to me changed
        self.repeat_audio=False #flag to know if current message should be reproduced again
        self.time_audio=(time.time()-time_init) #time when last message was activated
        
    def safety_callback(self,safety_info):
        self.status=safety_info.hri_status
        self.safety_message=safety_info.audio_message
        self.critical_dist=safety_info.critical_dist
        self.new_goal=safety_info.new_goal
        [self.current_audio,self.change_audio]=self.activate_audio()
    
    def activate_audio(self):
        new_audio=self.safety_message
        current=self.current_audio
        change=self.change_audio
        if current!=new_audio:
            #self.past_audio=self.current_audio
            current=new_audio
            change=True
            print("Update audio alert")
        return current,change
        
    def select_message(self,audio_index):
        folder="/home/leo/rasberry_ws/src/mesapro/audio/"
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
    rate = rospy.Rate(1/0.001) # ROS loop rate in Hz
    #aux=1
    while not rospy.is_shutdown():
        audio_index=hri.current_audio
        if audio_index!=0 and hri.new_goal!="Unknown": #if there is a message to be reprodiced
            if hri.change_audio==True or hri.repeat_audio==True:
                message=hri.select_message(audio_index)
                p = multiprocessing.Process(target=playsound, args=(message,))  
                p.start()
                hri.time_audio=time.time()
                hri.repeat_audio=False
                hri.change_audio=False
                print("Audio alert is played")
                while hri.change_audio==False and hri.repeat_audio==False:
                    if time.time()-hri.time_audio>=intervals[audio_index]:
                        hri.repeat_audio=True
                    #print("Audio alert is repeated")
                p.join() #to make sure the new message is not overlapping the past message
        else:
            print("No audio alert")
        rate.sleep() 
        
        
