#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:14:54 2022

@author: leo
"""

import multiprocessing
from playsound import playsound

p = multiprocessing.Process(target=playsound, args=("/home/leo/rasberry_ws/src/mesapro/audio/waiting_for_free_space_to_continue.mp3",))
p.start()

count=0
while count<=1000000:
    
    print(count)
    count=count+1
#input("press ENTER to stop playback")
#p.join()

#p.terminate()