# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:18:05 2022

@author: KrimFiction
"""
from collections import deque

key_pressed = False
the_key = {0}
line = deque([0.0]*100, maxlen=100)