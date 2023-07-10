# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 12:10:56 2022
@author: KrimFiction
"""


#TODO: Interface with IFC device. 
#Test the basic functions. 
#Investigate the different prompts. 
#Look at C code of Ecochip
#Upgrade sampling frequency
#Start sampling. Stop sampling. Low-Power mode. Save to CSV. 

from __future__ import print_function
from msvcrt import kbhit, getwch
import keyboard
import serial
import sys
import time
import serial.tools.list_ports
import os.path
import tkinter as tk
from tkinter import filedialog
import threading
from threading import Thread
from threading import Timer
import utils.py

end_thread = False
key_pressed = False

def IFCProgram():
    global s_obj
    comPort = 'COM12'
    baudrate = '230400'
    utils.BeginComIFC(s_obj,comPort,baudrate)
    
    write_to_file_data = ''
    system_reseted = False
    ask_to_save_csv = False

    print("test 1")

    string_already_received = False
            
    while True:
        time.sleep(0.01)
        if s_obj.in_waiting > 0 or string_already_received:

            print("test 2")
            
            stop_python_prompt = False
            prompt_detected = False
            prompt_during_reading_detected = False
            if string_already_received == False:
                buffer = bytes()
                buffer += s_obj.read(s_obj.in_waiting)
                ascii_data = buffer.decode('ascii')
                
            string_already_received = False
            
            print("test 3")
            print(ascii_data)
            
            for elem in ascii_data:
                if elem == '$':
                    prompt_detected = True
                    break

            for elem in ascii_data:
                if elem == '&':
                    ascii_data = ascii_data.replace('\r\n&', '')
                    print('Ecochip wake up, 10 second to enter a command')

            for elem in ascii_data:
                if elem == '~':
                    s_obj.close()
                    ascii_data = ascii_data.replace('~', '')
                    print(ascii_data, end = '')
                    ascii_data = ''
                    print('EcoChip is now in low power mode')
                    time.sleep(5)
                    while True:
                        time.sleep(1)
                        try:
                            s_obj = serial.Serial(comPort, int(baudrate),
                                                      bytesize=8, timeout=1,
                                                      stopbits=serial.STOPBITS_ONE)
                            print('The EcoChip is reconnected')
                            waked_up_chr = 'Y'
                            time.sleep(0.1)
                            s_obj.write(waked_up_chr.encode())
                            time.sleep(0.1)
                            break
                        except:
                            pass
            if prompt_during_reading_detected == True:
                print('You have 20 seconds to press a key and enter')
                if utils.timed_input('$', 19):
                    input_data = input()
                    s_obj.write(input_data.encode())     
                prompt_during_reading_detected = False

            print(ascii_data, end = '')
            if prompt_detected == True:
                s_obj.flush()

                global key_pressed
                global the_key

                key_pressed = False
                keyboard.on_press(callback=utils.on_press, suppress=False)
                
                while True:
                    time.sleep(0.1)
                    if s_obj.in_waiting > 0:
                        buffer = s_obj.read(s_obj.in_waiting)
                        ascii_data = buffer.decode('ascii')
                        for elem in ascii_data:
                            if elem == '$' or elem == '^' or elem == '~':
                                string_already_received = True;
                                break
                        if string_already_received:
                            break
                        print(ascii_data, end='')
                    if key_pressed:
                        key_pressed = False
                        break
                if string_already_received == False:
                    input_data = input()
                    
                    if input_data == 'reset':
                        print('The program is restarting...')
                        s_obj.flush()
                        s_obj.close()
                        return True
                    elif input_data == 'close':
                        print('The program is closing...')
                        s_obj.flush()
                        s_obj.close()
                        return False
                    elif input_data == 'exit':
                        exit_condition = utils.waitForPrompt(s_obj)
                        if exit_condition == False:
                            return exit_condition
                        stop_python_prompt = True
                    elif input_data == 'reconnect':
                        exit_condition = utils.waitForPrompt(s_obj)
                        if exit_condition == False:
                            return exit_condition
                        stop_python_prompt = True
                    elif input_data == 'n' or input_data == 'o' or input_data == 'p':
                        ask_to_save_csv = True
                        write_to_file_data = ''
                        
                    if stop_python_prompt == False:  
                        s_obj.write(input_data.encode())

        elif s_obj.is_open == False:
            s_obj.close()
            print('The EcoChip is disconnected')
            while True:
                time.sleep(1)
                try:
                    s_obj = serial.Serial(comPort, int(baudrate),bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE)
                    print('The EcoChip is reconnected')
                    waked_up_chr = 'Y'
                    s_obj.write(waked_up_chr.encode())
            
                    break
                except:
                    pass



program_running = True
while program_running:
    program_running = IFCProgram()

