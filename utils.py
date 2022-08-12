# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:50:34 2022

@author: KrimFiction
"""

from __future__ import print_function
import serial
import time
import serial.tools.list_ports
import global_


def on_press(key):
    global_.the_key = '{0}'.format(key)
    global_.key_pressed = True

#Waits some time until a key is pressed
def timed_input(prompt='', timeout=None):
    print(prompt, end = '')
    start = time.time()
    global key_pressed
    while time.time() - start < timeout:
        if key_pressed:
            return True
        time.sleep(0.01)
    return False


#Writes a X prompt to the IFC device, then waits for a $ prompt in 
#response to assert that the connexion is working.
def waitForPrompt(s_obj):    
    prompt_chr = 'X'
    prompt_detected = False
    while prompt_detected == False:
        s_obj.flush()
        for i in range(1, 5):
            time.sleep(0.25)
            s_obj.write(prompt_chr.encode()) #Send the prompt to the IFC
            s_obj.flushOutput()
            buffer = bytes()
            time.sleep(0.25)
            while s_obj.in_waiting > 0:
                buffer = s_obj.read(1)
                ascii_data = buffer.decode('ascii')
                for elem in ascii_data:
                    if elem == '$':
                        prompt_detected = True
                        break
                if prompt_detected:
                    break
            if prompt_detected == True:
                print('IFC connected')
                #s_obj.flush()
                return True
            else:
                #s_obj.write(prompt_chr.encode()) #Send the prompt to the IFC
                print('...')

        if prompt_detected == False:
            while True:
                answer = input('Error connecting to the IFC system. Retry? (Y/N)')
                if answer == 'N':
                    s_obj.close();
                    return False
                elif answer == 'Y':
                    break
                else:
                    print('Unknown command')


def BeginComIFC(comPort,baudrate):
    print('***************************************************')
    print('* Communication program with the IFC system LAUNCHED *')
    print('***************************************************')

    print('List of all the availlable COM ports')
    ports = serial.tools.list_ports.comports()
    list_of_ports = []
    for port, desc, hwid in sorted(ports):
        list_of_ports.append(port)
        print(port)
        
    #comPort = input('Please enter the COM port for the EcoChip (ex: COM1):')

    if comPort not in list_of_ports:
        print("COM port not found")

    #baudrate = input('Please enter the baudrate (115200 for BLE, 230400 for USB):')

    try:
        s_obj = serial.Serial(comPort, int(baudrate),
            bytesize=8, timeout=1, parity = serial.PARITY_NONE,
            stopbits=serial.STOPBITS_TWO)
    except:
        print('Error opening the COM port. The program will close')
        return False

    print('COM port open sucessfully')
    print('Trying to connect with the IFC system...')

    time.sleep(1)

    exit_condition = waitForPrompt(s_obj)
    if exit_condition == False:
        return exit_condition
    return s_obj