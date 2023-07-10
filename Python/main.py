# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 12:10:56 2022
@author: KrimFiction
"""

from __future__ import print_function
import keyboard
import serial
import time
import serial.tools.list_ports
import utils
import global_
from collections import deque
import matplotlib.pyplot as plt
import os
import csv

end_thread = False
key_pressed = False

def IFCProgram():
    global s_obj
    comPort = 'COM6'
    baudrate = '460800'
    number_data = 128        #min 2
    
    #Initialize moving figure
    fig = plt.figure()
    gs = fig.add_gridspec(3,2, hspace=0,wspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle('R1, R2, Ima1 and Ima2')
      
    # name of csv file to save to
    filename = "lastData.csv"
    
    s_obj = utils.BeginComIFC(comPort,baudrate)
    if not s_obj:    
        return False
    
    global data
    global dataLSB
    global dataMSB
    global count
    global sampled_r1
    global sampled_r2
    global sampled_ima1
    global sampled_ima2
    global sampled_5V
    
    #Deque for increased speed
    count,sampled_r1,sampled_r2,sampled_ima1,sampled_ima2,sampled_5V = deque(), deque(), deque(), deque(), deque(), deque()
    
    
    #Begin in Full-Power Paused SingleFreq mode
    singleFreqFlag = 1
    PauseMeasurementFlag = 1
    PeakFlag = 1
    LowPowerFlag = 0
    single_freq_chr = 'S'
    s_obj.write(single_freq_chr.encode())
    pause_chr = 'P'
    s_obj.write(pause_chr.encode())
    while True:         
        time1,time2 = time.time(),0
        number_bytes_transmitted, number_bytes = 0,0
        number_count, max_count, min_count, total_data = 0, 0, 60000, 0
        buffer = [] #Buffer for data
        global_.key_pressed = False
        keyboard.on_press(callback=utils.on_press, suppress=False)
        s_obj.reset_input_buffer()
        s_obj.reset_output_buffer()
        
        #Main loop for data sampling, breaks when keyboard is pressed.
        #The IFC sends 'number_data' samples of COUNT, followed by the same
        #amount of SAMPLED_R1, followed by the same amount of SAMPLED_R2, 
        #etc. until SAMPLED_5V. This script simply samples the data buffers, 
        #and decodes these 12-bits ADC data to their true INT values.
        while PauseMeasurementFlag==0:
            #Wait for at least 3 full data sampling
            if s_obj.in_waiting >= (6*number_data+2)*2:
                number_bytes = s_obj.in_waiting
                number_bytes_transmitted += number_bytes
                data,dataMSB,dataLSB = bytes(),bytes(),bytes()
                data += s_obj.read(number_bytes)
                
                #Separate data in MSB and LSB
                dataLSB = data[0::2]
                dataMSB = data[1::2]
                
                #Put the data in int format, try for little or big endian
                buffer += [int.from_bytes([m,l],byteorder='big') for m,l in zip(dataMSB,dataLSB)]
                try: 
                    pos = buffer.index(5)
                except:
                    try:
                        pos = buffer.index(1280)
                        dataLSB = data[1::2]
                        dataMSB = data[0::2]
                        buffer = [int.from_bytes([m,l],byteorder='big') for m,l in zip(dataMSB,dataLSB)]
                    except:
                        pos = 0
                        print('Handshake not found; ', end='')
                    print("Reversed order; ", end='')
                
                data_treated = 0
                number_cycle = 0
                while len(buffer)>(6*number_data+2):
                    try:
                        pos = buffer.index(5)
                        pos2 = buffer.index(6)
                    except:
                        buffer = []
                        print("Error finding handshakes")
                        break
                    buffer_loop = buffer[pos:pos2] 
                    buffer = buffer[pos2+1:]
                    if (len(buffer_loop)-1)<6*number_data:
                        print('Incomplete Buffer')
                        buffer = []
                        break
                    data_treated += (len(buffer_loop)-1)
                    number_cycle += 1
                    
                    i,j = -1,0
                    for buf in buffer_loop:
                        #print(buf, end=' ')
                        #TODO: if buf=6, la fin du handshake.
                        if (i==-1):
                            if (buf==5):     #Handshake
                                i,j = 0,0
                            else:
                                continue
                        elif (i==0):      #count or frequency
                            if singleFreqFlag:
                                count.append(buf-300)
                            else:
                                count.append(buf*10000)    #Freq in kHz
                            j = j+1
                            if j>=number_data:
                                i,j = 1,0
                        elif (i==1):    #r1 
                            sampled_r1.append(buf)
                            j = j+1
                            if j>=number_data:
                                i,j = 2,0
                        elif (i==2):    #r2
                            sampled_r2.append(buf)
                            j = j+1
                            if j>=number_data:
                                i,j = 3,0
                        elif (i==3):    #ima1
                            sampled_ima1.append(buf)
                            j = j+1
                            if j>=number_data:
                                i,j = 4,0
                        elif (i==4):    #ima2
                            sampled_ima2.append(buf)
                            j = j+1
                            if j>=number_data:
                                i,j = 5,0
                        elif (i==5):    #5V
                            sampled_5V.append(buf)
                            j = j+1
                            if j>=number_data:
                                i,j = -1,0      
                                
                #Prints important information when the buffer is empty
                print('Bytes:',number_bytes, '(', data_treated*2,
                          '), Data:', int(data_treated/6), '/', number_bytes//12, ', Cycles:',
                          number_cycle)
                
                total_data += int(data_treated/6)
                try:
                    if singleFreqFlag:
                        max_count = max([buffer_loop[number_data], max_count])
                        min_count = min([buffer_loop[number_data],min_count])
                        if buffer_loop[number_data] > max_count:
                            max_count = 0
                            number_count += 60000
                    else:
                        min_count = 0
                except:
                    pass
                        
            else:
                #pass
                time.sleep(0.001)
            if global_.key_pressed:
                global_.key_pressed = False
                print('Key pressed')
                s_obj.reset_input_buffer()
                s_obj.reset_output_buffer()
                time2 += time.time() - time1
                try:
                    print("%2d data (%2d) in %2.2f seconds; %2.0f B/s @ %2.1f %%; %2.2f kHz" % 
                      (len(count), total_data, time2, number_bytes_transmitted/time2,
                       total_data*2*6/number_bytes_transmitted*100, (total_data)/time2/1000))
                    print("%2d number_count, %2d 5V, %2d min_count, %2d bytesTransmitted" % 
                       (number_count, sampled_5V[-1], min_count, number_bytes_transmitted))
                except:
                    pass
                time1 = time.time()
                break
            
        #When a key is pressed, ask for a command. 
        #X for handshake, S for SingleFreq, M for MultipleFreq, P for Pause,
        #R for Resume, G for Graphics
        input_data = input("|X|S|M|P|R|D|G|F|K|C|1|2|3|4|reset|close|save|\n")
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
        elif input_data == 'save':
            #Pause the sampling when saving data
            PauseMeasurementFlag = 1
            print('Saving to CSV in progress...')
            s_obj.write('P'.encode())
            fields = ['Count', 'Sampled_r1', 'Sampled_r2', 'Sampled_ima1', 'Sampled_ima2', 'Sampled_5V',]
            try:    
                os.remove(filename) #Remove the file if it exists already
            except:
                pass        
            # writing to csv file
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvwriter.writerows([p for p in zip(count, sampled_r1, sampled_r2, sampled_ima1, sampled_ima2, sampled_5V)])
        elif input_data == 'K': #Only sample the peaks at higher frequency
            PeakFlag = 1
            print('Peak sampling activated.')
            s_obj.write('K'.encode())
        elif input_data == 'C': #Constantly sample at lower frequency
            PeakFlag = 0
            print('Low-frequency sampling activated.')
            s_obj.write('C'.encode())
        elif input_data == 'S': #Measurement at a single frequency
            singleFreqFlag = 1
            print('Single frequency sampling activated.')
            s_obj.write(input_data.encode())
        elif input_data == 'P': #Pause the measurements
            PauseMeasurementFlag = 1
            print('Paused.')
            s_obj.write(input_data.encode())
        elif input_data == 'M': #Measurement at more than one frequency
            singleFreqFlag = 0
            print('Multiple frequency sampling activated...')
            s_obj.write(input_data.encode())
        elif input_data == 'R': #Resume sampling
            PauseMeasurementFlag = 0
            print('Start!')
            s_obj.write(input_data.encode())
        elif input_data == 'D': #Set difference value for peak measurement
            #20 is good for 20k,  30 is good for 47k.    
            s_obj.write(input_data.encode())   
            input_data = input("Input the peak difference value as 2 digits (EX: 05 for 5)\n")
            s_obj.write(input_data.encode()) 
        elif input_data == 'L': #Low-Power mode
            LowPowerFlag = 1
            print('Power OFF.')
            s_obj.write(input_data.encode())
        elif input_data == 'N': #Normal-Power mode
            #s_obj.close()  
            #s_obj = utils.BeginComIFC(comPort,baudrate)
            #s_obj.write(single_freq_chr.encode())
            #s_obj.write(pause_chr.encode())
            LowPowerFlag = 0
            print('Power ON.')
            s_obj.write(input_data.encode())
        elif input_data == 'G': #Display the result in Python
            PauseMeasurementFlag = 1
            print('Plotting the curves...')
            s_obj.write('P'.encode())
            if singleFreqFlag==0:
                plt.xscale("log")
            #sampled_r2 = sampled_r1
            #sampled_ima2 = sampled_ima1
            #sampled_5V = sampled_ima2
            axs[0,0].scatter(count, sampled_r1,color='red')
            axs[0,1].scatter(count, sampled_r2,color='red')
            axs[1,0].scatter(count, sampled_ima1,color='red')
            axs[1,1].scatter(count, sampled_ima2,color='red')
            axs[2,0].scatter(count, count,color='red')
            axs[2,1].scatter(count, sampled_5V,color='red')
        elif input_data == 'F': #Flush the sampled data
            print('Flushing the sampled data...')   
            s_obj.flushOutput()
            count.clear()
            sampled_r1.clear()
            sampled_r2.clear()
            sampled_ima1.clear()
            sampled_ima2.clear()
            sampled_5V.clear()

        else:  #Try to send the input anyway
            print('Operation in progress...')
            s_obj.write(input_data.encode())
        
        #If the connection is cut, wait for reconnection. Begin in Paused Single Freq
        if s_obj.is_open == False:
            s_obj.close()
            print('The IFC is disconnected')
            while True:
                time.sleep(1)
                try:
                    s_obj = serial.Serial(comPort, int(baudrate),
                          bytesize=8, timeout=1, parity = serial.PARITY_NONE,
                          stopbits=serial.STOPBITS_TWO)
                    print('The IFC is reconnected')
                    s_obj.write('P'.encode())
                    s_obj.write('S'.encode())
                    time.sleep(0.2)
                    break
                except:
                    pass
        


#The effective code
program_running = True
while program_running:
    program_running = IFCProgram()

