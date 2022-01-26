# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:44:33 2021

@author: evsal
"""



#input_directory=r"E:\09222021"
input_directory=r'C:\Users\esalerno\Google Drive\MagLab\Gd3Cu\DEER_092021'


#input pdb filename, only .xyz file types accepted
input_filename="DEER_long_1p7us_5K_4dB_93p55_96p45_25nsCu" 
input_filename='DEER_1625_5K_4dB_93p55_96p4_20nsCu'




#integ_limit=[100,175]
#corrections=[[0,50]]

integ_limit=[100,300]
corrections=[[0,50]]

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
#from scipy.optimize import curve_fit
import matplotlib.cm as cm


import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'


"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# create the root window
root = tk.Tk()
root.title('Tkinter File Dialog')
root.resizable(False, False)
root.geometry('300x150')
filez=[]

def select_files():
    filetypes = (
        ('All files', '*.*'),
        ('text files', '*.txt')
        
    )

    filenames = fd.askopenfilenames(
        title='Open files',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected Files',
        message=filenames
        
    )
    filez.append(filenames)

# open button
open_button = ttk.Button(
    root,
    text='Open Files',
    command=select_files
)

open_button.pack(expand=True)

root.mainloop()
"""

font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)

#fig = go.Figure()



filename_in=input_directory+'\\'+input_filename+'.dat'



#print(combined_name)


times = np.genfromtxt(filename_in,max_rows=1,delimiter='  ',dtype='<U19')
raw_data = np.genfromtxt(filename_in,skip_header=1)

'''
#def plot_fcn(filename_in,lab):
    #import pandas as pd
    #df = pd.read_csv(combined_name, skiprows=1,delimiter = "  ")
    #print(df)
  #times = np.genfromtxt(filename_in,max_rows=1,delimiter='  ',dtype='<U19')
  # raw_data = np.genfromtxt(filename_in,skip_header=1)
    #time=raw_data[:,0]
    #data_Re=raw_data[:,1]
    #data_Im=raw_data[:,2]
    #data_mag=(np.sqrt((data_Im**2)+(data_Re**2)))
    #plt.plot(field,data_mag,label=lab)
    #print(np.shape(raw_data))
 #print(times)
#plot_fcn(combined_name,'xxx')

'''


times = np.genfromtxt(filename_in,max_rows=1,delimiter='  ',dtype='<U19')


raw_data = np.genfromtxt(filename_in,skip_header=1)

data=raw_data[:,1:]
time_steps=raw_data[:,0]

data=data.T
time_steps=time_steps.T




import re
#print(times)
#print(float(re.findall('\d*\.?\d+',times[-1])[0]))

times_2=[]
for i in range(1,len(times)):
    times_2.append(float(re.findall('\d*\.?\d+',times[i])[0]))
    
for i in range(0,len(times_2)):
    if times_2[i]<10:
        times_2[i]=times_2[i]*1000



fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
for i in range(0,len(data)):
    points_for_averaging=[]
    for j in range(0,len(corrections)):
        points_for_averaging=data[i][corrections[0][0]:corrections[0][1]]
        #np.append(points_for_averaging,data[i][corrections[j][0]:corrections[j][1]])
    #print(points_for_averaging)
    avg=np.mean(points_for_averaging)
    #avg=np.mean(data[i][corrections[0][0]:corrections[0][1]])
    data[i]=data[i]-avg
    
maxxx=max(map(max, data))
for i in range(0,len(corrections)):
    plt.vlines(corrections[i][0],0,maxxx,color='k')#,label='BL limit 1')
    plt.vlines(corrections[i][1],0,maxxx,color='k')#,label='BL limit 1')

#plt.legend()
from scipy.signal import savgol_filter


for i in range(0,len(data)):
    #data[i]=data[i]/maxxx
    
    #data[i] = savgol_filter(data[i], 101, 2)
    plt.plot(time_steps*1e9,data[i])
    
plt.xlabel('time(ns)')
plt.ylabel('signal')    


    
fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
    
magn_list=[]

for i in range(0,int((len(data))/2)):
    im=data[:][2*i-1]
    re=data[:][2*i]
    magn=np.sqrt((im**2)+(re**2))
    magn_list.append(magn)
    #plt.plot(raw_data.T[0,:],re+1)
    #plt.plot(time_steps,magn.T[:])

maxxx2=max(map(max, magn_list))

for i in range(0,len(magn_list)):
    magn_list[i]=magn_list[i]/maxxx2
    plt.plot(time_steps*1e9,magn_list[i])
plt.xlabel('time(ns)')
plt.ylabel('magnitude')


plt.vlines(integ_limit[0],0,1)
plt.vlines(integ_limit[1],0,1)


#plt.ylim(0.5,1)



#print(np.shape(magn_list))


#plt.xlim(1.25e-7,3.6e-7)
from scipy import integrate
integs=[]




for i in range(0,len(magn_list)):
    #xxx=np.sum(magn_list[i][integ_limit[0]:integ_limit[1]])
    xxx=integrate.simpson(magn_list[i][integ_limit[0]:integ_limit[1]],time_steps[integ_limit[0]:integ_limit[1]])
    integs.append(xxx)

#print(integs)

fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)

#print(np.array(integs)*100000000)

times_3= times_2[::2]
#print(times_2)
#print(len(integs))

plt.plot(times_3[:],integs)
plt.xlabel('time(ns)')
plt.ylabel('integrated signal')


'''

for i in range(0,len(filez[0])):
    plot_fcn(filez[0][i],'xxx')




#fig = go.Figure()
#fig.add_trace(go.Scatter(x=field, y=data_mag,
#                    mode='lines',
#                    name='test'))

#plt.legend()
plt.xlabel('Field (T)')
plt.ylabel('echo intensity')
plt.title('94 GHz')
plt.legend()
#plt.xlim(3.25,3.4)


'''
#plt.xlim(-300,1500)




