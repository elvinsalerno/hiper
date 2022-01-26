# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:44:33 2021

@author: evsal
"""
def NormalizeData(data):
    return data / (np.max(data)) 


#input_directory=r"E:\09222021"
input_directory=r'C:\SpecManData\Gd3NC80Cu\09242021'
input_directory=r'C:\SpecManData\Gd3NC80Cu\09222021'

#input pdb filename, only .xyz file types accepted
#input_filename="DEER_long_1p7us_5K_4dB_93p55_96p45_25nsCu" 
input_filename='DEER_test_4dB_3.30T'
#input_filename='DEER_1625_5K_4dB_93p55_96p4_20nsCu'



input_directory=r'C:\SpecManData\Elvin\MG_VIII_530_Gd3_1percent\Pop_transfer_experiments'
input_filename='t2_MGVIII530_det94.47GHz_10dB_3.400T_long'



#write in terms of ns
corrections=[[150,200],[350,400]]
integ_limit=[200,300]


timescale='us'


###############################################################################

#put 0 and sometyhing really large uinless you want to change
start_time=0
end_time=1500

start_trace=4
end_trace=2000


f_add= 92.2

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
#from scipy.optimize import curve_fit
import matplotlib.cm as cm
from scipy import interpolate



'''
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
'''

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




data=raw_data[start_time:end_time,1:]
time_steps=raw_data[start_time:end_time,0]

data=data.T
time_steps=time_steps.T




##########################################################
corrections_indices=[]
integ_indices=[]


for i in range(0,len(corrections)):
    interm_list=[]
    for j in range(0,2):
        specwavel1=min(time_steps*1e9, key=lambda x:abs(x-corrections[i][j]))
        indexwavel1, = np.where(time_steps*1e9 == specwavel1)
        interm_list.append(indexwavel1[0])
    corrections_indices.append(interm_list)


    
for i in range(0,len(integ_limit)):
    specwavel1=min(time_steps*1e9, key=lambda x:abs(x-integ_limit[i]))
    indexwavel1, = np.where(time_steps*1e9 == specwavel1)
    integ_indices.append(indexwavel1[0])




##########################################################




import re


times_2=[]
for i in range(1,len(times)):
    times_2.append(float(re.findall('\d*\.?\d+',times[i])[0]))
'''
print(times_2)
for i in range(0,len(times_2)):
    if times_2[i]<10:
        times_2[i]=times_2[i]*1000
'''

for i in range(0,len(times_2)):
    if times[i+1][-2]=='u':
        times_2[i]=times_2[i]*1000
    elif times[i+1][-2]=='m':
        times_2[i]=times_2[i]*1000000
    elif times[i+1][-3]=='M':
        times_2[i]=times_2[i]/1000
    else:
        pass







fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
for i in range(0,len(data)):
    points_for_averaging=[]
    for j in range(0,len(corrections_indices)):
        points_for_averaging=data[i][corrections_indices[0][0]:corrections_indices[0][1]]
        #np.append(points_for_averaging,data[i][corrections[j][0]:corrections[j][1]])
    #print(points_for_averaging)
    avg=np.mean(points_for_averaging)
    data[i]=data[i]-avg
    
maxxx=max(map(max, data))
for i in range(0,len(corrections)):
    plt.vlines(corrections[i][0],0,maxxx,color='b')#,label='BL limit 1')
    plt.vlines(corrections[i][1],0,maxxx,color='b')#,label='BL limit 1')

#plt.legend()
from scipy.signal import savgol_filter

#colors = cm.hot(np.linspace(0.,0.3, len(data)))
colors = cm.rainbow(np.linspace(0.,1, len(data)))

for i in range(0,len(data)):
    #data[i]=data[i]/maxxx  
    #data[i] = savgol_filter(data[i], 101, 2)
    plt.plot(time_steps*1e9,data[i],linewidth=0.5,color=colors[i])
    
plt.xlabel('time(ns)')
plt.ylabel('signal')    
plt.title('pick baseline limits')


    
fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
    
magn_list=[]

for i in range(0,int((len(data))/2)):
    im=data[:][2*i-1]
    re=data[:][2*i]
    magn=np.sqrt((im**2)+(re**2))
    #magn=im-re
    magn_list.append(magn)
    #plt.plot(raw_data.T[0,:],re+1)
    #plt.plot(time_steps,magn.T[:])

maxxx2=max(map(max, magn_list))

magn_list=magn_list[start_trace:end_trace]

colors = cm.hot(np.linspace(0.,0.3, len(magn_list)))
colors = cm.rainbow(np.linspace(0.,1, len(magn_list)))

for i in range(0,len(magn_list)):
    magn_list[i]=magn_list[i]/maxxx2
    plt.plot(time_steps*1e9,magn_list[i],linewidth=0.5,color=colors[i])
plt.xlabel('time(ns)')
plt.ylabel('magnitude')
plt.title('integration window')

plt.vlines(integ_limit[0],0,1,color='b')
plt.vlines(integ_limit[1],0,1,color='b')


#plt.ylim(0.5,1)



#print(np.shape(magn_list))

import scipy
from scipy import integrate
#plt.xlim(1.25e-7,3.6e-7)
#import scipy.integrate as integrate
integs=[]




for i in range(0,len(magn_list)):
   xxx=np.sum(magn_list[i][integ_indices[0]:integ_indices[1]])
   #xxx=scipy.integrate.simpson(magn_list[i][integ_indices[0]:integ_indices[1]],time_steps[integ_indices[0]:integ_indices[1]])
   integs.append(xxx)

#print(integs)





#print(np.array(integs)*100000000)

times_3= times_2[::2]
times_3=np.array(times_3)
#print(times_2)
#print(len(integs))

savGol='off'
from scipy.signal import savgol_filter
if savGol=='on':
    integs=savgol_filter(integs, 11, 2)


times_4=f_add+times_3





fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)

if timescale=='us':
    times_4=times_4/1000
    guess=[6,1,1]
else:
    guess=[6,1000,1]




plt.plot(times_4[start_trace:end_trace],NormalizeData(integs))
#plt.xlabel(r'$\tau_{pump}$ (ns)')
plt.xlabel(r'sat pulse freq (GHz)')
plt.ylabel('integrated signal')

#plt.xlim(93.4,94.5)

'''
print(*times_4[start_trace:end_trace],sep=',')
print('\n')
print(*NormalizeData(integs),sep=',')
'''







def expon(t,N0,k,c):#,c,d):
    return N0*np.exp(-t/k)+c


from scipy.optimize import curve_fit



pars, pcov = curve_fit(expon,times_4[start_trace:end_trace],NormalizeData(integs), p0=guess,maxfev=100000)#,bounds=(0,np.inf),maxfev=3800)
print(pars)
#pars_peaks=np.ndarray.tolist(pars_peaks)
#ax1.plot(peaks,corrfunc(peaks,*pars_peaks))#,'--',linewidth=0.5,color=colors[i,:])

#print(pars)

plt.plot(times_4[start_trace:end_trace],expon(times_4[start_trace:end_trace],*pars),'r--',label=str(round(pars[1],2))+' '+timescale)
#corrected_wavelengths=corrfunc(Crystal_wavelengths,*pars_peaks)
#print(type(corrected_wavelengths))

print(' N_0 =',pars[0],'\n','rate1 =',pars[1]**-1,'ns^-1' ,'\n' , 'lifetime =',pars[1],'\n','C =',pars[2])

plt.legend()
plt.xlabel('time ('+timescale+')')
plt.ylabel('echo intensity')


