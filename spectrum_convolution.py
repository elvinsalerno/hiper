# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 10:25:54 2021

@author: hiper
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy


file_directory=r"C:\SpecManData\Elvin\MG_VIII_530_Gd3_1percent"
filename_in="MG_VIII_530_400_1800_800_52dB_crystal_5K_Data"


fields_array=[]
data_mag_array=[]

filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.txt'
raw_data = np.genfromtxt(filename_comby,skip_header=2,delimiter=',')
field=raw_data[:,0]
data_Re=raw_data[:,1]
data_Im=raw_data[:,2]
data_mag=(np.sqrt((np.square(data_Im))+(np.square(data_Re))))
#freq = (93.55*3.306)/field

savGol='off'
from scipy.signal import savgol_filter
if savGol=='on':
    data_mag=savgol_filter(data_mag, 11, 2)

plt.plot(field,data_Re)
plt.plot(field,data_Im)
plt.plot(field,data_mag)

#plt.axvline(x = 94.40)
#plt.axvline(x = 93.55)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


data_mag=NormalizeData(data_mag)


###############################################################################
###############################################################################
exp_freq=94.0
pick_field=3.42


g_list=(exp_freq)/(13.94)/(field)
g_list=np.array(g_list)
freq_vals=pick_field*13.94*g_list



f_data=scipy.interpolate.interp1d(freq_vals,data_mag)

freq_spaces=np.linspace(freq_vals[0],freq_vals[-1],10000)


data_interp=f_data(freq_spaces)

freq_vals2=freq_spaces-94

plt.figure(55)
plt.plot(freq_vals2,data_interp)


###############################################################################
###############################################################################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft , fftfreq

N = 10000
# sample spacing
T = 10000 / N

x = np.linspace(0.0 , N*T , N)



def step(x):
    if x < 10 or x >20 :
        return 0
    else:
        return 1 
   


   
f = [step(y) for y in x]
#f = np.sin(x)
yf = fft(f)
yf = NormalizeData(np.abs(yf))
xf = fftfreq(N , T)

plt.figure(2)
plt.plot(x, f, 'r')

plt.figure(3)
plt.plot(xf , yf , 'b')
plt.xlim(-0.2,0.2)
plt.xlabel("GHz")

###############################################################################
###############################################################################

plt.figure(4)


#conv_signal=scipy.signal.convolve(data_interp,yf)


#freq_spaces_3=np.linspace(freq_vals2[0],freq_vals2[-1],len(conv_signal))
#plt.plot(freq_spaces_3,conv_signal)




###############################################################################
###############################################################################

plt.figure(44)
mult_signal=[]
for num1, num2 in zip(data_interp,yf):
	mult_signal.append(num1*num2)



#mult_signal=data_interp*yf
freq_spaces_4=np.linspace(freq_vals2[0],freq_vals2[-1],len(mult_signal))
plt.plot(freq_spaces_4,mult_signal)






