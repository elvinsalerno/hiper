# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:30:39 2021

@author: hiper
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft , fftfreq

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

###################################################################

file_directory=r"E:\111021" 
filename_in="FSE_5K_30_400_60ns_21db_up_2_Data"
filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.txt'

pick_field = 3.32 # T
experiment_freq = 93.55 # GHz

##################################################################

font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)


fig = plt.figure(num=1,figsize=(8.25,5.25), dpi=300)


raw_data = np.genfromtxt(filename_comby,skip_header=2,delimiter=',')
field=raw_data[:,0]
data_Re=raw_data[:,1]
data_Im=raw_data[:,2]
data_mag=(np.sqrt((np.square(data_Im))+(np.square(data_Re))))
#freq = (93.55*3.306)/field
data_mag = NormalizeData(data_mag)


freq_vals=[]
for item in field:
    freq_vals.append((experiment_freq*pick_field)/item)

####################################################################
# FFT Stuff

N = 1000000
# sample spacing
T = 10000 / N
x = np.linspace(0.0 , N*T , N)
def step(x, a , b): # for example step of 10 ns , step(x, 10, 20)
    if x < a or x > b :
        return 0
    else:
        return 1 
   


f = [step(y , 5, 10) for y in x]
#f = np.sin(x)
yf = fft(f)
yf = NormalizeData(np.abs(yf))
xf = fftfreq(N , T)

f1 = [step(y , 5, 10) for y in x]
#f = np.sin(x)
yf1 = fft(f1)
yf1 = NormalizeData(np.abs(yf1))
xf1 = fftfreq(N , T)
#####################################################################
plt.figure(4)
plt.plot(xf + 93.55 , yf , 'b' , label = ' Gd Probe 93.55 GHz')
plt.plot(xf1 + 94.45 , yf1 , 'g' , label = ' Cu Pump 94.45 GHz')
plt.plot(freq_vals, data_mag , 'r')
plt.xlim(92,97)
plt.xlabel("GHz")
plt.legend(loc='upper right', fontsize = 'x-small', shadow = 'true')
plt.show()