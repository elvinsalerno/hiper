# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:01:44 2021

@author: hiper
"""




import matplotlib.pyplot as plt

import numpy as np


file_directory=r"E:\mike_Gd_crystal"
filename_in="FID_3rdlinemax_2_800_4_4dB"

skip_n_points=0
#n2=2500



filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.txt'


font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)

raw_data = np.genfromtxt(filename_comby,skip_header=3,delimiter=',')

times=raw_data[0]
data_im=raw_data[1]
data_re=raw_data[2]

data_mag=(np.sqrt((np.square(data_im))+(np.square(data_re))))

T=(times[1]-times[0])





from scipy.fft import fft, fftfreq
# Number of sample points
N = len(times)

x = times
y =data_re
yf = fft(y)
from scipy.signal import blackman
w = blackman(N)
ywf = fft(y*w)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')


#print(type(yf))
#print(yf)
#yf_inv=yf[::-1]
#xf_inv=xf[::-1]

#plt.plot(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
#plt.plot(xf[1:N//2],2.0/N * np.abs(yf[1:N//2]),'b')
#plt.plot(-xf_inv[1:N//2],2.0/N *np.abs(yf_inv[N//2:-2]),'b')
    

maximum=max(2.0/N * np.abs(yf[1:N//2]))
max_index,=np.where(2.0/N * np.abs(yf[1:N//2])==maximum)
print('max is %s' %xf[max_index][0], 'GHz')
plt.vlines(xf[max_index],0,maximum,'r')    
plt.vlines(-xf[max_index],0,maximum,'r')    

#plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
#plt.plot(xf,yf)
#plt.legend(['FFT', 'FFT w. window'])
plt.grid()
plt.show()

fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
#plt.plot(times,data_mag)
plt.plot(times,data_im)
plt.plot(times,data_re)
plt.plot(times,data_mag)
plt.xlim(0,400)