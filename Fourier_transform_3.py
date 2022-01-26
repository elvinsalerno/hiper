# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 18:09:46 2021

@author: hiper
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:01:44 2021

@author: hiper
"""




import matplotlib.pyplot as plt

import numpy as np


file_directory=r"C:\SpecManData\Elvin\MG_VIII_530_Gd3_1percent\Round_2\12082021\chirped_exc_and_det"
filename_in="Fourier_det_obs93.4to94.5GHz_4dB_3.390T_center94Ghz"

skip_n_points=0
#n2=2500

Col=10




filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.DAT'


font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)

raw_data = np.genfromtxt(filename_comby,skip_header=1)#,delimiter=',')





times=raw_data[:,0]
data_im=raw_data[:,2*Col]
data_re=raw_data[:,2*Col-1]




plt.figure(1)

plt.plot(times,data_im)
plt.plot(times,data_re)





#data = [data_im[i]+ 1j* data_re[i] for i in range(len(data_im)) ]
data = [data_re[i]+ 1j* data_im[i] for i in range(len(data_im)) ]
#print(data)
#data_mag=(np.sqrt((np.square(data_im))+(np.square(data_re))))

T=times[1]-times[0]


from scipy.fft import fft, fftfreq
# Number of sample points
N = len(times)

x = times
y =data-np.mean(data)
yf = fft(y)

#print(yf)


from scipy.signal import blackman
w = blackman(N)
ywf = fft(y*w)
xf = fftfreq(N, T)#[:N//2]


############################

#fftData = np.fft.fft(data)
#freq = np.fft.fftfreq(lenData, 1/fSamp)
yf = np.fft.fftshift(yf)
xf = np.fft.fftshift(xf)


#yf=(np.imag(yf))
###########################








yf=np.abs(yf)

savGol='on'
from scipy.signal import savgol_filter
if savGol=='on':
    yf=savgol_filter(yf, 15, 2)



xf=xf/1e9
xf=xf#+94


plt.figure(2)

plt.xlim(-0.3,0.3)

#print(xf)

plt.plot(xf, yf, '-b')
#plt.xlim(70,110)
#plt.xlim(-.2,.2)
#plt.ylim(0,6)
plt.xlabel("GHz")