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


file_directory=r"C:\SpecManData\Elvin\MG_VIII_530_Gd3_1percent\try_Fourier_Detected"
filename_in="chirp_93.5to94.5_4dB_5K_5"

skip_n_points=0
#n2=2500

Col=6




filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.DAT'


font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)

raw_data = np.genfromtxt(filename_comby,skip_header=1)#,delimiter=',')

times=raw_data[:,0]
data_im=raw_data[:,2*Col]
data_re=raw_data[:,2*Col-1]

#data = [data_im[i]+ 1j* data_re[i] for i in range(len(data_im)) ]
data = [data_im[i]+ 1j* data_re[i] for i in range(len(data_im)) ]
print(data)
#data_mag=(np.sqrt((np.square(data_im))+(np.square(data_re))))

T=(times[1]-times[0])


from scipy.fft import fft, fftfreq
# Number of sample points
N = len(times)

x = times
y =data-np.mean(data)
yf = fft(y)

print(yf)


from scipy.signal import blackman
w = blackman(N)
ywf = fft(y*w)
xf = fftfreq(N, T)#[:N//2]


############################

#fftData = np.fft.fft(data)
#freq = np.fft.fftfreq(lenData, 1/fSamp)
yf = np.fft.fftshift(yf)
xf = np.fft.fftshift(xf)


yf=(np.imag(yf))
###########################








yf=np.abs(yf)

savGol='off'
from scipy.signal import savgol_filter
if savGol=='on':
    yf=savgol_filter(yf, 11, 2)






#print(xf)
import matplotlib.pyplot as plt
plt.plot(xf/1e9, yf, '-b')
#plt.xlim(93.79,94.0)
plt.xlim(-.2,.2)
#plt.ylim(0,6)
plt.xlabel("GHz")