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

times=raw_data[:,0]
data_im=raw_data[:,2]
data_re=raw_data[:,1]

data = [data_re[i]+ 1j* data_im[i] for i in range(len(data_im)) ]
print(data)
data_mag=(np.sqrt((np.square(data_im))+(np.square(data_re))))

T=(times[1]-times[0])


from scipy.fft import fft, fftfreq
# Number of sample points
N = len(times)

x = times
y =data
yf = fft(y)


from scipy.signal import blackman
w = blackman(N)
ywf = fft(y*w)
xf = fftfreq(N, T)#[:N//2]

#print(xf)
import matplotlib.pyplot as plt
plt.plot(xf*1000, np.abs(yf), '-b')
plt.xlim(-0.2*1000,0.2*1000)
plt.ylim(0,5000)
plt.xlabel("MHz")