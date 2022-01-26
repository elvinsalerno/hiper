# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:17:38 2022

@author: hiper
"""

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.cm as cm

input_directory=r"C:\SpecManData\Krish\short_biradical_wintyer_break\0.25mM"
input_filename="ChirpEcho_4dB_93.75to94.35_t90_1.6us_100Kshots_94RF"

input_directory=r"C:\SpecManData\Krish\Phase_check"
input_filename="100Mhz_squarePulse"

xlim=[93.5,94.5]

filename_in="".join(["\\",input_filename])
filename_comby=input_directory+filename_in+'.DAT'

skip_n_points=0

cutoff_ind=[110,800]
end_point = 5e-7

Col = 40
f_add=94.0

font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=4,figsize=(3.25,2.25), dpi=300)

raw_data = np.genfromtxt(filename_comby,skip_header=1)#,delimiter=',')


colors = cm.PuRd(np.linspace(0.7,1, len(range(30,40))))

i = 0
for Col in [1]:
    times=raw_data[:,0]
    data_im=raw_data[:,2*Col]
    data_re=raw_data[:,2*Col-1]
    
    '''
    data_re=data_re[:-1]
    data_im=data_im[1:]
    times=times[:-1]
    '''
    data_re=data_re[1:]
    data_im=data_im[1:]
    times=times[1:]
    
    ####################################
    plt.figure(1)
    
    '''
    plt.plot(times,data_im,label='im')
    plt.plot(times,data_re,label='re')
    plt.legend()
    #plt.show()
    plt.xlim(0,end_point)
    
    plt.vlines(times[cutoff_ind[0]],min(data_im),max(data_im),'k',zorder=10)
    plt.vlines(times[cutoff_ind[1]],min(data_im),max(data_im),'k',zorder=9)
    #################################
    '''
    times=times[cutoff_ind[0]:cutoff_ind[1]]
    data_im=data_im[cutoff_ind[0]:cutoff_ind[1]]
    data_re=data_re[cutoff_ind[0]:cutoff_ind[1]]
    
    
    T=times[1]-times[0]
    
    data = [data_re[i]+ 1j* data_im[i] for i in range(len(data_im)) ]
    
    
    # Number of sample points
    N = len(times)
    
    x = times
    y =data-np.mean(data)
    #y=data
    yf = fft(y)
    
    #print(yf)
    
    
    from scipy.signal import blackman
    #w = blackman(N)
    #ywf = fft(y*w)
    xf = fftfreq(N, T)#[:N//2]
    
    ############################
    yf = np.fft.fftshift(yf)
    xf = np.fft.fftshift(xf)
    ###########################
    
        
    #yf=np.abs(yf)
    
    savGol='fon'
    from scipy.signal import savgol_filter
    if savGol=='on':
        yf=savgol_filter(yf, 13, 2)
    
    
    xf=xf/1e9+f_add
    
    
    # replace zero frequency point by the average
    
    idx = np.where(xf == 94.00)[0][0]
    avg_point = (yf[idx - 1] + yf[idx +1])/2
    yf = np.where(xf == 94.00 , avg_point , yf)
    
    
    ################################
    
    plt.figure(3)
    #plt.plot(cut_times_list[0],cut_re_list[0], 'r')
    
    ################################
    
    plt.figure(2)
    plt.plot(xf, np.real(yf), '-', color = colors[i], linewidth=0.2)
    i += 1
    print(i)
    #plt.xlim(70,110)
    #plt.xlim(-.2,.2)
    #plt.ylim(0,6)
    #plt.xlim(xlim[0],xlim[1])
    plt.xlabel("GHz")
    plt.ylabel('intensity')
    
    
    theta= np.arctan2(np.imag(yf), np.real(yf)) * 180 / np.pi
    
    plt.figure(4)
    plt.plot(xf, theta, 'b-' , linewidth=0.2 )
    #plt.xlim(xlim[0],xlim[1])
    #plt.xlim(93.7,94)
    #plt.ylim(-30,30)
        