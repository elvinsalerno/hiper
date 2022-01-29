# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:36:57 2022

@author: esalerno
"""


#write as a list, eg Choose_type = ['hsec'] or Choose_type=['hsec', 'wurst']
#options are 'wurst', 'hsec', 'gaussian', or 'square'
#Don't use 'square' when actually generating pulse! It is only for 
#visualization of the fourier tranform
Choose_type=['wurst']


#in case of Gaussian/square pulse, only the start and RF matters. But still 
#must put some value for stop
#for a reverse chirp, start at the higher frequency and stop at the lower
#in GHz
start=[93.5]

#in GHz
stop=[94]

#Global RF, typically 94
RF=94


#Another way to change the direction of the chirp is to use the same file
#but just use a negative sign for the mod_depth

###############################################################################

directory_data_out=r'C:\Users\evsal\Google Drive\MagLab\pattern_test'

#write 'yes' or 'no'
save_file='yes'

###############################################################################

#Which pulse to examine, indexing starts at 0
Analyze_index=0


#length of pulse in ns. Only matters for visualization, not for Specman
pulse_length_ns=100


N_points=10000

#limits of x-axis for fourier plot
x_limit=[-1,1]


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


if len(Choose_type)!= len(start) or len(Choose_type)!= len(stop):
    print('please fix: length of input lists does not match!\n')
else:
    pass

if Choose_type[0]=='square':
    pulse_length_ns=pulse_length_ns*2
else:
    pass


t=np.linspace(0,pulse_length_ns*1e-9,N_points)


if Analyze_index+1>len(Choose_type):
    print('Analyze_index larger than number of pulses\n')
    Analyze_index=0
else:
    pass



freq_out_list=[]
amp_out_list=[]

freq_list=[]
amp_list=[]
phase_list=[]
B1x_list=[]
B1y_list=[]

mod_depth_list=[]



def FUNC(start,stop,RF,Choose_type):


    #check how this works
    start_stop_diff_list=[np.round(abs(start-RF),3),np.round(abs(stop-RF),3)]
    mod_depth_list.append(max(start_stop_diff_list))
    
    start=start-RF 
    stop=stop-RF
    
    
    vmax=(stop-start)/2
    
    v_off=vmax +start
    
    vmax=vmax*1e9
    v_off=v_off*1e9
    
    #length_t=len(t)
        
    B1max=1
    phi_max=1
    
    T=t[-1]-t[0]
    
    if Choose_type=='hsec':
        B=5.3
        def B_func(t):
            return B1max*1/(np.cosh(B* (1 - 2*t/T)))
        def v_func(t):
            #return v_off+vmax*np.tanh(B* (1 - 2*t/T))
            return (v_off+vmax*np.tanh(B* (1 - 2*t/T)))
    elif  Choose_type=='wurst':
        N=100
        def B_func(t):
            return B1max*(1-(abs(np.cos(np.pi*t/T)))**N)
        def v_func(t):
            return v_off+vmax-(2*vmax*t/T)
    elif Choose_type=='square':
        def B_func(t):
            y=t.copy()
            for i in range(len(t)):
                if t[i]>pulse_length_ns*1e-9/4 and t[i]<pulse_length_ns*1e-9*3/4:
                    y[i] = 1
                else:
                    y[i] = 0.0000001
            return y          
        def v_func(t):
            return start*1e9+t*0      
        
    elif Choose_type=='gaussian':
        def B_func(t):
            sigma=t[-1]/6.2 #6.2
            mean=t[-1]/2
            return B1max * np.exp(-np.square(t - mean) / (2 * sigma ** 2))
        def v_func(t):
            return start*1e9+t*0
        
    else:
        pass
            
    freq_out=v_func(t)
    freq_out=freq_out/max(abs(freq_out))
    
    amp_out=B_func(t)
    
    freq_out_list.append(freq_out)
    amp_out_list.append(amp_out)
    
    amp_list.append(B_func(t))
    freq_list.append(v_func(t))
    
    phase_array=[]
    for i in range(0,len(t)):
        phase_array.append(phi_max*2*np.pi*quad(v_func, 0, t[i])[0])
    phase_array=np.array(phase_array)#+90*np.pi/180
        
    phase_list.append(phase_array)
    
    B1x_list.append(B_func(t)*np.cos(phase_array))
    B1y_list.append(B_func(t)*np.sin(phase_array))


for i in range(0,len(Choose_type)):
    #pulse_index.append(i)
    FUNC(start[i],stop[i],RF,Choose_type[i])
    print("Mod_depth", str(i), '=',mod_depth_list[i])


font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=4,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=5,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=6,figsize=(3.25,2.25), dpi=300)


plt.figure(1)
plt.plot(t*1e9,amp_list[Analyze_index])
plt.ylabel('MW Amplitude')
plt.xlabel("Time (ns)")

plt.figure(2)
plt.plot(t*1e9,freq_list[Analyze_index]*1e-9)
plt.ylabel('Frequency (GHz)')
plt.xlabel("Time (ns)")



plt.figure(3)
plt.plot(t*1e9,phase_list[Analyze_index])
plt.ylabel('Phase')
plt.xlabel("Time (ns)")





#B1x_array=B_func(t)*np.cos(phase_array)
#B1y_array=B_func(t)*np.sin(phase_array)


plt.figure(4)
plt.plot(t*1e9,B1x_list[Analyze_index])
plt.plot(t*1e9,B1y_list[Analyze_index])
plt.ylabel('MW Amplitude')
plt.xlabel("Time (ns)")






###############################################################################

yf_list=[]
xf_list=[]
times_list=[]
re_list=[]
im_list=[]

    
def fourier_func(Col): 
    times=t
    data_im=B1y_list[Col]#[:,2*Col+1]
    data_re=B1x_list[Col]#[:,2*Col+2]
    
    re_list.append(data_re)
    im_list.append(data_im)
    times_list.append(times)
    
    #T=sampling*1e-12
    T=t[1]-t[0]
    
    '''
    cutoff_ind=[0,5000]
    times=times[cutoff_ind[0]:cutoff_ind[1]]
    data_im=data_im[cutoff_ind[0]:cutoff_ind[1]]
    data_re=data_re[cutoff_ind[0]:cutoff_ind[1]]
    '''
    
    
    

    data = [data_re[i]+ 1j* data_im[i] for i in range(len(data_im)) ]
    

    
    
    
    #data=data_re
    #data=np.imag(data_re)
            
    from scipy.fft import fft, fftfreq
    # Number of sample points
    N = len(times)
    
    #x = times
    #y =data-np.mean(data)
    y=data
    yf = fft(y)

    xf = fftfreq(N, T)#[:N//2]

    ############################

    yf = np.fft.fftshift(yf)
    xf = np.fft.fftshift(xf)

    ###########################
    yf=np.abs(yf)
    
    savGol='fon'
    from scipy.signal import savgol_filter
    if savGol=='on':
        yf=savgol_filter(yf, 13, 2)
    
    ###########################################################################
    xf_list.append(xf)
    yf_list.append(yf)
    

fourier_func(Analyze_index)

plt.figure(5)
plt.plot(xf_list[0]*1e-9, yf_list[0], '-b')
plt.title("Fourier transform")
plt.xlabel("GHz")
plt.ylabel('intensity')
plt.xlim(x_limit)

###############################################################################




#print(*freq_out,sep=',')
plt.figure(6)
plt.title('file out')
#plt.plot(mod_depth_list[Analyze_index]*freq_out_list[Analyze_index],label='freq')
plt.plot(freq_out_list[Analyze_index],label='freq')
plt.plot(amp_out_list[Analyze_index],label='Amp')
plt.legend()
plt.xlabel('N')
plt.ylabel('Value')

###############################################################################






if len(Choose_type)>1:
    filenameout=''
    for i in range(0,len(Choose_type)):
        filenameout=filenameout+Choose_type[i]+'_'
    filenameout=filenameout+'.ptn'
else:
    filenameout=Choose_type[0]+'.ptn'


if save_file=='yes':
    filenameout1="".join(["\\",filenameout])
    filenameoutcomby=directory_data_out+filenameout1
    
    f = open(filenameoutcomby, "w")
    

    for i in range(0,len(freq_out_list[0])):
        for j in range(0,len(Choose_type)):
            f.write("%f %f "%(amp_out_list[j][i],freq_out_list[j][i]))
        f.write('\n')
    f.close()
else:
    pass

print('\n',filenameout,'\n')
#print(out_table)


#print(B1x_array+1j*B1y_array-B_func(t)*np.exp(1j*phase_array))






