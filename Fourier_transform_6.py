# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 18:09:46 2021

@author: hiper
"""








input_directory=r"C:\Users\esalerno\Google Drive\MagLab\Jon_fft_data"
input_filename="Jon_off_combined"

input_directory=r'C:\SpecManData\Elvin\CHORUS_BDPA_12132021'
input_filename='BDPA_CHORUS_10dB_center94.2_chirp93.55to94.65_1'

input_directory=r'C:\SpecManData\AWG_arbpulse'
input_filename='Boldenhausen_Fourier_det_obs93.5to94.5GHz_15dB_3.67T_center94.0Ghz_Amp0.4_varypumpFreq'
input_filename='xxx'


skip_n_points=0
#n2=2500

#Col=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
Col=range(4,5)
Col=[0]

plot_browser='ofn'



echo_x_lim=None
#echo_x_lim=[1e-6,1.6e-6]

xlim=[93.5,93.88]

xlim=[93.5,94]
xlim=None
#xlim=[94,94.5]


#give in terms of us
cutoff_ind=[1000,1300]





f_add=0.

#time between samples, in picoseconds
sampling=500

times_corr=300            #-326



print_data='offn'
#Col=[20]
#Col=[]

#t_max_index=6000

normalize_on='off'


#make zero point the average of two adjacent points
fix_zero='fon'
fix_zero_point=f_add


################################
#print list of x-axis from Specman
experiment_axis='no' 
exp_unit=' GHz'
length=31
start=94
step=0.02





###############################################################################
###############################################################################
###############################################################################

#cutoff_ind=cutoff_ind*100/sampling






import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np




filetype='.txt'

filename_in="".join(["\\",input_filename])
filename_comby=input_directory+filename_in+filetype

print(filename_comby[0:-4])




if experiment_axis=='yes':
    x_axis=np.arange(0,length)*step+start
    print('\n',x_axis)
    #print(np.round(x_axis[Col[-1]],2))



font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)
raw_data = np.genfromtxt(filename_comby,skip_header=1)#,delimiter=',')

yf_list=[]
xf_list=[]
times_list=[]
re_list=[]
im_list=[]
cut_re_list=[]
cut_im_list=[]
cut_times_list=[]

theta_list=[]
#from scipy import interpolate
    
def fourier_func(Col): 
    times=raw_data[:,0]
    data_im=raw_data[:,2*Col+1]
    data_re=raw_data[:,2*Col+2]
    
    
    
    if fix_zero=='on':
        ##############To add delay between channels###################
        #data_re=data_re[:-1]
        #data_im=data_im[1:]
        #times=times[:-1]
        
        data_re_interp=np.append(data_re,np.array([0]))
        data_re_interp=np.insert(data_re_interp,0,np.array([0]))
        
        times_interp=np.append(times,np.array([times[-1]+3*(sampling*1e-12)]))
        times_interp=np.insert(times_interp,0,np.array([times[0]-3*(sampling*1e-12)]))
           
        interp_func=interpolate.interp1d(times_interp,data_re_interp,kind='cubic')
          
        
        data_re=interp_func(times-times_corr*1e-12)
        
        
        ##################################
    else:
        pass

    
    
    
    
    
    #data_re=[0]*len(data_re)
     
    #data_re2=data_re.tolist()
    #print(data_re2,sep=',')
    
    
    re_list.append(data_re)
    im_list.append(data_im)
    times_list.append(times)
    
    times=times[cutoff_ind[0]:cutoff_ind[1]]
    data_im=data_im[cutoff_ind[0]:cutoff_ind[1]]
    data_re=data_re[cutoff_ind[0]:cutoff_ind[1]]
    
    cut_re_list.append(data_re)
    cut_im_list.append(data_im)
    cut_times_list.append(times)
    
    

    

    
    T=sampling*1e-12
    
    #zero padding
    #times=np.append(times,np.array([times[-1]+T*i for i in range(1,2*len(times)+1)]))
    #data_im=np.append(data_im,np.array([0]*2*len(data_im)))
    #data_re=np.append(data_re,np.array([0]*2*len(data_re)))
    
    #print(len(times),len(data_im),len(data_re))

    data = [data_re[i]+ 1j* data_im[i] for i in range(len(data_im)) ]
    #data=np.imag(data_re)
    
        
    from scipy.fft import fft, fftfreq
    # Number of sample points
    N = len(times)
    
    #x = times
    y =data-np.mean(data)
    #y=data
    yf = fft(y)
    
    #print(yf)
    
    
    #from scipy.signal import blackman
    #w = blackman(N)
    #ywf = fft(y*w)
    xf = fftfreq(N, T)#[:N//2]
    
    #N_space=np.arange(-N/2,N/2,1)
    
    
    #xf =(1/T)*N_space/N
    #print(len(xf))
    
    ############################
    
    #fftData = np.fft.fft(data)
    #freq = np.fft.fftfreq(lenData, 1/fSamp)
    yf = np.fft.fftshift(yf)
    xf = np.fft.fftshift(xf)
    
    
    #yf=(np.imag(yf))
    ###########################
    
    
    xf=xf/1e9+f_add
    
    
    
    ######################
    theta=np.arctan(np.imag(yf)/np.real(yf))
    #theta=np.arctan2(np.imag(yf),np.real(yf))
    '''
    plt.figure(100)
    plt.plot(xf,np.imag(yf)/np.real(yf))
    plt.ylim(-100,100)
    plt.plot(xf,np.real(yf))
    plt.plot(xf,np.imag(yf))
    plt.xlim(93.95,94.05)
    '''
    ######################
    
    #yf=np.abs(yf)
    yf=(np.sqrt(np.real(yf)**2+np.imag(yf)**2))
    #yf=yf
        
    savGol='fon'
    from scipy.signal import savgol_filter
    if savGol=='on':
        yf=savgol_filter(yf, 13, 2)
    
    

    
    '''
    
    ###########################################################################
    fig = plt.figure(num=64,figsize=(3.25,2.25), dpi=300)
    frequency=[93,93.4,93.45,93.5,93.55,93.6,93.65,93.7,93.75,93.8,93.85,93.9,93.95,94,94.05,94.1,94.15,94.2,94.25,94.3,94.35,94.4,94.45,94.5,94.55,94.6,95.2]
    value=[0.1,1.25,2.87,8.25,17.25,26.9,31.9,28.12,22.60,16.60,15.12,15.12,15.12,13.25,13.90,13.90,17.12,17.50,20.40,20.40,21.10,16.90,14.90,8.10,2.85,1.50,0.1]
    
    
    frequency=np.array(frequency)
    value=np.array(value)
    
    plt.title('EIK profile')
    plt.plot(frequency,value)
    plt.xlim(93.5,94.5)
    

    interp_func=interpolate.interp1d(frequency,value,kind='cubic')
    #yf=np.array(yf)
    yf=yf/interp_func(xf)
    '''


    
    ###########################################################################
    xf_list.append(xf)
    yf_list.append(yf)
    theta_list.append(theta)
    
    #print(data_re,sep=',')
    #print(data_im,sep=',')
    #print(times,sep=',')




    
    
    
    

'''
plt.xlim(-0.3,0.3)

#print(xf)

plt.plot(xf, yf, '-b')
#plt.xlim(70,110)
#plt.xlim(-.2,.2)
#plt.ylim(0,6)
plt.xlabel("GHz")
'''


for i in range(0,len(Col)):
    fourier_func(Col[i])

#print(times_list[0],sep=',')
'''
print('\n',im_list)
print('\n',re_list)
'''


###################################


plt.figure(1)


plt.plot(times_list[0],im_list[-1],label='last_trace_re')
plt.plot(times_list[0],re_list[-1],label='last_trace_im')
plt.legend()
plt.xlim(echo_x_lim)
plt.xlabel('time')
plt.ylabel('intensity')
#plt.show()
#plt.xlim(0,5e-7)



plt.vlines(times_list[0][cutoff_ind[0]],min(im_list[-1]),max(re_list[-1]),'k',zorder=10)
plt.vlines(times_list[0][cutoff_ind[1]],min(im_list[-1]),max(re_list[-1]),'k',zorder=9)
#plt.xlim(0,0.25e-6)





plt.figure(3)
plt.plot(cut_times_list[0],cut_re_list[0])
plt.plot(cut_times_list[0],cut_im_list[0])
##################################


xf_list=np.array(xf_list)
xf_list=np.mean(xf_list,axis=0)

yf_list=np.array(yf_list)
yf_list=np.mean(yf_list,axis=0)

theta_list=np.array(theta_list)
theta_list=np.mean(theta_list,axis=0)


#yf_list=yf_list[20:-20]
#xf_list=xf_list[20:-20]


#print(np.shape(yf_list))


if normalize_on=='on':
    yf_list=yf_list/max(yf_list)
else:
    pass


# remove 94GHz frequency
'''
idx = np.where(xf_list == 94.0)
xf_list = xf_list[xf_list  != 94.0]
yf_list = np.delete(yf_list , idx)

'''
# replace zero frequency point by the average
if fix_zero=='on':
    idx = np.where(xf_list == fix_zero_point)[0][0]
    avg_point = (yf_list[idx - 1] + yf_list[idx +1])/2
    yf_list = np.where(xf_list == fix_zero_point , avg_point , yf_list)
    
    idx_theta = np.where(xf_list == fix_zero_point)[0][0]
    avg_theta=(theta_list[idx_theta - 1] + theta_list[idx_theta +1])/2
    theta_list=np.where(xf_list == fix_zero_point , avg_theta , theta_list)
    
else:
    pass


#deal with angle between im and re


fig = plt.figure(num=123,figsize=(3.25,2.25), dpi=300)
plt.figure(123)
plt.plot(xf_list,abs(theta_list*180/np.pi))
#plt.xlim(93.9,94.1)
plt.xlim(xlim)
plt.xlabel('GHz')
plt.ylabel('theta (deg)')




#print((theta_list)*(180/np.pi))








#plt.ylim(0,3)
plt.figure(2)
plt.plot(xf_list, yf_list, '-b')
#plt.xlim(70,110)
#plt.xlim(-.2,.2)
#plt.ylim(0,6)
plt.xlabel("GHz")
plt.ylabel('intensity')
plt.xlim(xlim)
if experiment_axis=='yes':
    plt.title(str(np.round(x_axis[Col[-1]],2))+exp_unit)
else:
    pass

###########################################################
'''
fig = plt.figure(num=99,figsize=(3.25,2.25), dpi=300)
plt.figure(99)
plt.plot(xf_list, np.imag(yf_list), '-b')
plt.xlabel("GHz")
plt.ylabel('intensity')
plt.xlim(xlim)
'''
###########################################################

#plt.xlim(94,94.2)


if print_data=='on':
    print('\nxf\n')
    print(*xf_list,sep=',')
    print('\n')
    print('yf\n')
    print(*yf_list,sep=',')
else:
    pass

sum_y=np.sum(yf_list)

#plt.ylim(0,3)
#print(sum_y)
#plt.xlim(93.5,94.5)


#plt.ylim(0,0.4)



x_axis_label='GHz'

if plot_browser=='on':
    
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default='browser'
    
    fig = go.Figure()
    
    
    
    
    fig.add_trace(go.Scatter(x=xf_list, y=yf_list,
                mode='lines',
                name='xxx') )
    fig.update_xaxes(title=x_axis_label)
    
    fig.update_layout(title=str(f_add)+'GHz')
    
    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=field, y=data_mag,
                        mode='lines',
                        name='test'))
    '''
    fig.update_yaxes(title='magnitude')
    
    fig.show()
else:
    pass

#*****************************************************************************








'''
print(np.shape(cut_im_list))
print(np.shape(cut_re_list))
print(np.shape(cut_times_list))
'''

"""

re_step_1=cut_re_list[0]-cut_re_list[1]
im_step_1=cut_im_list[0]-cut_im_list[1]

re_step_2=im_step_1+cut_re_list[2]+cut_re_list[3]
im_step_2=re_step_1-cut_im_list[2]-cut_im_list[3]


plt.figure(96)
plt.plot(cut_times_list[0],im_step_2)






def fourier_func_2(real,imag,times): 
    times=times
    data_im=imag
    data_re=real
    
    

    

    
    T=times[1]-times[0]
    


    data = [data_re[i]+ 1j* data_im[i] for i in range(len(data_im)) ]
    #data=np.imag(data_re)
    
        
    from scipy.fft import fft, fftfreq
    # Number of sample points
    N = len(times)
    
    #x = times
    y =data-np.mean(data)
    #y=data
    yf = fft(y)

    xf = fftfreq(N, T)#[:N//2]
    
    #N_space=np.arange(-N/2,N/2,1)
    
    
    #xf =(1/T)*N_space/N
    #print(len(xf))
    
    ############################
    
    #fftData = np.fft.fft(data)
    #freq = np.fft.fftfreq(lenData, 1/fSamp)
    yf = np.fft.fftshift(yf)
    xf = np.fft.fftshift(xf)
    
    
    #yf=(np.imag(yf))
    ###########################
    
        
    yf=np.abs(yf)
    
    savGol='fon'
    from scipy.signal import savgol_filter
    if savGol=='on':
        yf=savgol_filter(yf, 13, 2)
    
    
    xf=xf/1e9+f_add
    plt.figure(99)
    plt.plot(xf,yf)
    
    
fourier_func_2(re_step_2,im_step_2,cut_times_list[0])

"""