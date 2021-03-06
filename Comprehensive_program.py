

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:44:33 2021

@author: evsal
"""


input_directory=r'C:\SpecManData\Elvin\MG_VIII_530_Gd3_1percent\Round4'

#input_filename='ELDOR_MGVIII530_det94.47GHz_10dB_3.400T_2'
#input_filename='T1_10dB_1.60T_traces_longer_linearScale'
input_filename = 'sweep_Freq_93.5to94.5_ctr94._20dB_3.381T_dontvaryref_pumpGauss93.77'


#write in terms of ns
corrections=[[350,450]]
integ_limit=[200,300]
#MF = 0  # ;)

#choose "on' or 'off'
plot_browser='ofn'


x_axis_label='sat pulse (ns)'
#x_axis_label='Delay after first pulse (us)'
#x_axis_label='sat pulse length (ns)'

#choose 'yes' or 'no'
normalize_data_out='yes'

#Choose 'yes' or 'no'
print_data='no'

#xlim=[93.55,93.65]
xlim=None#[93.5,94.5]

###############################################################################

#put 0 and sometyhing really large uinless you want to change
#The range for the x-axis, given in terms of n traces to skip
start_trace=2
end_trace=120

#skip these points for the x-axis, given in terms of x-axis
#choose 'on' and give limits or 'off'
skip_traces='offn'
skip_trace_start=6.5
skip_trace_end=11.5

#value added to numbers on x-axis
f_add= 94.-1.8
###############################################################################

#If fitting t1 or t2


#choose 't1' or 't2' or 'no' or '2t1' or '2t2'
fit_data='no'



#guess for the fitting program
#[amplitude, lifetime (ns/ms/us), constant]
#for one process, write as [A1, k1, c] 
#for two processes, write as [A1,lifetime_1,c,A2,lifetime2]
#or just leave as guess=[]
guess=[]


#if x-scale doesnt change, leave as 'ns'
#choose 'ns' or 'us' or 'ms' ###or "CPMG"
timescale='ns'


###############################################################################

EIK_out_corr='yes'


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

def NormalizeData(data):
    return data / (np.max(data)) 

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



print('file\n',filename_in)


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

data=raw_data[:,1:]
time_steps=raw_data[:,0]


#This is trash
#start_time=20
#end_time=4000
#data=raw_data[start_time:end_time,1:]
#time_steps=raw_data[start_time:end_time,0]


data=data.T
time_steps=time_steps.T



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
im_list=[]
re_list=[]

for i in range(0,int((len(data))/2)):
    im=data[:][2*i-1]
    re=data[:][2*i]
    magn=np.sqrt((im**2)+(re**2))
    #magn=im-re
    im_list.append(im)
    re_list.append(re)
    magn_list.append(magn)
    #plt.plot(raw_data.T[0,:],re+1)
    #plt.plot(time_steps,magn.T[:])
    


maxxx2=max(map(max, magn_list))

magn_list=magn_list[start_trace:end_trace]


times_3= times_2[::2]
times_3=np.array(times_3)


colors = cm.hot(np.linspace(0.,0.3, len(magn_list)))
colors = cm.rainbow(np.linspace(0.,1, len(magn_list)))
#magn_list2=magn_list.copy()
for i in range(0,len(magn_list)):
    #magn_list2[i]=magn_list2[i]/maxxx2
    plt.plot(time_steps*1e9,magn_list[i],linewidth=0.5,color=colors[i])
plt.xlabel('time(ns)')
plt.ylabel('magnitude')
plt.title('integration window')

plt.vlines(integ_limit[0],0,maxxx2,color='b')
plt.vlines(integ_limit[1],0,maxxx2,color='b')



import scipy
from scipy import integrate
#plt.xlim(1.25e-7,3.6e-7)
#import scipy.integrate as integrate
integs=[]
integs_re=[]
integs_im=[]



for i in range(0,len(magn_list)):
   xxx=np.sum(magn_list[i][integ_indices[0]:integ_indices[1]])
   xxx_re=np.sum(re_list[i][integ_indices[0]:integ_indices[1]])
   xxx_im=np.sum(im_list[i][integ_indices[0]:integ_indices[1]])
   #xxx=scipy.integrate.simpson(magn_list[i][integ_indices[0]:integ_indices[1]],time_steps[integ_indices[0]:integ_indices[1]])
   integs.append(xxx)
   integs_re.append(xxx_re)
   integs_im.append(xxx_im)







savGol='off'
from scipy.signal import savgol_filter
if savGol=='on':
    integs=savgol_filter(integs, 9, 2)


times_4=f_add+times_3

DDC='no'
tau=250
if DDC=='yes':
    times_4=times_4*2*tau
else:
    pass

#******************************************************************************
#******************************************************************************
#*******************EIK intensity correction********************************




if EIK_out_corr=='yes':
    
    fig = plt.figure(num=64,figsize=(3.25,2.25), dpi=300)
    frequency=[93,93.4,93.45,93.5,93.55,93.6,93.65,93.7,93.75,93.8,93.85,93.9,93.95,94,94.05,94.1,94.15,94.2,94.25,94.3,94.35,94.4,94.45,94.5,94.55,94.72]
    value=[0.1,1.25,2.87,8.25,17.25,26.9,31.9,28.12,22.60,16.60,15.12,15.12,15.12,13.25,13.90,13.90,17.12,17.50,20.40,20.40,21.10,16.90,14.90,8.10,2.85,1.50]
    
    
    frequency=np.array(frequency)
    value=np.array(value)
    
    plt.title('EIK profile')
    plt.plot(frequency,value)
    plt.xlim(93.5,94.5)
    
    """
    def interp_func(x,a,b,c):
        return ((8*b/(a**4))*((a**2)-2*(x**2))*(x**2))+c
    
    #guess=[2666,-2e6,4e8,-4e10,3e12,-1e14,2e15]
    from scipy.optimize import curve_fit



    pars, pcov = curve_fit(interp_func,frequency,value)#,p0=guess,maxfev=100000000,method='dogbox')
    print(pars)
    
    
    plt.plot(frequency,interp_func(frequency,*pars),'r--')
    integs_correction_arr=integs/interp_func(times_4,*pars)
    """
    interp_func=interpolate.interp1d(frequency,value,kind='cubic')
    print(*interp_func(times_4[start_trace:end_trace]),sep=',')
    integs=np.array(integs)
    #print(len(integs[start_trace:end_trace]))
    #print(len(times_4[start_trace:end_trace]))
    integs_correction_arr=integs/interp_func(times_4[start_trace:end_trace])
    
    
    
    fig = plt.figure(num=44,figsize=(3.25,2.25), dpi=300)
    #plt.title('EIK corrected')
    plt.plot(times_4[start_trace:end_trace],integs_correction_arr,'b')
    plt.xlabel ('sat pulse frequency (GHz)')
    plt.ylabel('signal')
    #plt.xlim(93.5,94.5)
    #plt.ylim(0,8)
else:
    pass

#******************************************************************************
#******************************************************************************
#******************************************************************************

plot_im_re='no'

if plot_im_re=='yes':
    fig = plt.figure(num=33,figsize=(3.25,2.25), dpi=300)
    
    plt.plot(times_4[start_trace:end_trace],integs_im,label='im')
    plt.plot(times_4[start_trace:end_trace],integs_re,label='re')
    #plt.xlabel(r'$\tau_{pump}$ (ns)')
    plt.xlabel(x_axis_label)
    plt.legend()
    plt.ylabel('integrated signal')
else:
    pass
"""


fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)

plt.plot(times_4[start_trace:end_trace],NormalizeData(integs))
#plt.xlabel(r'$\tau_{pump}$ (ns)')
plt.xlabel(x_axis_label)
plt.ylabel('integrated signal')
"""
#plt.xlim(93.4,94.5)

if print_data=='yes' or print_data=='on':
    print('\nx-data')
    print(*times_4[start_trace:end_trace],sep=',')
    print('\ny-data')
    
    if normalize_data_out=='yes':
        print(*NormalizeData(integs),sep=',')
        
    else:
         print(*integs,sep=',')
    print('\nmax/min\n'+str(np.max(integs)/np.min(integs)))
else:
    pass

'''
print('im,re')
print(*integs_im,sep=',')
print(*integs_re,sep=',')
'''


'''

for i in range(0,len(filez[0])):
    plot_fcn(filez[0][i],'xxx')




#fig = go.Figure()
#fig.add_trace(go.Scatter(x=field, y=data_mag,
#                    mode='lines',
#                    name='test'))

#plt.legend()
plt.xlabel('Field (T)')
plt.ylabel('echo intensity')
plt.title('94 GHz')
plt.legend()
#plt.xlim(3.25,3.4)


'''
#plt.xlim(350,500)


"""
###############################################################################
###############################################################################
###############################################################################
x_limit=(0,300e-7)
LIMIT1=x_limit[0]
LIMIT2=x_limit[1]
space=np.linspace(LIMIT1,LIMIT2,1000)


from scipy.optimize import curve_fit

def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

guess = [
1e-7,1,0.1e-7,
]




x_limit=(0,200)
space=np.linspace(LIMIT1,LIMIT2,1000)
fig = plt.figure(num=4,figsize=(3.25,2.25), dpi=300)
gauss_integ_list=[]

for i in range(0,len(magn_list)):
    x, y = time_steps,magn_list[i][:]

    #plt.plot(x,y)
    pars, pcov = curve_fit(func, x, y, p0=guess,maxfev=38000)
    #pars_list.append(pars)
    plt.plot(x,func(x,*pars))
    
    zzz=np.sum(func(x,*pars))
    gauss_integ_list.append(zzz)
plt.title('Gaussian Fit to Echo')
fig = plt.figure(num=5,figsize=(3.25,2.25), dpi=300)
plt.plot(times_3[start_trace:end_trace],gauss_integ_list)
plt.title('Integrate Gaussian vs taupump')
#parsmatrix = np.array([pars[x:x+(3)] for x in range(0, len(pars),3)]) 

#print(pars_list)
"""



if timescale=='us':
    times_4=times_4/1000
    if len(guess)==0:
        guess=[1.0,3, 1.5]
elif timescale=='ms':
    times_4=times_4/1000000
    if len(guess)==0:
        guess=[6,0.001,1]
else:
    if len(guess)==0:
        guess=[6,3000,3]




############################DELETE INDEX RANGE##################################

if skip_traces=='on':
    fig = plt.figure(num=44,figsize=(3.25,2.25), dpi=300)

    plt.title('skip traces plot')
    plt.xlabel(x_axis_label)
    plt.ylabel('integrated signal')
    
    
    

    plt.plot(times_4[start_trace:end_trace],(integs),'b',label='data')
    plt.vlines(skip_trace_start,min(integs),max(integs),color='k')
    plt.vlines(skip_trace_end,min(integs),max(integs),color='k')
    
    specwavel1=min(times_4, key=lambda x:abs(x-skip_trace_start))
    specwavel2=min(times_4, key=lambda x:abs(x-skip_trace_end))
    index_del_trace_1, = np.where(times_4 == specwavel1)
    index_del_trace_2, = np.where(times_4 == specwavel2)
    
    integs = np.delete(integs, slice(index_del_trace_1[0], index_del_trace_2[0]), axis=0)
    times_4 = np.delete(times_4, slice(index_del_trace_1[0], index_del_trace_2[0]))
else:
    pass
######################################################################


############################FIT LORENTZIAN ELDOR##################################
'''
subtract_Lorentzian='on'

if subtract_Lorentzian=='on':
    def lorentzian( x, x0, a, gam ):
        return 1-a * gam**2 / ( gam**2 + ( x - x0 )**2)
    
    fig = plt.figure(num=43,figsize=(3.25,2.25), dpi=300)
    plt.plot(times_4[start_trace:end_trace],NormalizeData(integs),'b',label='data')
    plt.title('Lorentzian plot')
    plt.xlabel(x_axis_label)
    plt.ylabel('integrated signal')
    
    from scipy.optimize import curve_fit
    Lor_guess=[94,1,0.001]
    pars, pcov = curve_fit(lorentzian,times_4[start_trace:end_trace],NormalizeData(integs), p0=Lor_guess,bounds=[[93,0,0],[95,np.inf,0.03]],maxfev=1000000)#,bounds=(0,np.inf),maxfev=3800)
    #print(pars)
    #pars_peaks=np.ndarray.tolist(pars_peaks)
    #ax1.plot(peaks,corrfunc(peaks,*pars_peaks))#,'--',linewidth=0.5,color=colors[i,:])
    
    #print(pars)
    
    #pars=[94,1,2]
    plt.plot(times_4[start_trace:end_trace],lorentzian(times_4[start_trace:end_trace],*pars),'r--',label=str(round(pars[1],2))+' '+timescale )
    
    integs=integs+-30*lorentzian(times_4[start_trace:end_trace],*pars)

'''

######################################################################
fig = plt.figure(num=46,figsize=(3.25,2.25), dpi=300)



if normalize_data_out=='yes':
    plt.plot(times_4[start_trace:end_trace],NormalizeData(integs),'b',label='data')
else:
    plt.plot(times_4[start_trace:end_trace],(integs),'b',label='data')

#plt.xlabel(r'$\tau_{pump}$ (ns)')
plt.xlabel(x_axis_label)
plt.ylabel('integrated signal')

plt.xlim(xlim)






#plt.ylim(0,85)

#******************************************************************************
#******************Fit t1 or t2************************************************
#******************************************************************************

if fit_data=='2t1' or fit_data=='2t2':
    if timescale=='us':
        if len(guess)!=5:
            guess=[0.6,3, 1.5, 0.4, 2]
    elif timescale=='ms':
        if len(guess)!=5:
            guess=[0.6,0.001,1,0.4,0.002 ]
    else:
        if len(guess)!=5:
            guess=[0.6,3000,3, 0.4, 2000]


print('\nguess=',guess)


if fit_data=='t2':
    def expon(t,N0,k,c):#,c,d):
        return N0*np.exp(-t/k)+c
elif fit_data=='t1':
    def expon(t,N0,k, c):#,c,d):
        return -N0*np.exp(-t/k) + c
elif fit_data=='2t1':
    def expon(t,N0_1,k1, c,N0_2,k2):#,c,d):
        return -(N0_1/(N0_1+N0_2))*np.exp(-t/k1) - (N0_2/(N0_1+N0_2))*np.exp(-t/k2) + c
elif fit_data=='2t2':
    def expon(t,N0_1,k1,c,N0_2,k2):#,c,d):
        return (N0_1/(N0_1+N0_2))*np.exp(-t/k1)+(N0_2/(N0_1+N0_2))*np.exp(-t/k2)+c


times_sim=np.linspace(times_4[start_trace:end_trace][0],times_4[start_trace:end_trace][-1],1000)

if fit_data=='t1' or fit_data=='t2' or fit_data=='2t2' or fit_data=='2t1':
    from scipy.optimize import curve_fit
        
    if normalize_data_out=='yes':
        pars, pcov = curve_fit(expon,times_4[start_trace:end_trace],NormalizeData(integs), p0=guess,bounds=(0,np.inf),maxfev=1000000)#,bounds=(0,np.inf),maxfev=3800)
    else:
        pars, pcov = curve_fit(expon,times_4[start_trace:end_trace],(integs), p0=guess,bounds=(0,np.inf),maxfev=1000000)#,bounds=(0,np.inf),maxfev=3800)
   
    
    
    if fit_data=='t1' or fit_data=='t2':
        plt.plot(times_sim,expon(times_sim,*pars),'r--',label=str(round(pars[1],2))+' '+timescale )
        print('\n','N_0 =',pars[0],'\n','rate1 =',pars[1]**-1,timescale,'^-1' ,'\n' , 'lifetime =',pars[1],timescale,'\n','C =',pars[2])
        
    elif fit_data=='2t2' or fit_data=='2t1':
        plt.plot(times_sim,expon(times_sim,*pars),'r--',label=str(round(pars[1],2))+', '+str(round(pars[4],2))+' '+timescale )
        print('\n','N0_1 =',pars[0],'(',pars[0]/(pars[0]+pars[4]),')','\n','rate_1 =',pars[1]**-1,timescale,'^-1' ,'\n' , 'lifetime_1 =',pars[1],timescale,'\n','*******************','\n','N0_2 =',pars[4],'(',pars[4]/(pars[0]+pars[4]),')','\n','rate_2 =',pars[4]**-1,timescale,'^-1','\n','lifetime_2 =',pars[4],timescale,'\n','*****************','\n','C =',pars[2])
    else:
        pass
    plt.legend()
    plt.xlabel('time ('+timescale+')')
    plt.ylabel('echo intensity')
else:
    pass



#******************************************************************************
#*******************************PLOTLY*****************************************
#******************************************************************************

if plot_browser=='on':
    
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default='browser'
    
    fig = go.Figure()
    
    
    
    
    fig.add_trace(go.Scatter(x=f_add+times_3[start_trace:end_trace], y=integs,
                mode='lines',
                name='xxx'))
    fig.update_xaxes(title=x_axis_label)
    
    
    
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

fourier_transform = 'offn'
if fourier_transform == 'on':
    
    integs = integs - np.mean(integs) # to remove zero frequency signal
    signal = np.fft.fft(NormalizeData(integs))
    freq = np.fft.fftfreq((times_4[start_trace:end_trace]).shape[-1])
    
    fig = plt.figure(num=72,figsize=(3.25,2.25), dpi=300)

    plt.plot(freq, signal.real)
    plt.xlim(0,0.2)
   # plt.ylim(0,15)


