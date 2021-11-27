# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:44:33 2021

@author: evsal
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
#from scipy.optimize import curve_fit
#import io
#import matplotlib.cm as cm



#file_directory=r"C:\SpecManData\Sam\GdDOTA"
#filename_in="T1_20K_VT"

file_directory=r"C:\SpecManData\Gd3NC80Cu\11152021"
filename_in="t1_ELDOR_Gd3@c60_10dB_5K_3.390T_94GHz_1.5GHzsaturation"
skip_n_points=0
n2=150

#choose 'us' or 'ns'
timescale= 'us'




filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.dat'


font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)

raw_data = np.genfromtxt(filename_comby,skip_header=1)#,delimiter=' ')

if timescale=='us':
    time_ns=raw_data[:,0]/1000
else:
    time_ns=raw_data[:,0]
data_Re=raw_data[:,1]
data_Im=raw_data[:,2]

#data_mag=raw_data[:,3]

data_mag=(np.sqrt((np.square(data_Im))+(np.square(data_Re))))
#print(data_mag)
plt.plot(time_ns[skip_n_points:n2],data_mag[skip_n_points:n2],color='b',label='data')#,'o',fillstyle='none',markersize=3
#plt.plot(time_ns,data_Re)
#plt.plot(time_ns,data_Im)

def expon(t,N0,k,c):#,c,d):
    return N0*np.exp(-t/k)+c


from scipy.optimize import curve_fit

guess=[6,1000,1]

pars, pcov = curve_fit(expon,time_ns[skip_n_points:n2],data_mag[skip_n_points:n2], p0=guess,maxfev=100000)#,bounds=(0,np.inf),maxfev=3800)
print(pars)
#pars_peaks=np.ndarray.tolist(pars_peaks)
#ax1.plot(peaks,corrfunc(peaks,*pars_peaks))#,'--',linewidth=0.5,color=colors[i,:])

#print(pars)

plt.plot(time_ns[skip_n_points:n2],expon(time_ns[skip_n_points:n2],*pars),'r--',label=str(round(pars[1],2))+' '+timescale)
#corrected_wavelengths=corrfunc(Crystal_wavelengths,*pars_peaks)
#print(type(corrected_wavelengths))

print(' N_0 =',pars[0],'\n','rate1 =',pars[1]**-1,'ns^-1' ,'\n' , 'lifetime =',pars[1],'\n','C =',pars[2])

plt.legend()
plt.xlabel('time ('+timescale+')')
plt.ylabel('echo intensity')


#print(np.sqrt(np.diag(pcov)))

#print(time_ns)