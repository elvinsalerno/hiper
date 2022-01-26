# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:37:15 2022

@author: hiper
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import quad





Choose_type=['hsec']
START=[94.3] #in GHz
STOP=[94.6]  #in GHz

#from oscillator*12+1.8
RF=[94.]

#choose 'up' or 'down'
CHIRP_DIRECTION=['up']


N_points=1000


N=100








t=np.linspace(0,100e-9,1000)



vmax=0.3e9



v_off=-0.3e9


###############################################################################
###############################################################################
###############################################################################
B1max=1
phi_max=1
B=5.3

if Choose_type[0]=='hsec':
    def B_func(t):
        return B1max*1/(np.cosh(B* (1 - 2*t/T)))
    def v_func(t):
        return vmax*np.tanh(B* (1 - 2*t/T))

elif  Choose_type[0]=='wurst':
    def B_func(t):
        return B1max*(1-(abs(np.cos(np.pi*t/T)))**N)
    def v_func(t):
        return v_off+vmax-(2*vmax*t/T)
else:
    pass

#START-RF+1.8



T=t[-1]-t[0]



font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)
fig = plt.figure(num=4,figsize=(3.25,2.25), dpi=300)


plt.figure(1)
plt.plot(t,B_func(t))
plt.ylabel('MW Amplitude')
plt.xlabel("Time")

plt.figure(2)
plt.plot(t,v_func(t))
plt.ylabel('Frequency')
plt.xlabel("Time")

phase_array=[]
for i in range(0,len(t)):
    phase_array.append(phi_max*2*np.pi*quad(v_func, 0, t[i])[0])
phase_array=np.array(phase_array)#+90*np.pi/180

plt.figure(3)
plt.plot(t,phase_array)
plt.ylabel('Phase')
plt.xlabel("Time")





B1x_array=B_func(t)*np.cos(phase_array)
B1y_array=B_func(t)*np.sin(phase_array)


plt.figure(4)
plt.plot(t,B1x_array)
plt.plot(t,B1y_array)
plt.ylabel('MW Amplitude')
plt.xlabel("Time")



###############################################################################


directory_data_out=r'C:\SpecMan4EPR\patterns'

#write 'yes' or 'no'
save_file='yes'





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
    
    #for i in range(0,len(RF)):
    for i in range(0,len(B1x_array)):
        for j in range(0,len(RF)):
            f.write(" %f "%(B1x_array[i]))
        f.write('\n')
    f.close()
else:
    pass

print(filenameout,'\n')
#print(out_table)























