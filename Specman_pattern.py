# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:00:14 2021

@author: hiper
"""



#Choose 'gaussian' or 'wurst'
'''
Choose_type=['wurst','wurst']
START=[93.2,93.2] #in GHz
STOP=[94.8,93.9]  #in GHz

#from oscillator*12+1.8
RF=[94.,94]

#choose 'up' or 'down'
CHIRP_DIRECTION=['up','up']
'''

Choose_type=['wurst']
START=[92.3] #in GHz
STOP=[92.7]  #in GHz

#from oscillator*12+1.8
RF=[94.]

#choose 'up' or 'down'
CHIRP_DIRECTION=['up']


N_points=10000


#*************************SAVING File****************************************


directory_data_out=r'C:\SpecMan4EPR\patterns'

#write 'yes' or 'no'
save_file='yes'


###############################################################################
###############################################################################
###############################################################################





import numpy as np
import matplotlib.pyplot as plt

#x=np.arange(0,np.pi,0.0001)
x=np.linspace(0,np.pi,N_points)
print(len(x),'points')







y_final=[]
z_final=[]
F1_list=[]

def func(START,STOP,RF,CHIRP_DIRECTION,Choose_type):
    ##########################################
    if Choose_type=='wurst':
        N=100
        def y_func(x):
            y = 1-(abs(np.cos(x)))**N
            return y
            
    elif Choose_type=='gaussian':
        def y_func(x):
            sigma=x[-1]/6.2 #6.2
            mean=x[-1]/2
            scale=1
            y= scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))
            return y
    else:
        pass
 
    ##########################################
    
    y=y_func(x)

    F1=START-RF+1.8  #(this should be written in the specman, for arbpulse frequency) as f1
    F1_list.append(np.round(F1,3))
    #print('F1 is',np.round(F1,3))
    
    F2=STOP-RF+1.8 
    
    
    if Choose_type=='gaussian':
        z = np.array([1]  * (len(x)-1))
        z3 = [1]  * len(x) 
        z5=[1]  * len(x)
    else:
        
        f1=F1/F1
        f2=F2/F1;
        
        STEP=abs(f2-f1)/(len(x)-1)
        
        z=np.arange(f1,f2,STEP)
        
        STEP2=abs(f2-f1)/(len(x))
        z2=np.arange(f1,f2,STEP2)
        z3=z2*F1+RF-1.8
        
        STEP2=abs(f2-f1)/(len(x))
        z4=np.arange(f1,f2,STEP2)
        z5=z4*F1

    #####################
    fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)
    plt.plot(z3,y)
    plt.title('EIK output')
    
    #####################
    fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
    plt.plot(z5,y)
    

    #####################
    #plt.plot(x,y)
    plt.xlabel('Frequency (GHz')
    plt.ylabel('Amplitude')    
    plt.title('AWG output')
    
    if CHIRP_DIRECTION=='down':
        z=z[::-1]
        y=y[::-1]
    else:
        pass
    
    z_final.append(z)
    y_final.append(y)



pulse_index=[]
for i in range(0,len(RF)):
    pulse_index.append(i)
    func(START[i],STOP[i],RF[i],CHIRP_DIRECTION[i],Choose_type[i])
#****************************************
column_names=['start','stop','direction','RF','F']

data_array=[START,STOP,CHIRP_DIRECTION,RF,F1_list]
data_array=np.array(data_array).T.tolist()


import pandas


out_table=pandas.DataFrame(data_array,pulse_index,column_names)






#*****************************************


################################save############################################


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
    for i in range(0,len(z_final[0])):
        for j in range(0,len(RF)):
            f.write("%f %f "%(y_final[j][i],z_final[j][i]))
        f.write('\n')
    f.close()
else:
    pass

print(filenameout,'\n')
print(out_table)