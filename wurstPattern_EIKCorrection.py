
"""
Created on Sun Nov 21 18:00:14 2021

@author: hiper
"""






import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

RF=[94.00]

Choose_type=['wurst']
START=[93.55] #in GHz
STOP=[94.4]  #in GHz

size = 171
#%%
file_directory=r"C:\SpecManData\Manoj\01192022"
#filename_in="EIK_Output_both_PMYTO_ChB_terminate_RepRate_200us_4dB"
filename_in="30dB_ctr94GHz_93.55to94.40_3"



filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.dat'

raw_data = np.genfromtxt(filename_comby,skip_header=1)#,delimiter=' ')


data_freq=raw_data[:,0] + ( RF[0] - 1.8)
data_real=raw_data[:,1]
data_imag = raw_data[:,2]
plt.figure(22)
plt.plot(data_freq,data_real*1000 , 'b' , label = 'EIK_output')
#plt.plot(data_freq,data_imag*1000 , 'grey' , label = 'EIK_input')
plt.xlim(93.55,94.4)
#plt.ylim(10,20)
plt.legend()
data_gain = data_real / data_imag

#%%

interp_func_output=interpolate.interp1d(data_freq,data_real,kind='cubic' , fill_value="extrapolate")

fnmr = np.linspace(START[0],STOP[0], size)
print(fnmr)
print(len(fnmr))
eik_output =[interp_func_output(item) for item in fnmr]

fig = plt.figure(num=65,figsize=(3.25,2.25), dpi=300)
#plt.plot(fnmr,eik_output,'b')
#plt.plot(fnmr,eik_input,'grey')
#plt.xlim(93.4,94.6)

#%%

amplitude_output = [ 1/ item for item in eik_output] 
amplitude_output = np.round_(amplitude_output/np.max(amplitude_output) , 3)
plt.plot(fnmr, amplitude_output,'darkgreen')

plt.plot(fnmr, amplitude_output,'g', label = 'amplitude')
print('amplitude_output : ')
print(*amplitude_output, sep=',')
plt.legend()

#%%
'''
amplitude = [ 1/ item for item in eik_output] 

fig = plt.figure(num=66,figsize=(3.25,2.25), dpi=300)
amplitude = np.round_(amplitude/np.max(amplitude) , 3)

print('amplitude : ')
print(*amplitude, sep=',')
plt.plot(fnmr, amplitude,'grey')
print(len(amplitude))
'''
#%%

amplitude_corr = amplitude_gain * amplitude_input

amplitude_corr = amplitude_corr/np.max(amplitude_corr)

plt.plot(fnmr, amplitude_corr,'g')
print('amplitude : ')
print(*amplitude_corr, sep=',')

#%%

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



#from oscillator*12+1.8


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

#x=np.arange(0,np.pi,0.0001)
x=np.linspace(0,np.pi,N_points)
print(len(x),'points')


correction = np.array([0.99,0.971,0.941,0.913,0.89,0.867,0.842,0.822,0.799,0.781,0.761,0.747,0.731,0.718,0.704,0.689,0.681,0.673,0.664,0.656,0.648,0.646,0.64,0.637,0.63,0.63,0.625,0.623,0.621,0.619,0.618,0.613,0.609,0.605,0.6,0.594,0.592,0.587,0.582,0.576,0.572,0.563,0.558,0.553,0.546,0.541,0.534,0.527,0.523,0.52,0.515,0.511,0.508,0.507,0.504,0.504,0.504,0.504,0.506,0.507,0.509,0.511,0.514,0.516,0.52,0.526,0.53,0.536,0.54,0.546,0.549,0.555,0.561,0.564,0.567,0.572,0.576,0.579,0.584,0.586,0.589,0.59,0.592,0.593,0.595,0.597,0.598,0.599,0.601,0.603,0.604,0.606,0.61,0.613,0.619,0.622,0.626,0.632,0.639,0.646,0.654,0.662,0.67,0.678,0.689,0.697,0.707,0.717,0.727,0.737,0.747,0.757,0.767,0.776,0.784,0.792,0.798,0.806,0.811,0.818,0.824,0.828,0.832,0.837,0.84,0.845,0.848,0.851,0.855,0.858,0.862,0.865,0.869,0.873,0.877,0.881,0.884,0.888,0.891,0.895,0.899,0.901,0.904,0.908,0.911,0.913,0.916,0.918,0.921,0.923,0.926,0.927,0.927,0.926,0.925,0.924,0.923,0.921,0.918,0.916,0.913,0.91,0.907,0.904,0.901,0.898,0.896,0.893,0.889,0.887,0.883,0.881,0.878,0.878,0.877,0.876,0.874,0.874,0.873,0.873,0.873,0.874,0.875,0.874,0.874,0.875,0.875,0.875,0.876,0.878,0.88,0.88,0.882,0.882,0.882,0.883,0.882,0.884,0.885,0.885,0.885,0.886,0.885,0.886,0.888,0.889,0.89,0.893,0.894,0.897,0.898,0.9,0.904,0.906,0.911,0.915,0.919,0.926,0.93,0.937,0.943,0.95,0.957,0.962,0.969,0.974,0.978,0.982,0.986,0.988,0.991,0.993,0.996,0.998,1.0,0.999,1.0,0.999,1.0,1.0,0.998,0.999,0.997,0.994,0.992,0.99,0.986,0.982,0.981,0.976,0.971,0.968,0.964,0.959,0.955,0.952,0.948,0.944,0.941,0.937,0.936,0.933,0.931,0.927,0.926,0.923,0.922,0.919,0.916,0.916,0.913,0.91,0.908,0.908,0.905,0.901,0.899,0.897,0.895,0.893,0.891,0.887,0.885,0.88,0.877,0.872,0.868,0.865,0.861,0.857,0.852,0.846,0.843,0.838,0.833,0.828,0.824,0.819,0.815,0.812,0.809,0.806,0.802,0.799,0.797,0.795,0.794,0.793,0.792,0.791,0.79,0.791,0.791,0.791,0.795,0.796,0.798,0.8,0.803,0.803,0.805,0.806,0.809,0.808,0.808,0.809,0.805,0.804,0.803,0.801,0.798,0.797,0.794,0.791,0.786,0.78,0.777,0.771,0.766,0.761,0.755,0.749,0.742,0.737,0.731,0.728,0.722,0.719,0.714,0.712,0.71,0.708,0.708,0.706,0.707,0.71,0.712,0.712,0.715,0.719,0.72,0.724,0.724,0.727,0.728,0.727,0.729,0.728,0.73,0.725,0.724,0.72,0.717,0.716,0.711,0.707,0.703,0.699,0.695,0.689,0.685,0.68,0.676,0.671,0.668,0.663,0.659,0.656,0.655,0.652,0.652,0.652,0.654,0.654,0.657,0.662,0.666,0.672,0.678,0.684,0.693,0.7,0.708,0.718,0.728,0.737,0.747,0.754,0.761,0.771,0.779,0.787,0.795,0.803,0.808,0.813,0.817,0.822,0.826,0.833,0.836,0.837,0.843,0.845,0.85,0.852])
freq426 = np.linspace(START[0],STOP[0], 426)


interp_func_correction=interpolate.interp1d(freq426,correction,kind='cubic')

eik_correction =[interp_func_correction(item) for item in fnmr]
eik_correction = eik_correction/np.max(eik_correction)

y_final=[]
z_final=[]
F1_list=[]

def func(START,STOP,RF,CHIRP_DIRECTION,Choose_type, correction):
    ##########################################
    if Choose_type=='wurst':
        N=100
        def y_func(x):
            y = 1-(abs(np.cos(x)))**N
            y = np.multiply(y,correction)
            #y = np.multiply(y,amplitude_corr)
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
    func(START[i],STOP[i],RF[i],CHIRP_DIRECTION[i],Choose_type[i] , eik_correction)
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

#%%