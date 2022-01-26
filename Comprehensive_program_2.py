# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:36:03 2022

@author: hiper
"""



# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:44:33 2021

@author: evsal
"""


input_directory=r'C:\SpecManData\Elvin\MG_VIII_530_Gd3_1percent\Round4\12202022\hole_burning_definitive'
input_filename='varyPumplength_94.16GHzpumpGaussian_obs93.74GHz_ctr94GHz_3.67T_15dB_longer'

#write in terms of ns
corrections=[[500,1000]]
integ_limit=[150,400]


#choose "on' or 'off'
plot_browser='on'


x_axis_label='Pump pulse length (ns)'
#x_axis_label='Delay after first pulse (us)'
#x_axis_label='sat pulse length (ns)'

#choose 'yes' or 'no'
normalize_data_out='no'

#Choose 'yes' or 'no'
print_data='yes'

#xlim=[93.55,93.65]
xlim=None#[0,7]
xlim=[0,1000]

###############################################################################

#put 0 and sometyhing really large uinless you want to change
#The range for the x-axis, given in terms of n traces to skip
start_trace=1
end_trace=300

#skip these points for the x-axis, given in terms of x-axis
#choose 'on' and give limits or 'off'
skip_traces='offn'
skip_trace_start=6.5
skip_trace_end=11.5

#value added to numbers on x-axis
f_add= 0#93.74-1.8
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
#Correct the data with the EIK profile? Must ensure that the traces used fall
#within ~93.5 to 94.5. Adjust with start_trace and end_trace. otherwise they 
#will be outside of the collected range of EIK data and the program will fail:
#"Value error: A value in x_new is above(below) the interpolation range."
#Adjust the weight factor manually until the EIK corr trace approximately 
#matches  background of the raw data

EIK_out_corr='no'
EIK_weight_factor=-1000

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



print('file\n',filename_in[0:-4])


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

for i in range(0,len(corrections_indices)):
    if corrections_indices[i][0]==corrections_indices[i][1]:
        print("\nERROR corrections out of range. make smaller\n")
    else:
        pass


    
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
    





times_3= times_2[::2]
times_3=np.array(times_3)
times_4=times_3+f_add


if EIK_out_corr=='yes':
    spec_x_data_1=min(times_4, key=lambda x:abs(x-93.5))
    spec_x_data_2=min(times_4, key=lambda x:abs(x-94.5))
    index_trace_1, = np.where(times_4 == spec_x_data_1)
    index_trace_2, = np.where(times_4 == spec_x_data_2)
    
    start_trace=index_trace_1[0]
    end_trace=index_trace_2[0]







colors = cm.hot(np.linspace(0.,0.3, len(magn_list)))
colors = cm.rainbow(np.linspace(0.,1, len(magn_list)))
#magn_list2=magn_list.copy()
for i in range(0,len(magn_list)):
    #magn_list2[i]=magn_list2[i]/maxxx2
    plt.plot(time_steps*1e9,magn_list[i],linewidth=0.5,color=colors[i])
plt.xlabel('time(ns)')
plt.ylabel('magnitude')
plt.title('integration window')

maxxx2=max(map(max, magn_list))
magn_list=magn_list[start_trace:end_trace]


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
    
    
    
    frequency=[93.4,93.402,93.404,93.406,93.408,93.41,93.412,93.414,93.416,93.418,93.42,93.422,93.424,93.426,93.428,93.43,93.432,93.434,93.436,93.438,93.44,93.442,93.444,93.446,93.448,93.45,93.452,93.454,93.456,93.458,93.46,93.462,93.464,93.466,93.468,93.47,93.472,93.474,93.476,93.478,93.48,93.482,93.484,93.486,93.488,93.49,93.492,93.494,93.496,93.498,93.5,93.502,93.504,93.506,93.508,93.51,93.512,93.514,93.516,93.518,93.52,93.522,93.524,93.526,93.528,93.53,93.532,93.534,93.536,93.538,93.54,93.542,93.544,93.546,93.548,93.55,93.552,93.554,93.556,93.558,93.56,93.562,93.564,93.566,93.568,93.57,93.572,93.574,93.576,93.578,93.58,93.582,93.584,93.586,93.588,93.59,93.592,93.594,93.596,93.598,93.6,93.602,93.604,93.606,93.608,93.61,93.612,93.614,93.616,93.618,93.62,93.622,93.624,93.626,93.628,93.63,93.632,93.634,93.636,93.638,93.64,93.642,93.644,93.646,93.648,93.65,93.652,93.654,93.656,93.658,93.66,93.662,93.664,93.666,93.668,93.67,93.672,93.674,93.676,93.678,93.68,93.682,93.684,93.686,93.688,93.69,93.692,93.694,93.696,93.698,93.7,93.702,93.704,93.706,93.708,93.71,93.712,93.714,93.716,93.718,93.72,93.722,93.724,93.726,93.728,93.73,93.732,93.734,93.736,93.738,93.74,93.742,93.744,93.746,93.748,93.75,93.752,93.754,93.756,93.758,93.76,93.762,93.764,93.766,93.768,93.77,93.772,93.774,93.776,93.778,93.78,93.782,93.784,93.786,93.788,93.79,93.792,93.794,93.796,93.798,93.8,93.802,93.804,93.806,93.808,93.81,93.812,93.814,93.816,93.818,93.82,93.822,93.824,93.826,93.828,93.83,93.832,93.834,93.836,93.838,93.84,93.842,93.844,93.846,93.848,93.85,93.852,93.854,93.856,93.858,93.86,93.862,93.864,93.866,93.868,93.87,93.872,93.874,93.876,93.878,93.88,93.882,93.884,93.886,93.888,93.89,93.892,93.894,93.896,93.898,93.9,93.902,93.904,93.906,93.908,93.91,93.912,93.914,93.916,93.918,93.92,93.922,93.924,93.926,93.928,93.93,93.932,93.934,93.936,93.938,93.94,93.942,93.944,93.946,93.948,93.95,93.952,93.954,93.956,93.958,93.96,93.962,93.964,93.966,93.968,93.97,93.972,93.974,93.976,93.978,93.98,93.982,93.984,93.986,93.988,93.99,93.992,93.994,93.996,93.998,94,94.002,94.004,94.006,94.008,94.01,94.012,94.014,94.016,94.018,94.02,94.022,94.024,94.026,94.028,94.03,94.032,94.034,94.036,94.038,94.04,94.042,94.044,94.046,94.048,94.05,94.052,94.054,94.056,94.058,94.06,94.062,94.064,94.066,94.068,94.07,94.072,94.074,94.076,94.078,94.08,94.082,94.084,94.086,94.088,94.09,94.092,94.094,94.096,94.098,94.1,94.102,94.104,94.106,94.108,94.11,94.112,94.114,94.116,94.118,94.12,94.122,94.124,94.126,94.128,94.13,94.132,94.134,94.136,94.138,94.14,94.142,94.144,94.146,94.148,94.15,94.152,94.154,94.156,94.158,94.16,94.162,94.164,94.166,94.168,94.17,94.172,94.174,94.176,94.178,94.18,94.182,94.184,94.186,94.188,94.19,94.192,94.194,94.196,94.198,94.2,94.202,94.204,94.206,94.208,94.21,94.212,94.214,94.216,94.218,94.22,94.222,94.224,94.226,94.228,94.23,94.232,94.234,94.236,94.238,94.24,94.242,94.244,94.246,94.248,94.25,94.252,94.254,94.256,94.258,94.26,94.262,94.264,94.266,94.268,94.27,94.272,94.274,94.276,94.278,94.28,94.282,94.284,94.286,94.288,94.29,94.292,94.294,94.296,94.298,94.3,94.302,94.304,94.306,94.308,94.31,94.312,94.314,94.316,94.318,94.32,94.322,94.324,94.326,94.328,94.33,94.332,94.334,94.336,94.338,94.34,94.342,94.344,94.346,94.348,94.35,94.352,94.354,94.356,94.358,94.36,94.362,94.364,94.366,94.368,94.37,94.372,94.374,94.376,94.378,94.38,94.382,94.384,94.386,94.388,94.39,94.392,94.394,94.396,94.398,94.4,94.402,94.404,94.406,94.408,94.41,94.412,94.414,94.416,94.418,94.42,94.422,94.424,94.426,94.428,94.43,94.432,94.434,94.436,94.438,94.44,94.442,94.444,94.446,94.448,94.45,94.452,94.454,94.456,94.458,94.46,94.462,94.464,94.466,94.468,94.47,94.472,94.474,94.476,94.478,94.48,94.482,94.484,94.486,94.488,94.49,94.492,94.494,94.496,94.498,94.5,94.502,94.504,94.506,94.508,94.51,94.512,94.514,94.516,94.518,94.52,94.522,94.524,94.526,94.528,94.53,94.532,94.534,94.536,94.538,94.54,94.542,94.544,94.546,94.548,94.55,94.552,94.554,94.556,94.558,94.56,94.562,94.564,94.566,94.568,94.57,94.572,94.574,94.576,94.578,94.58,94.582,94.584,94.586,94.588,94.59,94.592,94.594,94.596,94.598,94.6]
    value=[2.60E-05,2.20E-05,8.29E-06,2.12E-05,2.77E-05,2.48E-05,2.72E-05,2.67E-05,2.92E-05,3.72E-05,4.13E-05,2.80E-05,4.93E-05,3.09E-05,3.79E-05,3.48E-05,3.83E-05,4.22E-05,4.17E-05,3.78E-05,4.69E-05,3.38E-05,5.11E-05,5.83E-05,5.88E-05,5.92E-05,6.38E-05,6.70E-05,6.30E-05,7.21E-05,7.53E-05,7.93E-05,8.21E-05,7.54E-05,8.87E-05,9.12E-05,0.000100424,9.11E-05,8.96E-05,0.000118801,0.000107633,0.000126091,0.000124221,0.000108207,0.000136457,0.000130596,0.000152099,0.00013823,0.000155361,0.000161361,0.000167217,0.000183162,0.000185612,0.000183055,0.000189706,0.000207726,0.000214834,0.00022727,0.000222366,0.000234173,0.000240609,0.000239019,0.000269917,0.000260292,0.000282644,0.00027731,0.000297652,0.000311452,0.000330588,0.000331614,0.000331826,0.000356541,0.000376376,0.00039932,0.00041054,0.000427617,0.00045856,0.000470683,0.00049686,0.000518821,0.000537758,0.0005682,0.000564864,0.000608049,0.000625232,0.000665117,0.000670282,0.000704632,0.000719507,0.000757605,0.000763204,0.000807913,0.000818188,0.000831624,0.000852694,0.000874274,0.000880998,0.000887723,0.000875245,0.000901133,0.000921604,0.000935223,0.000949959,0.000937612,0.000950197,0.000957355,0.000953751,0.000982172,0.000981221,0.00100853,0.00102667,0.00102669,0.00105574,0.00108365,0.00108636,0.00110722,0.00111056,0.00114673,0.00115927,0.00117247,0.00119071,0.00119603,0.00123228,0.00122913,0.00125073,0.00126484,0.00126831,0.00126835,0.00128381,0.00128496,0.00128405,0.00126474,0.00127613,0.00126286,0.0012474,0.00123279,0.00123024,0.00120334,0.00117227,0.00115641,0.00114642,0.00113255,0.00108954,0.00107683,0.00105351,0.00102523,0.00100872,0.000993603,0.000964623,0.000965037,0.000946794,0.000922722,0.000933952,0.000904442,0.000895548,0.000888446,0.000870864,0.000873231,0.000854588,0.000833252,0.000830022,0.000829953,0.000819047,0.000815306,0.000803422,0.000794591,0.000779194,0.00075524,0.000754777,0.000749691,0.000737804,0.000732445,0.000702536,0.000692528,0.000674018,0.00067484,0.000639792,0.000635792,0.000626138,0.0005991,0.000587421,0.000568247,0.000562794,0.000544879,0.000527089,0.000509768,0.000495954,0.000484254,0.000479062,0.000468996,0.00048606,0.000451912,0.000454933,0.000438166,0.000437183,0.00043843,0.000439968,0.000430965,0.000414931,0.000414751,0.000403296,0.000431317,0.000405131,0.000412197,0.000414497,0.000392055,0.000391118,0.000385579,0.000386599,0.000374663,0.000385385,0.000385648,0.000383122,0.000372119,0.000369917,0.000353168,0.000353181,0.000347583,0.000361177,0.000351511,0.00035298,0.000349048,0.00035811,0.000343739,0.000336668,0.000346541,0.000347541,0.00034776,0.000345813,0.000351469,0.000346325,0.000353857,0.000346726,0.000357221,0.000352167,0.000365603,0.000346755,0.000343428,0.000353139,0.000358943,0.000366715,0.000357359,0.00035113,0.000356782,0.000369601,0.000377012,0.00036818,0.00036038,0.000363478,0.000374728,0.000367305,0.000371811,0.000359722,0.000363027,0.000370304,0.00036203,0.000361357,0.000348619,0.000359053,0.000357895,0.000366932,0.00035721,0.000347487,0.000340219,0.000350902,0.000352213,0.000360254,0.000347992,0.000347695,0.000357473,0.000346486,0.000353058,0.000348425,0.000346249,0.000348966,0.000347535,0.000338569,0.000347866,0.00034897,0.000353518,0.000342128,0.000341428,0.00034364,0.000323307,0.000339836,0.000339775,0.000350856,0.000333605,0.000330988,0.000319987,0.000321163,0.000317401,0.000320364,0.000323423,0.000311501,0.000302288,0.00030242,0.000309918,0.00030924,0.000314864,0.000317319,0.000307364,0.00030862,0.000294058,0.000305557,0.000295109,0.000293966,0.000293612,0.000296397,0.000296149,0.00028707,0.000292797,0.000292019,0.000306924,0.000286493,0.000286587,0.0002868,0.000293076,0.000300582,0.000295486,0.000300685,0.000301557,0.000290775,0.000304124,0.000299222,0.000300966,0.000297506,0.000296073,0.000297694,0.000294032,0.000308573,0.000328143,0.000309997,0.000326017,0.000313829,0.000304159,0.000323979,0.000314373,0.000304696,0.000310808,0.000321803,0.000312878,0.000307651,0.000313683,0.000324065,0.000313073,0.000328637,0.000322721,0.000329828,0.000328005,0.000325062,0.000323671,0.00033872,0.000335649,0.000339059,0.000342915,0.000330123,0.000336656,0.000340834,0.000346942,0.000344649,0.00035269,0.000362618,0.000361879,0.000360022,0.000367923,0.00037459,0.000369593,0.000367293,0.000381245,0.000370876,0.000373819,0.000378304,0.000386874,0.000382356,0.00038503,0.000392875,0.00039451,0.000388735,0.00040811,0.000412607,0.000402629,0.000408825,0.000411594,0.000410457,0.000411628,0.000405221,0.000398991,0.000406894,0.000406584,0.000411279,0.000412636,0.000401851,0.000423301,0.000398497,0.000403273,0.000412579,0.000408441,0.000406143,0.000396886,0.00040325,0.0004022,0.000404692,0.00040334,0.000403779,0.000416492,0.000404692,0.000411096,0.000407771,0.000436054,0.000419604,0.00043401,0.000440001,0.000434188,0.000452626,0.000458296,0.000465882,0.00046413,0.000472697,0.000481123,0.000476174,0.000485054,0.000506987,0.00050394,0.000514516,0.000507282,0.000512602,0.000507982,0.000510948,0.000513286,0.000512662,0.000521432,0.000510276,0.000514104,0.000514249,0.000518089,0.000507969,0.000508253,0.000501855,0.000500447,0.000499418,0.000510315,0.000495052,0.000501195,0.000511034,0.000518527,0.000510869,0.00052578,0.000512092,0.000519628,0.000532253,0.000525296,0.000558492,0.00055561,0.000540211,0.000563345,0.000568411,0.000569059,0.000574313,0.000584538,0.000590875,0.000605115,0.000604173,0.000605051,0.000615037,0.000620564,0.000622207,0.00061523,0.000617655,0.000612709,0.000606964,0.000603767,0.000588442,0.000592503,0.000582394,0.000572761,0.000558761,0.000553939,0.000538754,0.000518476,0.000515393,0.000518741,0.000495789,0.000502222,0.00047195,0.000482112,0.000451012,0.000463664,0.000448059,0.000437799,0.000441639,0.000427858,0.000430054,0.000434464,0.000432874,0.000417185,0.000417604,0.000401819,0.000409292,0.000409764,0.000410817,0.000409253,0.000408757,0.000404871,0.000393327,0.000394265,0.000392371,0.000378474,0.00037859,0.000374161,0.000375474,0.000354502,0.000333261,0.000344417,0.0003205,0.000330926,0.000303117,0.00030482,0.000293708,0.000272422,0.000268705,0.000267572,0.000269676,0.000251861,0.000228126,0.000240493,0.000227514,0.000225634,0.000215935,0.000208624,0.000215592,0.000218583,0.000215436,0.000198779,0.000207982,0.000198622,0.000194446,0.000200607,0.000190499,0.000198648,0.000202117,0.0001707,0.000182261,0.000172251,0.000161376,0.000188943,0.000173679,0.000166899,0.000171608,0.000173612,0.000160713,0.000145385,0.000166459,0.000168568,0.00015592,0.000156128,0.000147123,0.000141485,0.000141666,0.000134523,0.000130225,0.000127613,0.000134936,0.000111087,0.000123635,0.000125399,9.45E-05,0.000118604,9.23E-05,9.49E-05,0.000104583,8.75E-05,9.83E-05,7.92E-05,7.89E-05,8.35E-05,7.36E-05,8.50E-05,8.33E-05,6.34E-05,8.25E-05,6.57E-05,6.08E-05,5.85E-05,7.85E-05,6.66E-05,6.71E-05,5.96E-05,4.89E-05,6.04E-05,4.82E-05,5.09E-05,4.97E-05,5.09E-05,4.79E-05,4.57E-05,3.93E-05,3.95E-05,3.25E-05,3.58E-05,4.00E-05]
    
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
    
    #print(times_4[0])
    #print(times_4[-1])
    
    interp_func=interpolate.interp1d(frequency,value,kind='cubic')
    #print(*interp_func(times_4[start_trace:end_trace]),sep=',')
    integs=np.array(integs)
    #print(len(integs[start_trace:end_trace]))
    #print(len(times_4[start_trace:end_trace]))
    integs_correction_arr=integs+EIK_weight_factor*interp_func(times_4[start_trace:end_trace])
    

    
    fig = plt.figure(num=44,figsize=(3.25,2.25), dpi=300)
    #plt.title('EIK corrected')
    plt.plot(times_4[start_trace:end_trace],max(integs)-EIK_weight_factor*interp_func(times_4[start_trace:end_trace]),'r',label='EIK corr')
    plt.plot(times_4[start_trace:end_trace],integs,label='raw')
    plt.plot(times_4[start_trace:end_trace],integs_correction_arr,'b',label='corrected')
    plt.xlabel ('sat pulse frequency (GHz)')
    plt.ylabel('signal')
    plt.legend(bbox_to_anchor=(1.05, 1))
    #plt.xlim(93.5,94.5)
    #plt.ylim(0,8)
    
    integs=integs_correction_arr
else:
    pass

#******************************************************************************
#******************************************************************************
#******************************************************************************

'''
fig = plt.figure(num=33,figsize=(3.25,2.25), dpi=300)

plt.plot(times_4[start_trace:end_trace],integs_im,label='im')
plt.plot(times_4[start_trace:end_trace],integs_re,label='re')
#plt.xlabel(r'$\tau_{pump}$ (ns)')
plt.xlabel(x_axis_label)
plt.legend()
plt.ylabel('integrated signal')




fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)

plt.plot(times_4[start_trace:end_trace],NormalizeData(integs))
#plt.xlabel(r'$\tau_{pump}$ (ns)')
plt.xlabel(x_axis_label)
plt.ylabel('integrated signal')

'''




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
    print('\nguess=',guess)
        
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

