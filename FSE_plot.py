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
import matplotlib.cm as cm


import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'



####################################################################
#This is the method by which to call the input file. 
#'1' opens a dialogue box from which you can select the filename. 
#'2' Requires you to type the directory as a raw string r'directory' as well 
#    as type the xyz filename below
#Generally you should use '1', but if it doesnt work (might not work well on
#    mac or linux), try '2', and change the directory and filename below. 
File_input_method='2'


#This only necessary if you used File_input_method='2'
#The directory where the .xyz file is located
input_directory=r"C:\Users\evsal\Google Drive\Globus_access"

#input pdb filename, only .xyz file types accepted
input_xyz_file="adocbl_trunc_fullado" 
#####################################################################



if File_input_method=='1':
    import os
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    (directory,file) = os.path.split(file_path)
    (xyz_file,ext) = os.path.splitext(file)
    
if File_input_method=='2':
    file_directory=input_directory
    xyz_file=input_xyz_file




file_directory=r"C:\Users\hiper\Desktop\Elvin_Mixer_test\multiple_freq_sweeps"
file_directory=r'E:\Manoj\CuGd3_NewSample_Sep_16_2021\New_mixer\09222021'

font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
fig = plt.figure(num=1,figsize=(8.25,5.25), dpi=300)

fig = go.Figure()


def plot_fcn(file_directory,filename_in,lab):
    filename_in="".join(["\\",filename_in])
    filename_comby=file_directory+filename_in+'.txt'
    raw_data = np.genfromtxt(filename_comby,skip_header=2,delimiter=',')
    field=raw_data[:,0]
    data_Re=raw_data[:,1]
    data_Im=raw_data[:,2]
    data_mag=(np.sqrt((data_Im**2)+(data_Re**2)))
    plt.plot(field,data_mag,label=lab)
    
    fig.add_trace(go.Scatter(x=field, y=data_mag,
                    mode='lines',
                    name=lab))


#plot_fcn(file_directory,'test_93p6GHZ_25_250_50_5k_21dB_Data','93.6 GHz')
#plot_fcn(file_directory,'test_93p55GHZ_25_250_50_5k_21dB_Data','93.55 GHz')
#plot_fcn(file_directory,'test_94p45GHZ_25_250_50_5k_21dB_Data','94.45 GHz')
plot_fcn(file_directory,'FSE_93p55GHZ_20_350_40_5K_20dB_Data','xxx')


#plt.legend()
plt.xlabel('Field (T)')
plt.ylabel('echo intensity')
plt.title('94 GHz')
plt.legend()
#plt.xlim(3.25,3.4)


'''
fig = go.Figure()
fig.add_trace(go.Scatter(x=field, y=data_mag,
                    mode='lines',
                    name='test'))
'''
fig.update_yaxes(title='magnitude')
fig.update_xaxes(title='Field (T)')
fig.show()





