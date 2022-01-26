# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:44:33 2021

@author: evsal
"""
#write 'field' or 'freq'
interactive_plot='freq'
#interactive_plot='field'


exp_freq=[94]
plot_field=3.28

#choose 'yes' or 'no'
stack_plot='yes'



slider_start = 3.1
slider_end = 3.5
###############################################################################
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


exp_freq.extend([94,94,94,94,94,94,94,94,94,94,94,94])
exp_freq=np.array(exp_freq,dtype=object)

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

font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)


fig = plt.figure(num=1,figsize=(3.25,2.25), dpi=300)


def NormalizeData(data):
    return data/max(data)
    #return (data - np.min(data)) / (np.max(data) - np.min(data))
fields_array=[]
data_mag_array=[]


def plot_fcn(filename_in,lab,numb):

    #filename_in="".join(["\\",filename_in])
    #filename_comby=file_directory+filename_in+'.txt'
    raw_data = np.genfromtxt(filename_in,skip_header=2,delimiter=',')
    field=raw_data[:,0]
    data_Re=raw_data[:,1]
    data_Im=raw_data[:,2]
    data_mag=(np.sqrt((np.square(data_Im))+(np.square(data_Re))))
    #freq = (93.55*3.306)/field
    
    data_mag = NormalizeData(data_mag)

    savGol='off'
    from scipy.signal import savgol_filter
    if savGol=='on':
        data_mag=savgol_filter(data_mag, 11, 2)
    
    #plt.plot(field,data_Re,label='Re')
    #plt.plot(field,data_Im,label='Im')
    
    if stack_plot=='yes':
        plt.plot(field,data_mag+numb*1.1,label=lab)
    else:
        plt.plot(field,data_mag,label=lab)
    
    #plt.axvline(x = 94.40)
    #plt.axvline(x = 93.55)

    fields_array.append(field)
    data_mag_array.append(data_mag)
    

   
    
#plot_fcn(file_directory,'test_93p6GHZ_25_250_50_5k_21dB_Data','93.6 GHz')
#plot_fcn(file_directory,'test_93p55GHZ_25_250_50_5k_21dB_Data','93.55 GHz')
#plot_fcn(file_directory,'test_94p45GHZ_25_250_50_5k_21dB_Data','94.45 GHz')
#plot_fcn(file_directory,'FSE_93p55GHZ_20_350_40_5K_20dB_Data','xxx')

name_list=['0$^\circ$','72$^\circ$','108$^\circ$','144$^\circ$','149$^\circ$' ]
name_list=['xxx']*10
name_list=['11.5','12.5','12.5','13.5','15.5','0']


for i in range(0,len(filez[0])):
    plot_fcn(filez[0][i],name_list[i],i)
    print('\n',filez[0][i])



plt.xlabel('Field (T)')
plt.ylabel('echo intensity')
plt.title('Field dependence')
#plt.legend()
#plt.xlim(3,3.75)

##############################################################################
#fields_array=np.array(fields_array)
data_mag_array=np.array(data_mag_array,dtype=object)



derivative='off'


if derivative=='on':
    from scipy.signal import savgol_filter
    fig = plt.figure(num=2,figsize=(3.25,2.25), dpi=300)
    for i in range(0,len(fields_array)):
        #(fields_array[i],data_mag_array[i])
        deriv=np.diff(data_mag_array[i])
        deriv=savgol_filter(deriv, 11, 2)
        plt.plot(fields_array[i][0:-1],deriv)
    plt.xlabel('Field (T)')
    plt.ylabel('dI/dB')

##############################################################################
fig = plt.figure(num=3,figsize=(3.25,2.25), dpi=300)

sliders=np.linspace(slider_start,slider_end,100)
'''
freq_vals=[]
for i in range(0,len(fields_array)):
    freq_vals_in=[]
    for j in range(0,len(sliders)):
        g_list=(exp_freq[i])/(13.94)/(fields_array[i])
        g_list=np.array(g_list)
        freq_vals_in.append(sliders[j]*13.94*g_list)

    freq_vals.append(freq_vals_in)
'''

freq_vals=[]

for j in range(0,len(sliders)):
    g_list=(exp_freq[0])/(13.94)/(fields_array[0])
    g_list=np.array(g_list)
    freq_vals.append(sliders[j]*13.94*g_list)




    
#freq_vals=freq_vals[0]
#freq_vals=np.array(freq_vals)
'''
if interactive_plot=='field':
    for i in range(0,len(fields_array)):
        plt.plot(freq_vals[i],data_mag_array[i])
elif interactive_plot=='freq':
    '''
    
    
    
specwavel1=min(sliders, key=lambda x:abs(x-plot_field))
indexwavel1, = np.where(sliders == specwavel1)


plt.plot(freq_vals[indexwavel1[0]],data_mag_array[0])

plt.xlabel('Frequency (GHz)')
plt.ylabel('echo intensity')
plt.title(' Freq dependence at %s T'% (np.round(sliders[indexwavel1[0]],3)))
plt.xlim(93.8,94.3)

#freq_vals=freq_vals[0]
if interactive_plot=='field':
    fig = go.Figure()
elif interactive_plot=='freq':
    fig = go.Figure(layout_title_text="field=%s"% (np.round(sliders[indexwavel1[0]],3)))



if interactive_plot=='field':
    for i in range(0,len(data_mag_array)):
        fig.add_trace(go.Scatter(x=fields_array[i], y=data_mag_array[i],
                    mode='lines',
                    name='xxx'))
        fig.update_xaxes(title='Field (T)')
    fig.update_yaxes(title='magnitude')

    fig.show()

elif interactive_plot=='freq':
    for j in range(0,len(freq_vals)):
        fig.add_trace(go.Scatter(visible=False,x=freq_vals[j], y=data_mag_array[0],
                    mode='lines',
                    name='xxx'))
    fig.update_xaxes(title='Frequency (GHz)')

    fig.data[5].visible = True


    
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title":  str(np.round(sliders[i],3))+' T'}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=100,
        currentvalue={"prefix": "Field: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders
    )
    
    fig.show()
    
        



'''
#**********************************************
for step in np.arange(0, 5, 0.1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(step),
            x=np.arange(0, 10, 0.01),
            y=np.sin(step * np.arange(0, 10, 0.01))))
'''
# Make 10th trace visible


#**********************************************


#print(freq_vals)
    

'''
fig = go.Figure()
fig.add_trace(go.Scatter(x=field, y=data_mag,
                    mode='lines',
                    name='test'))
'''

'''
fig.update_yaxes(title='magnitude')

fig.show()
'''

