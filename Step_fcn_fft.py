# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:17:47 2021

@author: hiper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft , fftfreq

N = 10000
# sample spacing
T = 100 / N

x = np.linspace(0.0 , N*T , N)



def step(x):
    if x < 40 or x > 44 :
        return 0
    else:
        return 1 
   

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
   
f = [step(y) for y in x]
#f = np.sin(x)
yf = fft(f)
yf = NormalizeData(np.abs(yf))
xf = fftfreq(N , T)

plt.figure(1)
plt.plot(x, f, 'r')

plt.figure(2)
plt.plot(xf , yf , 'b')
plt.xlim(-0.5,0.5)
plt.xlabel("GHz")