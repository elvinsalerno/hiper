# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:39:20 2021

@author: hiper
"""

import numpy as np
import matplotlib.pyplot as plt


alpha=10
beta=67
gamma=0


#function does vector rotation in order Rx, Ry
#assume vector always starts at vector=np.array([[0],[0],[1]])
def vector_rotation(beta,alpha):
    vector=np.array([[0],[0],[1]])
    alpha=alpha*np.pi/180
    beta=beta*np.pi/180
    #gamma=gamma*np.pi/180
    
    Rx=np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
    Ry=np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
    #Rz=np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])
   
    result1=np.dot(Rx,vector)
    result2=np.dot(Ry,result1)
    #result3=np.dot(Rz,result2)
    
    #print(np.round(result3,3))
    
    gamma_final=np.arccos(result2[2])
    return np.round(gamma_final[0]*180/np.pi,5)
    

beta=np.array([0,10,20,30,45,46,109,223])
alpha=np.array([10]*len(beta))
#gamma=np.array([0]*len(beta))





'''
xxx=[]

for i in range(0,len(beta)):
    xxx.append(vector_rotation( alpha[i],beta[i]))


print(xxx)


'''

###############################################################################

gamma_res=np.array([10.0, 14.10604, 22.26874, 31.47495, 45.86397, 46.83474, 108.70055, 136.07453])
beta_in=np.array([0,10,20,30,45,46,109,223])


plt.plot(beta_in,gamma_res,'r--')



from scipy.optimize import curve_fit

#guess=[6,1000,1]

pars, pcov = curve_fit(vector_rotation,beta_in,gamma_res)#, p0=guess,maxfev=100000)#,bounds=(0,np.inf),maxfev=3800)
#print(pars)
#pars_peaks=np.ndarray.tolist(pars_peaks)
#ax1.plot(peaks,corrfunc(peaks,*pars_peaks))#,'--',linewidth=0.5,color=colors[i,:])

#print(pars)






plt.plot(beta_in,vector_rotation(beta_in,*pars),'r--')











