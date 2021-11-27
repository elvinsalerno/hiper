# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:24:45 2021

@author: hiper
"""


import numpy as np
import matplotlib.pyplot as plt


#function does vector rotation in order Rx, Ry
#assume vector always starts at vector=np.array([[0],[0],[1]])
def vector_rotation(beta,alpha,beta0):
    vector=np.array([[0],[0],[1]])
    alpha=alpha*np.pi/180
    beta=beta0*np.pi/180+beta*np.pi/180
    #gamma=gamma*np.pi/180
    
    Rx=np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]],dtype=object)
    Ry=np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]],dtype=object)
    #Rz=np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])
    #print(Rx)
    result1=np.dot(Rx,vector)
    result2=np.dot(Ry,result1)
    #result3=np.dot(Rz,result2)
    
    #print(np.round(result3,3))
    #print(*result2[2])
    
    gamma_final=np.arccos(*result2[2])
    return np.round(gamma_final*180/np.pi,5)
    

############################FITTING##########################################
fig1,ax1 = plt.subplots(1,1,num=1,figsize=(3,2.25), dpi=300)

beta_in=np.linspace(0,360,5)
alpha_in=np.array([21]*5)
beta0_in=np.array([80]*5)

gamma_res=vector_rotation(beta_in,alpha_in,beta0_in)

gamma_res=np.random.normal(gamma_res)

plt.plot(beta_in,gamma_res,'bo',)




from lmfit import Model, Parameter, report_fit

model = Model(vector_rotation, independent_vars=['beta'])
result = model.fit(gamma_res, beta=beta_in,alpha=40,beta0=80,method='cobyla')

#print(result[0])

print(result.values)

#result.plot()
#print(result.values['alpha'])
#print(report_fit(result.params))


beta_in=np.linspace(0,360,500)

plt.plot(beta_in,vector_rotation(beta_in,result.values['alpha'],result.values['beta0']),'r--',label=r"$\alpha$="+" %.1f"%(result.values['alpha'])+"\u00b0")
plt.ylabel(r"$\gamma$"+" (\u00b0)")
plt.xlabel(r"$\beta$"+" (\u00b0)")

plt.legend()




'''
from scipy.optimize import curve_fit

guess=[67,89]

pars, pcov = curve_fit(vector_rotation,beta_in,gamma_res, p0=guess,bounds=(0,90),maxfev=38000)

print('alpha =',pars[0],"\u00b0")
print('beta0 =',pars[1],"\u00b0")



plt.plot(beta_in,vector_rotation(beta_in,*pars),'r--',label=r"$\alpha$="+" %.1f"%(pars[0])+"(\u00b0)")
plt.ylabel(r"$\gamma$"+" (\u00b0)")
plt.xlabel(r"$\beta$"+" (\u00b0)")

plt.legend()
'''






