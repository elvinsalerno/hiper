# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:39:20 2021

@author: hiper
"""

import numpy as np

vector=np.array([[0],[0],[1]])
origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point

alpha=50
beta=67
gamma=0


alpha=alpha*np.pi/180
beta=beta*np.pi/180
gamma=gamma*np.pi/180

Rx=np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
Ry=np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
Rz=np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])



result1=np.dot(Rx,vector)
result2=np.dot(Ry,result1)



print(np.round(result2,3))



gamma_final=np.arccos(result2[2])#/np.sqrt((result2[0])**2+(result2[1])**2+(result2[2])**2))

phi = np.arccos(result2[0]/np.sin(gamma_final))


print('\n',(gamma_final)*180/np.pi)
print('\n phi is %.3f'%(phi*180/np.pi))



