# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:20:37 2021

@author: hiper
"""

import numpy as np


gamma_finals=np.array([23,43])



alpha=67
beta=52
gamma=0


alpha=alpha*np.pi/180
beta=beta*np.pi/180
gamma=gamma*np.pi/180



def Rx(alpha):
    return np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
def Ry(beta):
    return np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
def Rz(gamma):
    return np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])

def vector_final(Y,P):
    v_fin= np.array([[np.sin(Y)*np.cos(P)],[np.sin(Y)*np.sin(P)],[np.cos(Y)]])
    v_init1=np.dot(Rx())


vector=vector_final(gamma_finals[0],0)

result1=np.dot(Rx,vector)
result2=np.dot(Ry,result1)






gamma_final_rads=gamma_finals*np.pi/180


    

    















