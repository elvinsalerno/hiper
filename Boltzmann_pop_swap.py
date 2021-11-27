# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:28:38 2021

@author: evsal
"""

import matplotlib.pyplot as plt
import numpy as np



swap=[[0,1],[1,2],[2,3]]
probe=[[0,1],[1,2],[2,3],[3,4]]
energy_levels=np.array([0,2.70,6.30,11.69,19.20])

#energy_levels=np.array([93.68, 93.77, 93.89, 94.07, 94.32])
#energy_levels=energy_levels*29.98



fig1,ax1 = plt.subplots(1,1,num=1,figsize=(3,2.25), dpi=300)
fig1,ax1 = plt.subplots(1,1,num=2,figsize=(3,2.25), dpi=300)
fig1,ax1 = plt.subplots(1,1,num=3,figsize=(3,2.25), dpi=300)
c=3*(10**10)
h=6.6262*(10**-34)
k=1.38066*(10**-23)
N=6.022*(10**23)


probe=[]


for i in range(0,len(energy_levels)-1):
    probe.append([i,i+1])

transitions=[]
for i in range(0,len(energy_levels)-1):
    transitions.append(energy_levels[i+1]-energy_levels[i])

print(transitions)
transitions=np.array(transitions)
transitions=transitions/29.98
transitions=transitions+93.59


print(transitions)


vs=np.array([1]*int(len(energy_levels)))



#T=np.linspace(T1,T2,1000)
Temp=5

def num(Ei,Di,T):
    E=h*c*Ei
    return Di*np.e**(-E/(k*T))

denom_val=0
for i in range(0,len(energy_levels)):
    denom_val=denom_val+num(energy_levels[i],vs[i],Temp)
    
def Pi(Ei,Di,T):
    return num(Ei,Di,T)/denom_val

pops=Pi(energy_levels,vs,Temp)
#print(pops)
#print(ks)
#P0=Pi(E0,Di,T)

print('thermal pops')
print(pops,'\n')


difference_list_before=[]

for i in range(0,len(probe)):
    difference_list_before.append(pops[probe[i][0]]-pops[probe[i][1]])



plt.figure(1)
plt.barh(energy_levels,[max(pops)]*len(energy_levels),color='k',height=0.05)
plt.barh(energy_levels,pops,color='blue',height=0.3)
plt.ylabel("Energy (cm$^{-1}$)")
plt.xlabel("Population")
plt.title('Boltzmann populations')
#plt.ylim(-2,20)
#plt.xlim(0,1)


w=1

for i in range(0,len(swap)):
    pops[swap[i][0]],pops[swap[i][1]]=w*pops[swap[i][1]]+(1-w)*pops[swap[i][0]],w*pops[swap[i][0]]+(1-w)*pops[swap[i][1]]





print('swapped pops')
print(pops,'\n')

plt.figure(2)
plt.barh(energy_levels,[max(pops)]*len(energy_levels),color='k',height=0.05)
plt.barh(energy_levels,pops,color='blue',height=0.3)
plt.ylabel("Energy (cm$^{-1}$)")
plt.xlabel("Population")
plt.title('populations after swaps')
#plt.ylim(-2,20)
#plt.xlim(0,1)

difference_list_after=[]

for i in range(0,len(probe)):
    difference_list_after.append(pops[probe[i][0]]-pops[probe[i][1]])
    
print('thermal pop difference')
print(difference_list_before)
print('\nswapped pop difference')
print(difference_list_after)


difference_list_after=np.array(difference_list_after)
difference_list_before=np.array(difference_list_before)

ratio_after_before=np.array(difference_list_after)/difference_list_before

print('\npopulation difference after/before')
print(ratio_after_before)




plt.figure(3)
plt.bar(transitions,difference_list_before,color='blue',width=0.005)



