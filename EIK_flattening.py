
import numpy as np 
import matplotlib.pyplot as plt
from scipy import interpolate

fig = plt.figure(num=64,figsize=(3.25,2.25), dpi=300)

#%%

# EIK output measured directly 
file_directory=r"C:\SpecManData\Manoj"
filename_in="EIK_output_4db_1KHz_94Ghz"

filename_in="".join(["\\",filename_in])
filename_comby=file_directory+filename_in+'.dat'

raw_data = np.genfromtxt(filename_comby,skip_header=1)#,delimiter=' ')


data_freq=raw_data[:,0] + 92.2
data_real=raw_data[:,1]
data_imag = raw_data[:,2]

plt.plot(data_freq,data_real *1000)

#%%

interp_func=interpolate.interp1d(data_freq,data_real,kind='cubic')
plt.plot(data_freq,interp_func(data_freq))
fnmr = np.linspace(93.75,94.35,301)
print(fnmr)
print(len(fnmr))

#%%

eik_output =[interp_func(item) for item in fnmr]

fig = plt.figure(num=65,figsize=(3.25,2.25), dpi=300)
plt.plot(fnmr,eik_output,'r')
#plt.xlim(93.4,94.6)

#%%

amplitude = [ 1/ item for item in eik_output] 

fig = plt.figure(num=66,figsize=(3.25,2.25), dpi=300)
amplitude = np.round_(amplitude/np.max(amplitude) , 3)

print('amplitude : ')
print(*amplitude, sep=',')

plt.plot(fnmr, amplitude,'g')

#%%

print(len(amplitude))
#%%