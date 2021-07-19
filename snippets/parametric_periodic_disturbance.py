#!/usr/bin/python3

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

# Parameter offset
offset = 0 
# Parameter amplitude
amp_param = 1 
# Disturbance amplitude
amp_dist = 0.25
 # Trials per second
freq_trial = 1/float(3)

# OLD
def p_old(t,alpha):
  return offset + (amp_param + amp_dist * np.sin(2*np.pi*alpha*freq_trial*t)) * np.sin(2*np.pi*freq_trial*t)

# Periodic parameters with otherly periodic disturbance
def p_new(t, alpha):
  return offset + amp_param * np.sin(2*np.pi*freq_trial*t) + amp_dist * np.sin(2*np.pi*alpha*freq_trial*t)

# Alpha chooses the disturbance frequency relative to the trial frequency
alphas = np.linspace(0,1,4)
alphas = np.array([0.0,0.5,1.0])
alphas = np.array([0.0,10])
num = alphas.shape[0]

# The time-axis
tAxis = np.linspace(0,9,101)
dt = tAxis[1] - tAxis[0]

# Create 3D-figure
fig = plt.figure()

# Compute parameters
ps_old = np.array([p_old(tAxis,alpha) for alpha in alphas])
ps_new = np.array([p_new(tAxis,alpha) for alpha in alphas])

# Get frequencies by fft
fft_old = [fftpack.fft(ps_old[k,:]) for k in range(alphas.shape[0])]
fft_new = [fftpack.fft(ps_new[k,:]) for k in range(alphas.shape[0])]

freqs_old = [fftpack.fftfreq(len(fft_old[k])) * dt for k in range(alphas.shape[0])]
freqs_new = [fftpack.fftfreq(len(fft_new[k])) * dt for k in range(alphas.shape[0])]

#fft_old = [val for fft in fft_old if abs(fft) > 1e3]
#freqs_new = [[freq for freq in freqs_new if abs()] for fft in fft_new]
#fft_new = [[val for val in fft if abs(val) > 1e3] for fft in fft_new]

# Plot old disturbed parameters
plt.subplot(231)
[plt.plot(tAxis,p_old(tAxis,alpha),label='alpha='+str(alpha)) for alpha in alphas]
#[plt.axhline(np.mean(ps_old[k,:])) for k in range(num)]
plt.axhline(0,linestyle='--')
plt.grid()
plt.subplot(232)
plt.stem(freqs_old[0], np.abs(fft_old[0]))
plt.grid()
plt.subplot(233)
plt.stem(freqs_old[1], np.abs(fft_old[1]))
plt.grid()

#[plt.magnitude_spectrum(ps_old[k,:], Fs=dt, label='alpha='+str(alphas[k]),scale='dB') for k in range(alphas.shape[0])]

# Plot new disturbed parameters
plt.subplot(234)
[plt.plot(tAxis,p_new(tAxis,alpha),label='alpha='+str(alpha)) for alpha in alphas]
plt.axhline(0,linestyle='--')
plt.legend()
plt.grid()
plt.subplot(235)
#[plt.stem(freqs_new[k], np.abs(fft_new[k])) for k in range(alphas.shape[0])]

plt.stem(freqs_new[0], np.abs(fft_new[0]))
plt.grid()
plt.subplot(236)
plt.stem(freqs_new[1], np.abs(fft_new[1]))
plt.grid()

#plt.grid()
#plt.subplot(236)
#[plt.magnitude_spectrum(ps_new[k,:], Fs=dt, label='alpha='+str(alphas[k]),scale='dB') for k in range(alphas.shape[0])]
#plt.grid()


plt.show()
