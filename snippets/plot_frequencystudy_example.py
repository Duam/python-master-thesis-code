#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import utils.signals as signals

# Total time and number of samples
T_total = 60.0
N_total = 240

# Time- and sample-axes
tAxis = np.linspace(0, T_total, N_total, endpoint=False)
kAxis = np.arange(0, N_total)

# Time step
dt = tAxis[1] - tAxis[0]
print("dt =", dt)

# Period time and number of samples
T_per = 15.0
N_per = np.floor(T_per / dt)

# Step
#step_size = 1.0
#step_time = T_per * 2
#s = lambda t: step_size if t >= step_time else 0.0

# Noise
#noise_covar = 1e-2
#v = lambda t: np.random.normal(0,noise_covar)

# Periodic signal
signal_amplitude = 1.0
signal_frequency = 1/float(T_per)
#p = lambda t: signal_amplitude/2.0 * np.sin(2*np.pi*signal_frequency*t)

# Output = Periodic signal + Noise + Step 

# Evaluate!
output = [ y(t) for t in tAxis ]

# Trial markers. For plotting vertical lines
trial_markers = [ t for t in tAxis if 0 <= np.mod(t,T_per) < dt ]

# Function string
fun_str = r"$p(t) = signal_1(t) + signal_2(t)$"

# Plot!
fig, ax = plt.subplots(1,1)
plt.sca(ax)
#plt.suptitle("Trial-axis step function with noise")
plt.title(fun_str)
plt.ylabel(r"p(t)")
plt.xlabel("Time t")
plt.axhline(color='grey', linestyle='--')
#plt.axhline(step_size, color='grey', linestyle='--')
[plt.axvline(k, color='g', linestyle='--') for k in trial_markers ]
plt.plot(tAxis, output, 'x-', markersize=5, color='blue')
plt.grid()
plt.show()