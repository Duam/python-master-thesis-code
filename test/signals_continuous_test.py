import numpy as np
import matplotlib.pyplot as plt
import thesis_code.utils.signals_continuous as signals

# Define sample numbers and some indices
T_total = 100
t_start = -10
t_end = t_start + T_total

# Sampling
N = 300
dt = T_total/float(N)

# Define axes
kAxis = np.arange(N)
tAxis = np.linspace(t_start, t_end, N)

""" TEST: PULSE SIGNALS """

# Define signal parameters
T_per = 20
width = 3

# Create signal containers
pulse_single = np.zeros(N)
pulse_periodic = np.zeros(N)
pulse_FM_linear = np.zeros(N)

for k in kAxis:
  pulse_single[k] = signals.pulse_single(tAxis[k], width, amp=0.2)
  pulse_periodic[k] = signals.pulse_periodic(tAxis[k], T_per, width, amp=0.5)
  pulse_FM_linear[k] = signals.pulse_FM_linear(tAxis[k], T_per, width, amp=1.0)

# Plot pulse signals
fig,ax = plt.subplots(1,1)
plt.suptitle("Pulse test")
plt.title("Single, periodic, FM_linear")
plt.plot(tAxis, pulse_single, 'x-', label="single")
plt.plot(tAxis, pulse_periodic, 'x-', label="periodic")
plt.plot(tAxis, pulse_FM_linear, 'x-', label="FM_linear")
plt.ylabel("s")
plt.xlabel("k")
plt.legend(loc="best")

plt.show()