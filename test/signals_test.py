import numpy as np
import matplotlib.pyplot as plt
import thesis_code.utils.signals as signals

# Define sample numbers and some indices
N_total = 100
k_start = -10
k_end = k_start + N_total

# Define axis
kAxis = np.arange(k_start, k_end)
print(kAxis)
print(range(N_total))

""" TEST: PULSE SIGNALS """

# Define signal parameters
N_per = 12
width = 3

# Create signal containers
pulse_single = np.zeros(N_total)
pulse_periodic = np.zeros(N_total)
pulse_FM_linear = np.zeros(N_total)

for i in range(N_total):
  pulse_single[i] = signals.pulse_single(kAxis[i], width, amp=0.2)
  pulse_periodic[i] = signals.pulse_periodic(kAxis[i], N_per, width, amp=0.5)
  pulse_FM_linear[i] = signals.pulse_FM_linear(kAxis[i], N_per, width, amp=1.0)

# Plot pulse signals
fig,ax = plt.subplots(1,1)
plt.suptitle("Pulse test")
plt.title("Single, periodic, FM_linear")
plt.plot(kAxis, pulse_single, 'x-', label="single")
plt.plot(kAxis, pulse_periodic, 'x-', label="periodic")
plt.plot(kAxis, pulse_FM_linear, 'x-', label="FM_linear")
plt.grid()
plt.ylabel("s")
plt.xlabel("k")
plt.legend(loc="best")

""" TEST: COMPOUND SIGNALS """

tau = 1

spikes = signals.decaying_spike_periodic(N_total, N_per, tau, shift=k_start)

fig,ax = plt.subplots(1,1)
plt.suptitle("Compounds test")
plt.title("Decaying periodic spike")
plt.grid()
plt.plot(kAxis, spikes)

# Show plots
plt.show()
