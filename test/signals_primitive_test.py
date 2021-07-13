import numpy as np
import matplotlib.pyplot as plt
import thesis_code.utils.signals as signals

# Define sample numbers and some indices
N_total = 100

# Define axis
kAxis = np.arange(N_total)

# Define signal parameters
N_per = 10

# Create signals
triangle = [ signals.triangle(k,N_per) for k in kAxis ]

# Plot pulse signals
fig,ax = plt.subplots(1,1)
plt.suptitle("Primitive signals test")
plt.title("Triangle")
plt.plot(kAxis, triangle, 'x-', label="Triangle")
plt.grid()
plt.ylabel("s")
plt.xlabel("k")
plt.legend(loc="best")

# Show plots
plt.show()
