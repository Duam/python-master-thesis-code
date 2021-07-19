#!/usr/bin/python3

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, LinearLocator
import numpy as np

# x-Axis is a mix of logarithmically and linearly spaced points
x = np.array([1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1e0, 1e1, 1e2])

# This array tells us the which element in x is logarithmically and which is linearly spaced
log_order = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# Some dummy data
y = np.sin(x)

# Create a figure
fig, ax = plt.subplots(1,1)

# Some ticks
xTicks = np.arange(x.shape[0])
print(xTicks)

loglocator = LogLocator()
print(loglocator)

# Evenly spaced ticks on the x-axis
ax.set_xticks(xTicks)

# ticks are labeled with the actual value
ax.set_xticklabels(x)

# Plot the data over the ticks
plt.plot(xTicks, y)

# Activate grid
plt.grid()

# Show figure
plt.show()