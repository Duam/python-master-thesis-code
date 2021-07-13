#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,37)
y = np.arange(0,55)

xAxis, yAxis = np.meshgrid(x,y)
zData = xAxis**2 + yAxis

print('xAxis = ', xAxis, 'shape = ', xAxis.shape)
print('yAxis = ', xAxis, 'shape = ', yAxis.shape)
print('zData = ', zData, 'shape = ', zData.shape)

plt.figure()
cs = plt.contour(xAxis, yAxis, zData)
cb = plt.colorbar(cs)

plt.show()