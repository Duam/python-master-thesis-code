#!/usr/bin/python3

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Position and intensity of the source
s = np.array([0,0])
I = 1.0

##
# @brief Computes the suqared distance between two points
# @param a The first point
# @param b The second point
# @return The squared distance
##
def dist2(a,b):
  diff = a - b
  return np.dot(np.transpose(diff), diff)

def normalize(a):
  origin = np.zeros(a.shape)
  mag = np.sqrt(dist2(a,origin))
  if mag == 0:
    return np.zeros(a.shape)

  return a / mag

##
# @brief Computes an element of the vector field
# @param t The current time
# @param p The 2-D position
# @return A vector
##
def computeVectorAt(t, p):
  global s, I
  direction = p - s
  dist = np.sqrt(dist2(p,s))
  if dist == 0:
    return np.zeros(p.shape)

  return direction * I / (dist**3)

# Evaluate the vector field over some range
N = 21
pmin = -2
pmax = 2
xAxis = np.linspace(pmin, pmax, N)
yAxis = np.linspace(pmin, pmax, N)

grid = np.array([np.array([x,y]) for x,y in itertools.product(xAxis,yAxis)])
V    = np.array([computeVectorAt(0,p) for p in grid])

# Calculate the magnitudes of the vectors for color mapping
M    = np.sqrt(V[:,0]*V[:,0] + V[:,1]*V[:,1])

# Normalize the vector lengths
V    = np.array([normalize(v) for v in V])

# Plot the vector field
fig = plt.figure()
ax = fig.gca()
qq = plt.quiver(grid[:,0], grid[:,1], V[:,0], V[:,1], M, pivot='tail', cmap=plt.cm.jet)
plt.colorbar(qq, cmap=plt.cm.jet)
plt.show()