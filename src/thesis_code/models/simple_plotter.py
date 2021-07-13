#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from models.simple.simple import Simple

def plot(Xs, Us, Ys, Ps):

  # Create a figure with two subplots
  fig, axes = plt.subplots(2,1,sharex='col')

  # Populate the first subplot with state, control and output trajectories
  plt.sca(axes[0])
  plt.gca().ylabel(r'State $x$, Control $u$, Output $y$')
  plt.gca().plot(Xs, color='green', label='State',    linestyle='x-',  alpha=0.25)
  plt.gca().plot(Us, color='red',   label='Control',  linestyle='x--', alpha=0.25)
  plt.gca().plot(Ys, color='blue',  label='Output',   linestyle='x-')
  plt.legend(loc='best')

  # Populate the second subplot with parameters
  plt.sca(axes[1])
  plt.gca().ylabel(r'Parameter $p$')
  plt.gca().plot(Ps, linestyle='x-', color='blue', label='Simulated')
  plt.gca().legend(loc='best')

  return fig, axes

if __name__ == '__main__':
  pass