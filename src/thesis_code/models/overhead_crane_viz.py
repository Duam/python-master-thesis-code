#!/usr/bin/python3

##
# @file overhead_crane_viz.py
# @author Paul Daum
# @date 2018/12/01
##

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation


class OverheadCrane_Patch:
  # @brief Constructor.
  def __init__(self, x0: np.array):
    # Lenghts and heights
    rl = 2.5  # Half of rail length
    rh = 0.15 # Half of rail bar height
    ch = 0.1  # Half of cart height
    cw = 0.25 # Half of cart width
    br = 0.05 # Ball radius
    self.line_margin = 0.2
    self.line_height = 0.2

    self.pos_cart = x0[0]
    self.vel_cart = x0[1]
    self.len_cable = x0[2]
    self.ang_cable = x0[3]

    # Initialize patches
    self.rail_patch = PatchCollection([
      Polygon([[-rl,-rh], [-rl,rh]]),
      Polygon([[-rl,0.0], [rl,0.0]]),
      Polygon([[rl,-rh],  [rl,rh]])
    ], linewidth=3, color='black')
    self.rail_patch.set_offset_position('data')

    self.cart_patch = PatchCollection([
      Rectangle([-cw,-ch], cw*2, ch*2, facecolor='grey', edgecolor='black'),
      Circle([0.0,0.0], 5*ch/8, facecolor='w', edgecolor='black', linewidth=1),
      Circle([0.0,0.0], 3*ch/8, facecolor='w', edgecolor='black', linewidth=1)
    ], True)
    self.cart_patch.set_offset_position('data')

    self.cable_patch = Polygon([[0.0,0.0], [0.0,-1.0]], linewidth=1, color='black')
    self.ball_patch = Circle([0.0,-1.0], br, facecolor='grey', edgecolor='black')

  # @brief Updates the patch
  def update(self, x: np.array):
    # Fetch states
    self.pos_cart = x[0]
    self.vel_cart = x[1]
    self.len_cable = x[2]
    self.ang_cable = x[3]

    # Compute ball position
    xpos_ball = self.pos_cart - self.len_cable * np.sin(self.ang_cable)
    ypos_ball = self.len_cable * np.cos(self.ang_cable)

    # Update patches
    self.cart_patch.set_offsets([self.pos_cart, 0.0])
    self.ball_patch.center = xpos_ball, ypos_ball
    self.cable_patch.set_xy([
      [self.pos_cart,0.0], 
      [xpos_ball, ypos_ball]
    ])


##
# @brief A visualizer for the overhead crane
##
class OverheadCrane_Visualizer:
  # @brief Constructor.
  def __init__(self, Xs, Rs = None, dt = 0.01, figlabel = 'OverheadCrane'):
    # Our figure
    self.fig = plt.figure(figlabel)
    # Internal values
    self.dt = dt          # Animation sample time
    self.Xs = Xs          # State trajectory
    self.Rs = Rs          # Reference trajectory
    self.N = Xs.shape[1]  # Number of samples

  # @brief Creates the animation and returns it to the user.
  def createAnimation(self):
    # Compute RMs error
    if hasattr(self, 'Rs'):
      self.rms = list()
      for k in range(self.Xs.shape[1]):
        x_c = self.Xs[0,k]    # Horizontal cart position
        x_l = self.Xs[2,k]    # Cable length
        theta = self.Xs[4,k]  # Cable angle      
        bx = x_c - x_l * np.sin(theta) # Ball x position
        by = x_l * np.cos(theta)       # Ball y position
        r = self.Rs[k,:]
        y = [bx, by]
        self.rms += [
          np.sqrt(
            (y[0] - r[0])**2 +
            (y[1] - r[1])**2
          )
        ]

    # Create an animation
    anim = FuncAnimation(
      self.fig, 
      self.updateAnimation, 
      frames=range(self.Xs.shape[1]),
      init_func=self.initAnimation,
      interval = 1e3 * self.dt
    )

    return anim


  ##
  # @brief Initialization function for the animation. Is passed to the FunctAnimation object.
  def initAnimation(self):
    ##############################################################
    ###                     CRANE ANIMATION                    ###
    ##############################################################
    self.cranePlot = plt.subplot(2,1,1)
    self.cranePlot.set_xlim([-3.0,3.0])
    self.cranePlot.set_ylim([-3.0, 0.5])

    # Add the overhead crane patch
    self.cranePatch = OverheadCrane_Patch(self.Xs[:,0])
    self.cranePlot.add_collection(self.cranePatch.rail_patch)
    self.cranePlot.add_collection(self.cranePatch.cart_patch)
    self.cranePlot.add_patch(self.cranePatch.ball_patch)
    self.cranePlot.add_patch(self.cranePatch.cable_patch)

    # Reference
    if hasattr(self, 'Rs'):
      self.reference = Polygon(self.Rs, linestyle = '--', fill = False)
      self.cranePlot.add_patch(self.reference)

    # Trajectory
    self.trajectory = Polygon([[0,-1]], linestyle = '-', fill = False)
    self.cranePlot.add_patch(self.trajectory)

    ##############################################################
    ###                   REFERENCE MISMATCH                   ###
    ##############################################################
    self.errPlot = plt.subplot(2,1,2)
    self.errPlot.set_xlim([0,1])
    plt.xlabel('t [%]')
    plt.ylabel('rms')

    return self.cranePlot, self.errPlot

  def updateAnimation(self, i):
    # Update crane plot
    self.cranePlot = plt.subplot(2,1,1)
    #plt.plot(b_x, b_y, '.', color='black')
    self.cranePatch.update(self.Xs[:,i])

    # Update RMS error
    if hasattr(self, 'Rs'):
      # Get the plot handle
      self.errPlot = plt.subplot(2,1,2)
      self.errPlot.clear()
      self.errPlot.set_yscale('log')
      self.errPlot.set_xlim([0,(self.N+1)*self.dt])
      self.errPlot.set_ylim([0,np.max(self.rms)])
      
      # Get the rms
      N = self.Rs.shape[0]
      rms = np.zeros((N))
      rms[:i] = np.array(self.rms[:i])
    
      # Plot the rms
      tAxis = np.linspace(0,(N+1)*self.dt,N)
      plt.plot(tAxis[:i], rms[:i], '-', color='b')

    return self.cranePlot, self.errPlot

  


  