#!/usr/bin/python3

import casadi as cas

##
# @brief A simple pendulum model (undamped)
# @states 1. pendulum angle
#         2. pendulum angular velocity
# @controls 1. Force on pendulum tip in x-direction
#           2. Force on pendulum tip in y-direction
# @outputs 1. pendulum angle
#          2. pendulum angular velocity
##
class Pendulum:

  ##
  # @brief Initialiation
  # @param name The name of the pendulum
  # @param l The length of the pendulum
  # @param m The mass of the pendulum
  ##
  def __init__(self, name = "Pendulum", l = 1, m = 1):
    self.name = name
    self.L = l      # length in m
    self.m = m      # weight in kg
    self.g = 9.81   # grav. accel in m/s^2

  ## 
  # @brief Returns the x-y-position of the pendulum tip
  # @param theta
  # @return tip position
  ##
  def getTipPosition(self, theta):
    xPos = self.L * np.sin(theta)
    yPos = self.L * np.cos(theta)
    return xPos, yPos


  ##
  # @brief The model dynamics of the pendulum
  # @param x The current state:
  #          theta = angle difference from hanging down
  #          angVel = angular velocity
  # @param u The force on the pendulum mass
  #          u_x = Force in horizontal direction
  #          u_y = Force in vertical direction
  # @param w Process noise
  #          0: Noise on angular velocity
  #          1: Noise on angular acceleration
  # @return The state derivative
  #          angVel = angular velocity
  #          angAcc = angular acceleration
  ##
  def ode(self, x, u, w):
    # Fetch states
    theta = x[0]
    angVel = x[1]

    # Fetch controls
    F_x = u[0]
    F_y = u[1]

    # Fetch process noise
    angVel_noise = w[0]
    angAcc_noise = w[1]

    # Acceleration perpendicular to the pendulum
    # due to gravity
    g_perp = cas.sin(-theta) * self.g

    # Acceleration perpendicular to the pendulum
    # due to external force
    F_x_perp = cas.cos(-theta) * F_x
    F_y_perp = cas.sin(theta) * F_y
    a_perp = g_perp + (F_x_perp + F_y_perp) / self.m

    # Angular acceleration
    angAcc = a_perp * self.L

    # Return state derivative
    return cas.vcat([angVel + angVel_noise, angAcc + angAcc_noise])

  ##
  # @brief The output function of the pendulum
  # @param x The current state:
  #          angle (measured from hanging down)
  #          angular velocity
  # @param u The force on the pendulum mass
  #          x-Force
  #          y-Force
  # @param w Process noise
  #          0: Noise on angular velocity
  #          1: Noise on angular acceleration
  # @return The measured output:
  #         angle
  #         angular velocity
  ##
  def output(self, x, u, w):
      return x

  ##
  # @brief A function returning the pendulum parameters as a string
  # @return The parameters in a string
  ##
  def toString(self):
    infoStr = self.name + ": l = " + str(self.L) +  ", m = " + str(self.m)
    return infoStr


##
# Execute this file to test the pendulum model
##
if __name__ == '__main__':

  import sys, os
  sys.path.append(os.path.realpath('/'))
  sys.path.append(os.getcwd())

  import numpy as np
  import matplotlib.pyplot as plt
  from integrators.rk4step import rk4step
  
  print("Pendulum Test started.")

  pendulum = Pendulum("p0")
  print(pendulum.toString())

  # Simulation parameters
  T = 10.0 # seconds
  N = 100 # samples
  dt = T/N # Sample time
  nn = 10 # integrator steps per step
  h = dt/nn 

  # Initial state
  x0 = cas.vertcat([-np.pi/2.0, 0])

  # Create system model in CasADi
  x = cas.MX.sym('x', 2, 1)
  u = cas.MX.sym('u', 2, 1)
  f = cas.Function('f', [x,u], [pendulum.ode(x,u)], ['x','u'], ['xdot'])

  # Create an integrator
  Xk = x
  for k in range(nn):
      Xk = rk4step(f, Xk, u, h)
  F = cas.Function('F', [x,u], [Xk], ['x','u'], ['xnext'])

  # Choose controls
  Us = cas.DM.zeros((2, N))

  # Start the simulation
  Xs_sim = cas.DM.zeros((2,N+1))
  Xs_sim[:,0] = x0
  for k in range(N):
    Xs_sim[:,k+1] = F(Xs_sim[:,k], Us[:,k])
    
  Xs_sim = Xs_sim.full()

  # Prepare plotting
  tAxis = np.linspace(0, T, N+1)
  plt.figure(1)

  # Plot
  plt.subplot(211)
  plt.plot(tAxis, Xs_sim[0,:])
  plt.ylabel('Angle [rad]')
  plt.xlabel('t [s]')

  plt.subplot(212)
  plt.plot(tAxis, Xs_sim[1,:])
  plt.ylabel('Angular velocity [rad/s]')
  plt.xlabel('t [s]')

  plt.show()
