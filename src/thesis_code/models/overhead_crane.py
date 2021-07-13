#!/usr/bin/python3

##
# @file overhead_crane.py
# @author Paul Daum
# @date 2018/11/27
##

import casadi as cas

## 
# @brief An overhead crane model
# 
# This class represents the model of the overhead crane 
# from "Introducing a Model-Based Learning Control Software
# for Nonlinear Systems: RoFaLT" used to demonstrate
# periodic tracking problems.
##
class OverheadCrane:

  ##
  # @brief Constructor
  # @param tau Time constant
  # @param A Input gain
  ##
  def __init__(self, time_const = 3e-2, input_gain = 5e-2):
    self.tau = time_const
    self.A = input_gain
    self.g = 9.81
    self.x0 = cas.vertcat([
      0.0, 0.0, # Cart position & velocity
      1.0, 0.0, # Cable length & velocity
      0.0, 0.0  # Cable angle & angular velocity
    ])

  ##
  # @brief The model dynamics
  # @param x The current state (6-vector)
  # @param u The current control (2-vector)
  # @return The state derivative (6-vector)
  ##
  def ode(self, x, u):
    # States
    x_c = x[0]    # Cart horizontal position
    v_c = x[1]    # Cart horizontal velocity
    x_l = x[2]    # Cable length
    v_l = x[3]    # Cable roll-out/in velocity
    theta = x[4]  # Cable angle
    omega = x[5]  # Cable angular velocity

    # Controls
    u_c = u[0]    # Cart voltage
    u_l = u[1]    # Winch voltage

    # Intermediate value (used twice in return)
    #v_c_dot = - 1/self.tau * (v_c - self.A * u_c)
    v_c_dot = u_c
    #v_l_dot = - 1/self.tau * (v_l - self.A * u_l)
    v_l_dot = u_l

    # Return state derivative
    return cas.vertcat(
      v_c,
      v_c_dot,
      v_l,
      v_l_dot,
      omega,
      + 1/x_l * (v_c_dot * cas.cos(theta) + self.g * cas.sin(theta) + 2 * v_l * omega)
    )

  ##
  # @brief The model output
  # @param x The current state (6-vector)
  # @param u The current control (2-vector)
  # @return The ball position (2-vector)
  ##
  def output(self, x, u):
    # Fetch states
    x_c = x[0]
    x_l = x[2]
    theta = x[4]

    # Return model outputs
    return cas.vertcat(
      x_c + x_l * cas.sin(theta),
      x_l * cas.cos(theta)
    )

