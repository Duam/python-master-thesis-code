#!/usr/bin/python3

""" File: carousel_greybox.py
Author: Paul Daum
Date: 2019/03/11
This file contains the carousel greybox model with nonlinear addition from
Jonas Schlagenhauf's master thesis.
"""

import numpy as np
import casadi as cas

class Carousel:

  """ Carousel model
  Jonas Schlagenhauf's greybox model in continuous time with (optional) nonlinear addition.
  The carousel turns at a constant rate of 2rad/s.
  Describes the system dynamics
    xdot = A_sq * x^2 + A_lin * x + A_aff + B * u
    y = C * x + D * u
  """

  def __init__(self, subsamples:int = 10, simple_disturbance: bool = True):
    """ Initializes the carousel greybox model.
    Args:
      subsamples[int] -- Number of subsamples used in nonlinear system discretization
      simple_disturbance[bool] -- Uses the simple disturbance model if set (default True).

    """
    self._nx = 4
    self._nu = 1
    self._ny = 2

    # Define system matrices:
    # Nonlinear part
    self._A_sq = cas.DM([
      [  7.4516, -0.9472,  11.1731, -0.3881],
      [-27.9358,  5.5879, -73.4288,  1.1341],
	    [-5.1224,   0.7672, -11.0919,  0.1592],
	    [-12.1309, -0.0159,  11.0474,  0.3217]
    ])

    # Linear part
    self._A_lin = cas.DM([
      	[ 0,      1,       0,    0    ],
	      [-7.647, -3.219,  14.14, 1.691],
	      [ 0,      0,       0,    1    ],
	      [ 6.094,  2.194, -21.42, 1.674]
    ])

    # Affine part
    self._A_aff = cas.DM([
      [ 0.4905], 
      [-2.2357], 
      [-0.3410], 
      [ 0.7161]
    ])

    # Control matrix
    self._B = cas.DM([
      [ 0     ],
      [-0.1286],
      [ 0     ],
      [ 1.182]
    ])

    # Output matrix
    self._C = cas.DM([
      [1, 0, 0, 0],
  		[0, 0, 1, 0]
    ])

    # Feedforward matrix
    self._D = cas.DM([
      [0],
      [0]
    ])

    # Create system
    x_sym = cas.SX.sym('x',self._nx)
    u_sym = cas.SX.sym('u',self._nu)

    # Create system expression and function (nonlinear)
    xdot_sym  = cas.mtimes([self._A_sq, x_sym**2]) 
    xdot_sym += cas.mtimes([self._A_lin, x_sym]) 
    xdot_sym += self._A_aff 
    xdot_sym += cas.mtimes([self._B, u_sym])
    self._f_nonlin = cas.Function('f', [x_sym,u_sym], [xdot_sym], ['x','u'], ['xdot'])
    
    # Create jacobian functors for linearization
    self._A_linearized_fun = cas.Function('A_linearized_fun', [x_sym,u_sym], [cas.jacobian(xdot_sym,x_sym)])
    self._B_linearized_fun = cas.Function('B_linearized_fun', [x_sym,u_sym], [cas.jacobian(xdot_sym,u_sym)])    
      

  def ode(self, x, u):
    return self._f_nonlin(x,u)


  def ode_discr(self, x, u, dt, nn=10):
    # Discretize nonlinear system using RK4
    h = dt/float(nn)
    xnext = x
    for k in range(nn):
      k1 = self._f_nonlin(xnext, u)
      k2 = self._f_nonlin(xnext + h/2 * k1, u)
      k3 = self._f_nonlin(xnext + h/2 * k2, u)
      k4 = self._f_nonlin(xnext + h * k3, u)
      xnext = xnext + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xnext.full()


  def ode_linearized(self, x_linp, u_linp, x, u):
    # Compute offset and jacobians at the linearization point
    offs = self._f_nonlin(x_linp, u_linp)
    A = self._A_linearized_fun(x_linp, u_linp).full()
    B = self._B_linearized_fun(x_linp, u_linp).full()

    # Compute linearized dynamics
    return (offs + cas.mtimes([A, x - x_linp]) + cas.mtimes([B, u - u_linp])).full()


  def ode_linearized_discr(self, x_linp, u_linp, x, u, dt):
    from scipy.linalg import expm, inv

    # Compute offset and jacobians at the linearization point
    offs = self._f_nonlin(x_linp, u_linp)
    A = self._A_linearized_fun(x_linp, u_linp).full()
    B = self._B_linearized_fun(x_linp, u_linp).full()

    # Augment A, B, x for discretization
    B_aug = cas.blockcat([[B, offs]]).full()
    u_aug = cas.blockcat([[u],[cas.DM.ones(1,1)]]).full()

    # Compute discretized affine-linear system
    exp = expm(A * dt)
    I = np.eye(exp.shape[0],exp.shape[1])
    return (cas.mtimes([exp, x]) + cas.mtimes([inv(A), (exp - I), B_aug, u_aug])).full()


  def out(self, x, u):
    return (cas.mtimes([self._C, x_sym]) + cas.mtimes([self._D, u_sym])).full()

