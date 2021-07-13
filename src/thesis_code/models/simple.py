#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/05/15
  Brief: TODO
"""

import casadi

class Simple:

  """ A simple test system: 1-dimensional discrete PT1 element with disturbed outputs """

  def __init__(self, use_casadi:bool=True):
    # Set system size
    self._nx = 1 # States
    self._nu = 1 # Controls
    self._ny = 1 # Measurements
    self._np = 1 # External disturbances
    self._nv = 1 # Measurement noise

    # Create system
    self._fun_xnext = lambda x,u,p: 0.6 * x + u
    self._fun_out = lambda x,u,p,v: x + u + p + v

    # Set casadi flag
    self._casadi = use_casadi

    if use_casadi:
      # Create symbolics
      self._x_sym = casadi.SX.sym('x',self._nx)
      self._u_sym = casadi.SX.sym('u',self._nu)
      self._p_sym = casadi.SX.sym('p',self._np)
      self._v_sym = casadi.SX.sym('v',self._nv)
      self._y_sym = casadi.SX.sym('y',self._ny)

      # Create symbolic system
      self._xnext_sym = self._fun_xnext(self._x_sym, self._u_sym, self._p_sym)
      self._y_sym = self._fun_out(self._x_sym, self._u_sym, self._p_sym, self._v_sym)

      # Create functions
      self._fun_xnext_sym = casadi.Function('F', 
        [self._x_sym, self._u_sym, self._p_sym], [self._xnext_sym],
        ['x','u','p'], ['xnext'])
      self._fun_out_sym = casadi.Function('g',
        [self._x_sym, self._u_sym, self._p_sym, self._v_sym], [self._y_sym],
        ['x','u','p','v'], ['y'])

  def xnext(self, x, u, p):
    return self._fun_xnext_sym(x,u,p) if self._casadi else self._fun_xnext(x,u,p)

  def out(self, x, u, p, v):
    return self._fun_out_sym(x,u,p,v) if self._casadi else self._fun_out(x,u,p,v)