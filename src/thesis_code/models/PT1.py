#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/05/16
  Brief: A simple PT1 element
"""

class PT1:

  """ A simple scalar PT1 element. Can be configured with time constant T and gain K """

  def __init__(self, T:float, K:float=1.0):
    """ Initializes the PT1 element.
    args:
      T[float] -- Time constant
      K[float] -- Gain
    """
    self._T = T
    self._K = K

  def ode(self, x:float, u:float):
    """ The PT1 ODE equation
    args: 
      x[float] -- Current value
      u[float] -- Input
    returns:
      xdot[float] -- The response slope
    """
    return (self._K * u - x) / float(self._T)
  
  def ode_discr(self, x:float, u:float, dt:float):
    """ Discrete PT1 equation
    args:
      x[floaŧ] -- Current value
      u[float] -- Input
    returns:
      xnext[float] -- Next value
    """
    return 1/((self._T/float(dt))+1)  * (self._K * u - x) + x