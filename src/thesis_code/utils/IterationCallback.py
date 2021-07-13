#!/usr/bin/python3

"""
Author: Paul Daum
Date: 2019/01/10
This file implements an iteration callback function for usage in NLP solvers
"""

from casadi import *
from typing import Callable

# Adapted from CasADi callback example:
# http://casadi.sourceforge.net/api/html/d7/df0/solvers_2callback_8py-example.html
# Accessed 2019/01/10

class IterationCallback(Callback):
  """
  This class represents a callback function that is executed
  after each iteration of an optimization problem. 

  Initialization:
  name (string): Name of the object (For debugging)
  nx (int): The number of decision variables
  ng (int): The number of constraints
  np (int): The number of parameters
  cbfun (Callable[[int,dict],None]): User-provided callback function
  opts (dict): Options for the callback function (default = {})

  (Currently only tested using IpOpt)
  """
  def __init__(self, name: str, 
                     nx: int, 
                     ng: int, 
                     np: int, 
                     cbfun: Callable[[int,dict],None], 
                     opts: dict = {}):

    # Create the callback object
    Callback.__init__(self)

    # Store user-provided callback function
    self.cbfun = cbfun

    # Iteration count
    self.iter = 0

    # Fetch internal values (required by Callback object)
    self.nx = nx  # Number of decision variables
    self.ng = ng  # Number of constraints
    self.np = np  # Number of parameters

    # Initialize the callback object
    self.construct(name, opts)


  def eval(self, arg):
    """
    This is the function that the solver actually calls after each
    iteration. It calls the user-provided callback function
    with the current, intermediate result as a parameter. 
    The argument of the callback function is a dictionary
    with fields:
      'x': Decision variables
      'f': Cost function
      'g': Constraint function
      'lam_x': Multipliers of decision variables box constraints
      'lam_g': Multipliers of active constraints
      'lam_p': Multipliers of parameter constraints
    """

    # Put the intermediate solution into a dictionary
    res = {}
    for (i,s) in enumerate(nlpsol_out()): res[s] = arg[i]

    # Call the user-provided callback function
    self.cbfun(self.iter, res)
    self.iter += 1

    # Tell solver that everything is fine
    return [0]


  """
  The following methods are required for callback initialization 
  in the solver.
  """
  def get_n_in(self): return nlpsol_n_out()
  def get_n_out(self): return 1
  def get_name_in(self, i): return nlpsol_out(i)
  def get_name_ouot(self, i): return "ret"
  def get_sparsity_in(self, i):
    n = nlpsol_out(i)
    if n == 'f':
      return Sparsity.scalar()
    elif n in ('x', 'lam_x'):
      return Sparsity.dense(self.nx)
    elif n in ('g', 'lam_g'):
      return Sparsity.dense(self.ng)
    elif n in ('p', 'lam_p'):
      return Sparsity.dense(self.np)
    else:
      return Sparsity(0,0)


# Unit test
if __name__ == '__main__':

  # Create decision variables
  x = SX.sym('x')
  y = SX.sym('y')
  
  # Rosenbrock function
  f = (1-x)**2+100*(y-x**2)**2
  
  # Define the nlp
  nlp={'x':vertcat(x,y), 'f':f,'g':x+y}
  
  # Define the callback function
  def cbfun (i, sol):
    print('\nIteration', i, ': f =', sol['f'])

  # Create the iteration callback object
  callback = IterationCallback('callback', 2, 1, 0, cbfun)

  # Define solver options
  opts = {}
  opts['iteration_callback'] = callback # (!)
  opts['ipopt.tol'] = 1e-8
  opts['ipopt.max_iter'] = 50

  # Create the solver
  solver = nlpsol('solver', 'ipopt', nlp, opts)

  # Solve the problem
  sol = solver(lbx=-10, ubx=10, lbg=-10, ubg=10)