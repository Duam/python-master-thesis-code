#!/usr/bin/python3

#####################################
### @file RefgenNLP.py
### @author Paul Daum
### @date 04.12.2018
### @brief This class implements a 
### generic reference generator using
### optimal control techniques
#####################################

import numpy as np

# CasADi is used for algorithmic differentiation and creating the NLP
from casadi import *
from casadi.tools import struct_symSX, entry

# The generic parametric NLP class
from thesis_code.utils.ParametricNLP import ParametricNLP

# For argument type hints
from typing import Callable, Any, Iterable

class RefGenNLP:
  """ A simple target selector.
  Given an output reference sequence (y(0), ..., y(N-2)), its purpose 
  is to generate a state sequence (x(0), ..., x(N-1)) and an input sequence 
  (u(0), ..., u(N-2)) that will prodces the desired output reference. 
  """

  ##
  # @brief Creates the reference generator
  # @param f The discretized ode x(k+1) = f(x(k),u(k))
  # @param h The output function y(k) = h(x(k),u(k))
  # @param N The number of samples
  # @param nx The number of states
  # @param nu The number of controls
  # @param ny The number of outputs
  def __init__(self, F: Callable[[SX,SX],SX],
                     g: Callable[[SX,SX],SX],
                     N: int,
                     nx: int,
                     nu: int,
                     ny: int):
    
    """

    """

    print("=====================================")
    print("===       Creating RefGenNLP      ===")
    print("=====================================")

    #############################################
    ###            FETCH PARAMETERS           ###

    # Input sizes
    self.nx = nx
    self.nu = nu
    self.ny = ny

    # Sample number
    self.N = N

    print("Parameters fetched")

    ################################################
    ###        PREPARE CASADI EXPRESSIONS        ###

    # Create casadi variables
    x = SX.sym('x', nx, 1)
    u = SX.sym('u', nu, 1)
    y = SX.sym('y', ny, 1)

    # Create handles for the user to add constraints manually
    self.x = x
    self.u = u
    self.y = y

    # Discretized ODE & Output function
    self.F = Function('F', [x,u], [F(x,u)], ['x','u'], ['xnext'])
    self.g = Function('g', [x,u], [g(x,u)], ['x','u'], ['y'])

    print("Casadi objects created.")

    ################################################
    ###          OPTIMIZATION VARIABLES          ###

    # Create a parametric NLP
    self.nlp = ParametricNLP('ReferenceGenerator')

    # Add decision variables
    self.nlp.add_decision_var('X', (self.nx,self.N))
    self.nlp.add_decision_var('U', (self.nu,self.N-1))

    # Add parameters
    self.nlp.add_parameter('Y', (self.ny,self.N-1))
    
    # Create aliases for easy access
    self.X = self.nlp.get_decision_var('X')
    self.U = self.nlp.get_decision_var('U')
    self.Y = self.nlp.get_parameter('Y')

    print("Optimization variables created.")
    print("=====================================")
    print("===       RefGenNLP created       ===")
    print("=====================================")


  ##
  # @brief Initializes the reference generator.
  # Should be called after the object was constructed
  # and the user provided custom constraints
  # @param Q Trajectory deviation penalty matrix
  # @param R Control penalty matrix
  # (typically zero)
  def init(self, Q: DM, R: DM):

    print("=====================================")
    print("===     Initializing RefGenNLP    ===")
    print("=====================================")

    ################################################
    ###             FETCH PARAMETERS             ###

    # Check matrix sizes
    if Q.rows() != self.ny or Q.columns() != self.ny:
      raise ValueError("Q has incorrect sizes!")

    if R.rows() != self.nu or R.columns() != self.nu:
      raise ValueError("R has incorrect sizes!")

    # Fetch matrices
    self.Q = Q
    self.R = R

    ################################################
    ###          CREATE COST FUNCTION            ###

    self.nlp.set_cost(self.createCostFunction())
    print("Cost function created.")

    ################################################
    ###           CREATE CONSTRAINTS             ###

    self.createShootingConstraints()
    print ("Shooting constraints created.")

    ################################################
    ###            CREATE NLP SOLVER             ###

    # Specify solver
    nlpsolver = 'ipopt'

    # Specify solver options
    opts = {}
    opts['ipopt.print_info_string'] = 'no'
    opts['ipopt.print_level'] = 0
    opts['ipopt.print_info_string'] = 'no'
    opts['ipopt.sb'] = 'yes'
    opts['print_time'] = 0
    opts['ipopt.max_iter'] = 1000
    opts['ipopt.constr_viol_tol'] = 1e-8
    opts['ipopt.tol'] = 1e-16

    # Initialize solver
    def iterCb(i,sol):
      w = sol['x']
      lam_g = sol['lam_g']
      #KKT = self.nlp.evalKKT(w, lam_g).full()
      #print("Computing eigenvalues of KKT matrix..")
      #eigvals, eigvecs = numpy.linalg.eig(KKT)
      #eigvals_norm = [numpy.linalg.norm(lam) for lam in eigvals]
      print('\nIteration', i)
      #g = self.nlp.evalG(w)
      #print(g)
      #ind_max_g = numpy.argmax(g)
      #ind_min_g = numpy.argmin(g)
      #print('Largest g at ', ind_max_g, '=', g[ind_max_g])
      #print('Lowest g at', ind_min_g, '=', g[ind_min_g])
      #print('Shape:', KKT.shape)
      #print('Largest Eigenvalue =', max(eigvals_norm))
      #print('Lowest Eigenvalue =', min(eigvals_norm))

    self.nlp.init(nlpsolver, opts, iterCb)

    print("NLP solver created.")
    print("=====================================")
    print("=== RefGenNLP initialization done ===")
    print("=====================================")


  ##
  # @brief Creates the cost functio
  # @return The cost function as a CasADi SX
  def createCostFunction(self):
    # Create cost term
    J = 0

    # Create running cost
    # = e^T Q e + u^T R u
    for k in range(self.N-1):
      xk = self.X[:,k]
      uk = self.U[:,k]
      yk = self.Y[:,k]

      # Add the output deviation penalty
      ek = yk - self.g(xk, uk)
      J += mtimes([ ek.T, self.Q, ek ])

      # Add the control penalty
      J += mtimes([ uk.T, self.R, uk ])

    # Pseudo-measurement for stage N
    #xN = self.X[:,self.N]
    #uN = self.U[:,0]
    #yN = self.Y[:,0]
    #eN = yN - self.g(xN, uN)
    #J += mtimes([ eN.T, self.Q, eN ])

    # Return the total cost
    return J
  

  ##
  # @brief Creates the shooting constraints
  def createShootingConstraints(self):
    shoot = SX.sym('shoot', self.nx, self.N-1)
    # Create shooting constraints
    for k in range(self.N-1):
      xk = self.X[:,k]
      uk = self.U[:,k]
      xnext = self.X[:,k+1]
      shoot[:,k] = xnext - self.F(xk,uk)
    
    self.nlp.add_equality('shoot', shoot )

  ##
  # @brief User function to add new path constraints
  # @param expr A functor g(i) that returns the equality
  # constraing g = 0. g(i) must utilize the casadi 
  # symSX_struct objects defined in the RefGenNLP, such as
  # self.shooting['X',i] or self.shooting['U',i]
  def add_equality(self, name:str, expr:SX):
    self.nlp.add_equality(name, expr)


  ##
  # @brief Computes reference states and controls. Has to
  # be called after init().
  # @param yref The output reference to be tracked
  # @return xref, uref, solver_stats
  def run(self, yref: np.array, xinit: np.array):

    print("RefGenNLP called.")

    ################################################
    ###       SET OPTIMIZATION VARIABLES         ###

    assert yref.shape == (self.ny,self.N-1), "yref: incorrect shape!"
    assert xinit.shape == (self.nx,self.N), "xinit: incorrect shape!"

    # Set the optimization parameters
    param = self.nlp.struct_p(0)
    param['Y']  = DM(yref)

    # Set the initial guess
    winit = self.nlp.struct_w(0)
    winit['X'] = DM(xinit)
    winit['U'] = DM.zeros((self.nu, self.N-1))

    ################################################
    ###            SOLVE THE PROBLEM             ###

    # Optimize!
    result, stats = self.nlp.solve(winit, param)

    # Grab the solution and store it
    xs = result['w']['X'].full()
    us = result['w']['U'].full()

    print("RefGenNLP solved.")

    return xs, us, stats, result


###########################################################################
###                                                                     ###
###                         END OF REFGEN CLASS                         ###
###                                                                     ###
###########################################################################

##
# Test area to see if there are any errors
##
if __name__ == '__main__':

  # Create a test system
  nx = 1
  nu = 1
  ny = 1

  f = lambda x,u: x + u
  g = lambda x,u: x

  Q = blockcat([
    [1e-3]
  ])

  R = blockcat([
    [1e-6]
  ])

  S = DM.zeros((1,1))

  N = 10

  # Create a reference generator
  gen = RefGenNLP(f, g, N, nx, nu, ny)
  gen.init(Q,R,S)

  # Create a test reference trajectory
  yref = np.linspace(0, 0.9, N)
  
  x0 = np.array([0])

  # Run the reference generator
  xref, uref, stats = gen.run(x0, yref)

  import matplotlib.pyplot as plt
  plt.figure()

  tAxis = np.linspace(0, 1, N)
  plt.subplot(211)
  plt.plot(tAxis, xref)

  plt.subplot(212)
  plt.plot(tAxis, uref)

  plt.show()