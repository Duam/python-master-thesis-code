from casadi import *
import numpy as np
from thesis_code.utils.ParametricNLP import ParametricNLP
from typing import Callable


class NMPC:
  """ A simple nonlinear model predictive controller for
  path following.
  Given a target state trajectory (x(0), ..., x(N)),
  a target control trajectory (u(0), ..., u(N-1)) and
  an estimate of the current state (xest), where
  the index 0 indicates the current time, it tries to 
  minimize the deviation from said targets while 
  respecting the state evolution constraints.
  Additionally, it accepts equality and inequality
  constraints.
  The function signature is:
            (x, u) = NMPC (xest, xref, uref)
  """

  def __init__(self, F: Callable[[SX,SX],SX],
                     N: int,
                     nx: int,
                     nu: int,
                     Rx: Callable[[SX],SX] = None,
                     nrx: int = 0,
                     Ru: Callable[[SX],SX] = None,
                     nru: int = 0
                     ):
    """

    """
    print("=====================================")
    print("===         Creating NMPC         ===")
    print("=====================================")

    # If no reference function is given, create dummy ones
    if nrx == 0: 
      x = SX.sym('x', nx, 1)
      nrx = nx
      Rx = Function('Rx', [x], [x])
    if nru == 0:
      u = SX.sym('u', nu, 1)  
      nru = nu
      Ru = Function('Ru', [u], [u])

    # Check sizes
    assert F.n_in() == 2, "F: incorrect number of inputs (must be 2)!"
    assert F.n_out() == 1, "F: incorrect number of outputs (must be 1)!"
    F.assert_size_in(0, nx, 1)
    F.assert_size_in(1, nu, 1)
    F.assert_size_out(0, nx, 1)

    assert Rx.n_in() == 1, "Rx: incorrect number of inputs (must be 1)!"
    assert Rx.n_out() == 1, "Rx: incorrect number of outputs (must be 1)!"
    Rx.assert_size_in(0, nx, 1)
    Rx.assert_size_out(0, nrx, 1)

    assert Ru.n_in() == 1, "Fr: incorrect number of inputs (must be 1)!"
    assert Ru.n_out() == 1, "Fr: incorrect number of outputs (must be 1)!"
    Ru.assert_size_in(0, nu, 1)
    Ru.assert_size_out(0, nru, 1)

    # Fetch parameters
    self.F = F
    self.Rx = Rx
    self.Ru = Ru
    self.N  = N
    self.nx = nx
    self.nu = nu
    self.nrx = nrx
    self.nru = nru

    # Parametric NLP (later QP)
    self.nlp = ParametricNLP("PF_NMPC")

    # Add decision variables and parameters
    self.nlp.add_decision_var('X', (nx,N))
    self.nlp.add_decision_var('U', (nu,N-1))
    self.nlp.add_parameter('xest', (nx,1))
    self.nlp.add_parameter('xref', (nrx,N))
    self.nlp.add_parameter('uref', (nru,N-1))

    # Create aliases for easy access
    self.X    = self.nlp.get_decision_var('X')
    self.U    = self.nlp.get_decision_var('U')
    self.xest = self.nlp.get_parameter('xest')
    self.xref = self.nlp.get_parameter('xref')
    self.uref = self.nlp.get_parameter('uref')

    # Debug output
    print("=====================================")
    print("===          NMPC created         ===")
    print("=====================================")


  def init(self, Q: np.array, R: np.array, P: np.array):
    """
    """
    print("=====================================")
    print("===       Initializing NMPC       ===")
    print("=====================================")

    # Check weight matrix size
    assert Q.shape == (self.nrx,self.nrx), "Q: incorrect shape!"
    assert R.shape == (self.nru,self.nru), "R: incorrect shape!"
    assert P.shape == (self.nrx,self.nrx), "P: incorrect shape!"

    # Create cost function
    J = 0
    for k in range(self.N-1):
      dRx = self.Rx(self.X[:,k]) - self.xref[:,k]
      dRu = self.Ru(self.U[:,k]) - self.uref[:,k]
      J += 0.5 * mtimes([ dRx.T, Q, dRx ])
      J += 0.5 * mtimes([ dRu.T, R, dRu ])

    dRx = self.Rx(self.X[:,self.N-1]) - self.xref[:,self.N-1]
    J += mtimes([ dRx.T, P, dRx ])

    # Set cost function
    self.nlp.set_cost(J)

    # Create initial state constraint
    self.nlp.add_equality('init', self.X[:,0] - self.xest)

    # Create shooting constraints
    shoot = SX.sym('shoot',self.nx,self.N-1)
    for k in range(self.N-1):
      xnext = self.F(self.X[:,k], self.U[:,k])
      shoot[:,k] = self.X[:,k+1] - xnext
    
    self.nlp.add_equality('shoot', shoot)

    # Specify solver and options
    nlpsolver = 'ipopt'
    opts = {
      'ipopt.print_user_options': 'no',
      'ipopt.print_info_string': 'no',
      'print_time': 0,
      'ipopt.print_level': 0,
      'ipopt.sb': 'yes',
      'ipopt.max_iter': 1000,
      'ipopt.constr_viol_tol': 1e-8,
      'ipopt.tol': 1e-8
    }

    # Define callback function
    def iterCb(i, sol): 
      pass

    # Initialize solver
    self.nlp.init(nlpsolver, opts, iterCb)

    # Debug output
    print("=====================================")
    print("===   NMPC initialization done    ===")
    print("=====================================")
    

  def call(self, xest: np.array, 
                 xref: np.array = None, 
                 uref: np.array = None ):
    """
    """
    # Check sizes
    assert xest.shape == (self.nx,), "xest: incorrect shape!"
    if type(xref) != None: assert xref.shape == (self.nrx,self.N), "xref: incorrect shape!"
    if type(uref) != None: assert uref.shape == (self.nru,self.N-1), "uref: incorrect shape!"

    # Set parameters
    param = self.nlp.struct_p(0)
    param['xest'] = DM( xest )
    param['xref'] = DM.zeros((self.nrx,N))   if type(xref) == None else DM( xref )
    param['uref'] = DM.zeros((self.nru,N-1)) if type(uref) == None else DM( uref )

    # Set initial guess (TODO: improve guess. shift?)
    winit = self.nlp.struct_w(0)
    winit['X'] = DM.zeros( (self.nx,self.N) )
    winit['U'] = DM.zeros( (self.nu,self.N-1) )

    # Optimize and return!
    result, stats = self.nlp.solve(winit, param)
    X_sol = result['w']['X'].full()
    U_sol = result['w']['U'].full()
    return X_sol, U_sol, stats, result
