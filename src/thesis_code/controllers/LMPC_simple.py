from casadi import *
import numpy as np
from thesis_code.utils.ParametricNLP import ParametricNLP
from typing import Callable


class LMPC:
  """ A simple linear model predictive controller.
  Given a target state trajectory (x(0), ..., x(N)),
  a target control trajectory (u(0), ..., u(N-1)) and
  an estimate of the current state (xest), where
  the index 0 indicates the current time, it tries to 
  minimize the deviation from said targets while 
  respecting the (linear) state evolution constraints.
  Additionally, it accepts linear equality and inequality
  constraints.
  The function signature is:
            (dx, du) = LMPC (xest, xref, uref)
  The input applied to the system is then:
                   u(k) = uref(k) + du(k)
  """

  def __init__(self, F: Callable[[SX,SX],SX],
                     N: int,
                     nx: int,
                     nu: int):
    """

    """
    print("=====================================")
    print("===         Creating LMPC         ===")
    print("=====================================")

    # Check sizes
    assert F.n_in() == 2, "F: incorrect number of inputs (must be 2)!"
    assert F.n_out() == 1, "F: incorrect number of outputs (must be 1)!"
    F.assert_size_in(0, nx, 1)
    F.assert_size_in(1, nu, 1)
    F.assert_size_out(0, nx, 1)

    # Fetch parameters
    self.F = F
    self.N  = N
    self.nx = nx
    self.nu = nu

    # Create CasADi objects
    dx = SX.sym('x', nx, 1)
    du = SX.sym('u', nu, 1)
    xnext = F(dx,du)
    
    # Create linearization matrices as functions
    self.A = Function('A', [dx,du], [jacobian(xnext,dx)], ['dx','u'], ['A'])
    self.B = Function('B', [dx,du], [jacobian(xnext,du)], ['dx','u'], ['B'])

    # Parametric NLP (later QP)
    self.nlp = ParametricNLP("LMPC")

    # Add decision variables and parameters
    self.nlp.add_decision_var('dX', (nx,N))
    self.nlp.add_decision_var('dU', (nu,N-1))
    self.nlp.add_parameter('xest',  (nx,1))
    self.nlp.add_parameter('xref',  (nx,N))
    self.nlp.add_parameter('uref',  (nu,N-1))

    # Create aliases for easy access
    self.dX   = self.nlp.get_decision_var('dX')
    self.dU   = self.nlp.get_decision_var('dU')
    self.xest = self.nlp.get_parameter('xest')
    self.xref = self.nlp.get_parameter('xref')
    self.uref = self.nlp.get_parameter('uref')

    # Debug output
    print("=====================================")
    print("===          LMPC created         ===")
    print("=====================================")


  def init(self, Q: np.array, R: np.array):
    """
    """
    print("=====================================")
    print("===       Initializing LMPC       ===")
    print("=====================================")

    # Check weight matrix size
    assert Q.shape == (self.nx,self.nx), "Q: incorrect shape!"
    assert R.shape == (self.nu,self.nu), "R: incorrect shape!"

    # Create cost function
    J = 0
    for k in range(self.N-1):
      J += 0.5 * mtimes([ self.dX[:,k].T, Q, self.dX[:,k] ])
      J += 0.5 * mtimes([ self.dU[:,k].T, R, self.dU[:,k] ])

    # Set cost function
    self.nlp.set_cost(J)

    # Create initial state constraint
    self.nlp.add_equality('init', self.dX[:,0] - self.xest + self.xref[:,0])

    # Create shooting constraints
    shoot = SX.sym('shoot',self.nx,self.N-1)
    for k in range(self.N-1):
      A_k = self.A(self.xref[:,k], self.uref[:,k])
      B_k = self.B(self.xref[:,k], self.uref[:,k])
      xnext = mtimes([ A_k, self.dX[:,k] ]) + mtimes([ B_k, self.dU[:,k] ])
      shoot[:,k] = self.dX[:,k+1] - xnext
    
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
    print("===   LMPC initialization done    ===")
    print("=====================================")
    

  def call(self, xest: np.array, 
                 xref: np.array, 
                 uref: np.array ):
    """
    """
    # Check sizes
    assert xest.shape == (self.nx,), "xest: incorrect shape!"
    assert xref.shape == (self.nx,self.N), "xref: incorrect shape!"
    assert uref.shape == (self.nu,self.N-1), "uref: incorrect shape!"

    # Set parameters
    param = self.nlp.struct_p(0)
    param['xest'] = DM( xest )
    param['xref'] = DM( xref )
    param['uref'] = DM( uref )

    # Set initial guess (TODO: improve guess. shift?)
    winit = self.nlp.struct_w(0)
    winit['dX'] = DM.zeros( (self.nx,self.N) )
    winit['dU'] = DM.zeros( (self.nu,self.N-1) )

    # Optimize and return!
    result, stats = self.nlp.solve(winit, param)
    dX_sol = result['w']['dX'].full()
    dU_sol = result['w']['dU'].full()
    return dX_sol, dU_sol, stats, result
