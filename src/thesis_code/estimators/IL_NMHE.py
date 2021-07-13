#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/03/12
  Brief: This file implements the nonlinear moving horizon estimator (NMHE)
          for iterative learning of periodic disturbance parameters. In this
          version, the periodic parameters are represented in the time-domain.
"""

from casadi import Function, DM, SX, mtimes
import numpy
from thesis_code.utils.ParametricNLP import ParametricNLP
from typing import Callable
import copy
from thesis_code.utils.bcolors import bcolors

class IL_NMHE:

  """ 
  Iteratively Learning Nonlinear Moving Horizon Estimator (IL-NMHE) in TIME-DOMAIN.
  Solves the OCP using multiple shooting.

  Args:
    F: State evolution function
    g: Output function
    dt: Sampling time
    N_per: Samples per period (cycle/iteration)
    N_mhe: MHE horizon
    nx: State vector size
    nu: Control vector size
    ny: Measurement vector size
    np: Parameter vector size
    with_arrival_cost: Flag (default = False)

  Raises:
    AssertionError: If one of the constructor arguments is invalid.
  """

  def __init__(self, F: Callable[[SX,SX,SX],SX], 
                     g: Callable[[SX,SX,SX],SX], 
                     dt: float, 
                     N_per: int,
                     N_mhe: int,
                     nx: int, 
                     nu: int, 
                     ny: int, 
                     np: int,
                     with_arrival_cost: bool = False,
                     verbose: bool = True):

    self.verbose = verbose
    if self.verbose: print("CREATING IL-NMHE ..")

    # Check input types
    assert type(F) == Function, "F: Must be a CasADi function!"
    assert type(g) == Function, "g: Must be a CasADi function!"
    assert type(dt) == float, "Sampling time dt must be float!"
    assert type(N_per) == int, "Period length N_per must be int!"
    assert type(N_mhe) == int, "MHE horizon N_mhe must be int!"
    assert type(nx) == int, "State size nx must be int!"
    assert type(nu) == int, "Control size nu must be int!"
    assert type(ny) == int, "Measurement size ny must be int!"
    assert type(np) == int, "Parameter size np must be int!"
    assert type(with_arrival_cost) == bool, "Flag with_arrival_cost must be bool!"
    if self.verbose: print("Input types OK.")

    # Check I/O of: xnext = F(x, u, p)
    assert F.n_in() == 3, "F: Must have 3 inputs!"
    assert F.n_out() == 1, "F: Must have 1 output!"
    F.assert_size_in(0, nx, 1)
    F.assert_size_in(1, nu, 1)
    F.assert_size_in(2, np, 1)
    F.assert_size_out(0, nx, 1)
    if self.verbose: print("F I/O OK.")

    # Check I/O of: y = g(x, u, p)
    assert g.n_in() == 3, "g: Must have 3 inputs!"
    assert g.n_out() == 1, "g: Must have 1 output!"
    g.assert_size_in(0, nx, 1)
    g.assert_size_in(1, nu, 1)
    g.assert_size_in(2, np, 1)
    g.assert_size_out(0, nx, 1)
    if self.verbose: print("g I/O OK.")

    # Check sizes 
    assert dt > 0, "Sampling time dt must be positive!"
    assert N_per > 0, "Period length N_per must be positive!"
    assert N_mhe > 0, "MHE horizon N_mhe must be positive!"
    assert nx > 0, "State size nx must be positive!"
    assert nu >= 0, "Control size nu must be nonnegative!"
    assert ny > 0, "Measurement size ny must be positive!"
    assert np >= 0, "Parameter size ny must be nonnegative!"    
    if self.verbose: print("Input sizes OK.")

    if with_arrival_cost:
      print("ARRIVAL COST CURRENTLY NOT IMPLEMENTED. SETTING TO FALSE.")
      with_arrival_cost = False

    # Fetch inputs
    self._F = F            # State evolution function
    self._g = g            # Output function
    self._dt = dt          # Sampling time
    self._N_per = N_per    # Samples per period (cycle/iteration)
    self._N_mhe = N_mhe    # MHE horizon
    self._nx = nx          # State vector size
    self._nu = nu          # Control vector size
    self._ny = ny          # Measurement vector size
    self._np = np          # Parameter vector size
    self._with_arrival_cost = with_arrival_cost   # Arrival cost flag
    if self.verbose: print("Inputs fetched.")

    # Compute auxiliary horizons and indices for decision variables and memory
    self._Nx = N_mhe + 1                     # Number of state vectors (variable for nlp)
    self._Np = min([N_per, N_mhe])           # Number of parameter vectors (variable for nlp)
    self._Nu = N_mhe                         # Number of control vectors (fixed for nlp)
    self._Ny = N_mhe                         # Number of measurement vectors (fixed for nlp)
    self._Np_mem = N_per                     # Number of parameter vectors in memory
    self._I0 = - self._N_mhe                 # Starting index of MHE horizon
    self._I1 = self._Np - self._N_mhe - 1    # End index of parameter decision variables
    self._I2 = -1                            # End index of the MHE horizon
    self._sample_idx = 0                     # Sample index at MHE initialization

    """
    TODO: Describe horizon and index choices
    """

    if self.verbose: print("Auxiliary horizons and indices created.")

    # Create parametric NLP, add decision variables and parameters
    self._nlp = ParametricNLP("IL_NMHE", verbose=self.verbose)
    self._nlp.add_decision_var('X', (self._nx,self._Nx))
    self._nlp.add_decision_var('P', (self._np,self._Np))
    self._nlp.add_decision_var('dP', (self._np,self._Np))
    self._nlp.add_decision_var('ddP', (self._np,self._Np))
    self._nlp.add_parameter('U', (self._nu,self._Nu))
    self._nlp.add_parameter('Y', (self._ny,self._Ny))
    self._nlp.add_parameter('P_prior', (self._np,self._Np))
    if self._N_mhe >= self._N_per:
      self._nlp.add_parameter('P_prev', (self._np,self._Np))
    else:
      self._nlp.add_parameter('P_prev', (self._np,self._Np+2))
    if self.verbose: print("NLP created.")

    # Create aliases
    self._X = self._nlp.get_decision_var('X')
    self._P = self._nlp.get_decision_var('P')
    self._dP = self._nlp.get_decision_var('dP')
    self._ddP = self._nlp.get_decision_var('ddP')
    self._U = self._nlp.get_parameter('U')
    self._Y = self._nlp.get_parameter('Y')
    self._P_prior = self._nlp.get_parameter('P_prior')
    self._P_prev = self._nlp.get_parameter('P_prev')
    if self.verbose: print("Aliases created.")

    if self.verbose: print("IL-NMHE CREATION DONE.")

  ################################################################################

  def getDataIndex(self, k: int):
    """ 
    Index function. Maps from the MHE-index to data-index. Required because the
    MHE works on data in the "past" (indicated by negative sign index), but the
    data can only be accessed with nonnegative indices.
    
    Args:
      k: The MHE-index (-N_mhe to 0)

    Returns:
      The data-index.

    Raises:
      AssertionError: If the MHE-index is in an invalid range.
    """

    # Check input type and range
    assert type(k) == int, "MHE-index k must be int!"
    assert k >= -self._N_mhe, "MHE index k too small: Must reference a time down to earliest sample in the MHE horizon!"
    assert k <= 0, "MHE index k too big: Must reference a time only up to the current time!"
    return k + self._N_mhe

  ################################################################################

  def init(self, x0: numpy.ndarray,
                 P0: numpy.ndarray,
                 R:  numpy.ndarray,
                 W1: numpy.ndarray, 
                 W2: numpy.ndarray, 
                 W3: numpy.ndarray,
                 W4: numpy.ndarray):

    """ IL-NMHE initialization routine. Sets up the optimization problem.

    Args:
      x0: Initial state
      P0: Initial parameters over one trial
      R: Measurement noise weight
      W1: Parameter regularization weight
      W2: Parameter change (trial-to-trial) regularization weight
      W3: Parameter change (sample-to-sample) regularization weight

    Raises:
      AssertionError: If one of the inputs is of an invalid type or has the wrong shape.
    """

    if self.verbose: print("INITIALIZING IL-NMHE ..")
    
    # Check input types
    assert type(x0) == numpy.ndarray, "Initial state x0 must be numpy.ndarray!"
    assert type(P0) == numpy.ndarray, "Initial parameters P0 must be numpy.ndarray!"
    assert type(R) == numpy.ndarray, "Noise weight R must be numpy.ndarray!"
    assert type(W1) == numpy.ndarray, "Regularization weight W1 must be numpy.ndarray!"
    assert type(W2) == numpy.ndarray, "Regularization weight W2 must be numpy.ndarray!"
    assert type(W3) == numpy.ndarray, "Regularization weight W3 must be numpy.ndarray!"
    assert type(W4) == numpy.ndarray, "Regularization weight W4 must be numpy.ndarray!"
    if self.verbose: print("Input types OK.")

    # Check input sizes
    assert x0.shape == (self._nx,1), "Initial state x0 must have shape (nx,1)!"
    assert P0.shape == (self._np,self._N_per), "Initial parameter P0 must have shape (np,N_per)!"
    assert R.shape == (self._ny,self._ny), "Noise weight R must have shape (ny,ny)!"
    assert W1.shape == (self._np,self._np), "Regularization weight W1 must have shape (np,np)!"
    assert W2.shape == (self._np,self._np), "Regularization weight W2 must have shape (np,np)!"
    assert W3.shape == (self._np,self._np), "Regularization weight W3 must have shape (np,np)!"
    assert W4.shape == (self._np,self._np), "Regularization weight W4 must have shape (np,np)!"
    if self.verbose: print("Input sizes OK.")

    # Check positive-definiteness of input matrices
    """ # TODO
    for mat in (R, W1, W2, W3):
      try:
        numpy.linalg.cholesky(mat)
      except Exception as e:
        e.args += ("How do I retrieve the variable names?",)
        raise
    """

    # Fetch inputs
    self._R = R     # Measurement noise weight
    self._W1 = W1   # Parameter regularization weight
    self._W2 = W2   # Parameter change (trial-to-trial) regularization weight
    self._W3 = W3   # Parameter 1st derivative regularization weight
    self._W4 = W4   # Parameter 2nd derivative regularization weight
    if self.verbose: print("Inputs fetched.")

    # Create the cost function
    J = 0
    # TODO: Arrival cost?

    # Least squares term
    for k in range(self._N_mhe):
      kmodN = numpy.mod(k, self._N_per)
      y = self._Y[:,k]
      x = self._X[:,k]
      u = self._U[:,k]
      p = self._P[:,kmodN]
      e = y - self._g(x,u,p)
      J += 0.5 * mtimes([e.T, self._R, e])

    # Parameter regularization (regular and trial-to-trial)
    for k in range(self._Np):
      p = self._P[:,k]

      # Prior
      p_prior_diff = p - self._P_prior[:,k]
      J += 0.5 * mtimes([p_prior_diff.T, self._W1, p_prior_diff])

      # Trial-to-trial
      p_mem_diff = p - self._P_prev[:,k]
      J += 0.5 * mtimes([p_mem_diff.T, self._W2, p_mem_diff])

      # First derivative
      dp = self._dP[:,k]
      J += 0.5 * mtimes([dp.T, self._W3, dp])

      # Second derivative
      ddp = self._ddP[:,k]
      J += 0.5 * mtimes([ddp.T, self._W4, ddp])

    # Set cost function
    self._nlp.set_cost(J)
    if self.verbose: print("Cost function set.")

    # Create initial state constraint
    #self._nlp.add_equality('init', ) # TODO DO THIS?

    # Create shooting constraints
    shoot = SX.sym('shoot', self._nx, self._Nx-1)
    for k in range(self._N_mhe):
      kmodN = numpy.mod(k, self._N_per)
      x = self._X[:,k]
      u = self._U[:,k]
      p = self._P[:,kmodN]
      xnext = self._X[:,k+1]
      shoot[:,k] = xnext - self._F(x,u,p)

    self._nlp.add_equality('shoot', shoot)
    if self.verbose: print("Shooting constraints created.")
  
    # Create parameter (rate of change) constraints
    delta_p = SX.sym('delta_p', self._np, self._Np)
    if self._N_mhe >= self._N_per:
      delta_p[:,0] = self._dP[:,0] - self._P[:,0] + self._P[:,-1]
    else:
      delta_p[:,0] = self._dP[:,0] - self._P[:,0] + self._P_prev[:,-1]
    for k in range(1, self._Np):
      delta_p[:,k] = self._dP[:,k] - self._P[:,k] + self._P[:,k-1]

    self._nlp.add_equality('delta_p', delta_p)
    if self.verbose: print("Parameter rate of change constraints created.")

    # Create parameter (rate of rate of change) constraints
    delta_delta_p = SX.sym('delta_delta_p', self._np, self._Np)
    if self._N_mhe >= self._N_per:
      delta_delta_p[:,0] = self._ddP[:,0] - self._dP[:,0] + self._dP[:,-1]
    else:
      delta_delta_p[:,0] = self._ddP[:,0] - self._dP[:,0] + (self._P_prev[:,-1] - self._P_prev[:,-2])
    for k in range(1, self._Np):
      delta_delta_p[:,k] = self._ddP[:,k] - self._dP[:,k] + self._dP[:,k-1]

    self._nlp.add_equality('delta_delta_p', delta_delta_p)
    if self.verbose: print("Parameter rate of rate of change constraints created.")

    # Define solver and its configuration
    nlpsolver = 'ipopt'
    nlpsolver_opts = {
      'ipopt.print_user_options': 'no',
      'ipopt.print_info_string': 'no',
      'print_time': 0,
      'ipopt.print_level': 0,
      'ipopt.sb': 'yes',
      'ipopt.max_iter': 1000,
      'ipopt.constr_viol_tol': 1e-8,
      'ipopt.tol': 1e-16
    }

    # Define callback function
    def iterCb(i, sol): 
      pass

    # Initialize solver
    self._nlp.init(nlpsolver, nlpsolver_opts, iterCb)
    if self.verbose: print("NLP solver created (" + nlpsolver + ").")

    # Compute (NOTE: POSSIBLY INFEASIBLE/UNREALISTIC) dummy initial values
    u0 = numpy.zeros((self._nu,1))
    y0 = self._g(x0, u0, P0[:,0])
    if self.verbose: print("Dummy initial values computed.")

    # Create data containers and populate them with dummy initial values
    self._X_mem = numpy.repeat(x0, self._Nx, axis=1)
    self._U_mem = numpy.repeat(u0, self._Nu, axis=1)
    self._Y_mem = numpy.repeat(y0, self._Ny, axis=1)
    self._P_full_mem = P0
    self._P_prior_full_mem = P0
    self._dP_full_mem = numpy.zeros((self._nx,self._Np_mem))
    self._dP_full_mem[:,0] = self._P_full_mem[:,0] - self._P_full_mem[:,-1]
    self._dP_full_mem[:,1:] = numpy.diff(self._P_full_mem, axis=1)
    self._ddP_full_mem = numpy.zeros((self._nx,self._Np_mem))
    self._ddP_full_mem[:,0] = self._dP_full_mem[:,0] - self._dP_full_mem[:,-1]
    self._ddP_full_mem[:,1:] = numpy.diff(self._dP_full_mem, axis=1)

    if self.verbose: print ("Data containers created and populated. ")    
    if self.verbose: print("IL-NMHE INITIALIZATION DONE.")

  ################################################################################

  def reset (self, x0: numpy.ndarray,
                   P0: numpy.ndarray ):
    """ Resets the internal memory of the component
    
    Arguments:
      x0 {numpy.ndarray} -- Initial state estimate
      P0 {numpy.ndarray} -- Initial parameter trajectory estimate
    """

    # Compute dummy initial values
    u0 = numpy.zeros((self._nu,1))
    y0 = self._g(x0, u0, P0[:,0])
    if self.verbose: print("Dummy initial values computed.")

    # Populate memory with dummy initial values
    self._X_mem = numpy.repeat(x0, self._Nx, axis=1)
    self._U_mem = numpy.repeat(u0, self._Nu, axis=1)
    self._Y_mem = numpy.repeat(y0, self._Ny, axis=1)
    self._P_full_mem = P0
    self._P_prior_full_mem = P0
    self._dP_full_mem = numpy.zeros((self._nx,self._Np_mem))
    self._dP_full_mem[:,0] = self._P_full_mem[:,0] - self._P_full_mem[:,-1]
    self._dP_full_mem[:,1:] = numpy.diff(self._P_full_mem, axis=1)
    self._ddP_full_mem = numpy.zeros((self._nx,self._Np_mem))
    self._ddP_full_mem[:,0] = self._dP_full_mem[:,0] - self._dP_full_mem[:,-1]
    self._ddP_full_mem[:,1:] = numpy.diff(self._dP_full_mem, axis=1)

  ################################################################################
  
  def prepare (self, k: int, u: numpy.ndarray):
    """
    Sets up the optimization problem and initial guess. TODO
    Call this method before applying the control to the system.
    
    Args:
      k: The current sample index
      u: The applied control vector at index k

    Raises:
      AssertionError: If one of the inputs is of an invalid type or has the wrong shape.
    """
    # Check inputs TODO
    # Prepare TODO
    pass

  ################################################################################

  def solve(self, y: numpy.ndarray):
    """
    Solves the optiziation problem. TODO
    Call this method after applying the control to the system.
    
    Args:
      y: The measurement vector at index k

    Returns: 
      TODO

    Raises: 
      AssertionError: If one of the inputs is of an invalid type or has the wrong shape.
    """

    pass

  ################################################################################

  def call(self, k:int, u: numpy.ndarray, y: numpy.ndarray):
    """ 
    Calls the MHE, updates the optimization problem and initial guesses, then solves the OPC.
    TODO: Internally calls prepare(k,u) and then solve(y).
    
    Args:
      k: The current sample index
      u: The applied control at index k
      y: The resulting measurement at index k

    Returns:
      The tuple (x,P,stats) containing:
      x: The state at index k+1
      P: The parameter vector over one trial
      stats: NLP solver statistics

    Raises:
      AssertionError: If one of the inputs is of an invalid type or has the wrong shape.
    """

    # Check input types
    assert type(k) == int, "Sample index k must be int!"
    assert type(u) == numpy.ndarray, "Control vector must be numpy.ndarray!"
    assert type(y) == numpy.ndarray, "Measurement vector must be numpy.ndarray!"
    # Check input shapes
    assert u.shape == (self._nu,1), "Control vector shape must be (nu,1)!"
    assert y.shape == (self._ny,1), "Measurement vector shape must be (ny,1)!"

    # Shift current controls and measurements into their corresponding container
    self._U_mem = numpy.roll(self._U_mem, -1, axis=1)
    self._Y_mem = numpy.roll(self._Y_mem, -1, axis=1)
    self._U_mem[:,-1] = u.flatten()
    self._Y_mem[:,-1] = y.flatten()

    #if __debug__:
    #  print("_P_full_mem (before shift)=" + str(self._P_full_mem))

    # Shift parameters and states. Predict current state and shift it into initial guess
    self._P_full_mem   = numpy.roll(self._P_full_mem, -1, axis=1)
    self._P_prior_full_mem = numpy.roll(self._P_prior_full_mem, -1, axis=1)
    self._dP_full_mem  = numpy.roll(self._dP_full_mem, -1, axis=1)
    self._ddP_full_mem = numpy.roll(self._ddP_full_mem, -1, axis=1)
    self._X_mem = numpy.roll(self._X_mem, -1, axis=1)
    p_realigned = numpy.roll(self._P_full_mem, -self._N_mhe)
    self._X_mem[:,-1] = self._F(self._X_mem[:,-2], u, p_realigned[:,-1])

    #if __debug__:
    #  print("_P_full_mem (after shift)=" + str(self._P_full_mem))

    # Create the nlp initial guess struct
    initial_guess = self._nlp.struct_w(0)
    initial_guess['X']   = DM( copy.deepcopy(self._X_mem) )
    initial_guess['P']   = DM( copy.deepcopy(self._P_full_mem[:,:self._Np]) )
    initial_guess['dP']  = DM( copy.deepcopy(self._dP_full_mem[:,:self._Np]) )
    initial_guess['ddP'] = DM( copy.deepcopy(self._ddP_full_mem[:,:self._Np]) )
    #if __debug__:
    #  print("Initial guess (P) = " + str(initial_guess['P']))

    # Create the nlp parameter struct
    nlp_params = self._nlp.struct_p(0)
    nlp_params['U'] = DM( copy.deepcopy(self._U_mem) )
    nlp_params['Y'] = DM( copy.deepcopy(self._Y_mem) )
    nlp_params['P_prior'] = DM( copy.deepcopy(self._P_prior_full_mem[:,:self._Np]) )
    if self._N_mhe >= self._N_per:
      nlp_params['P_prev'] = DM( copy.deepcopy(self._P_full_mem[:,:self._Np]) )
    else:
      p_cpy =  copy.deepcopy(self._P_full_mem)
      p_reduced = numpy.append(p_cpy[:,:self._Np], [p_cpy[:,-2],p_cpy[:,-1]])
      nlp_params['P_prev'] = DM( p_reduced )

    # Solve the nlp
    result, stats = self._nlp.solve(initial_guess, nlp_params)

    if self.verbose and False:
      print("RESULT ==================== RESULT")
      print("f = " + str(result['f']))
      print("w['X'] = " + str(result['w']['X']))
      print("w['P'] = " + str(result['w']['P']))
      print("lam_w['X'] = " + str(result['lam_w']['X']))
      print("lam_w['P'] = " + str(result['lam_w']['P']))
      print("p['U'] = " + str(result['p']['U']) )
      print("p['Y'] = " + str(result['p']['Y']) )
      print("p['P_prev'] = " + str(result['p']['P_prev']) )
      print("lam_p['U'] = " + str(result['lam_p']['U']) )
      print("lam_p['Y'] = " + str(result['lam_p']['Y']) )
      print("lam_p['P_prev'] = " + str(result['lam_p']['P_prev']) )
      print("g['shoot'] = " + str(result['g']['shoot']))
      print("lam_g['X'] = " + str(result['lam_w']['X']))
      print("lam_g['P'] = " + str(result['lam_w']['P']))
      for k in range(self._N_mhe):
        x = result['w']['X'].full()[:,k]
        u = result['p']['U'].full()[:,k]
        p = result['w']['P'].full()[:,k]
        print("\tk="+str(k)+": g(x,u,p) = " +str(self._g(x,u,p)))
      #print("h = " + str(result['h']))
      #print(result)

    # Get solution
    X_sol = result['w']['X'].full()
    P_sol = result['w']['P'].full()

    # Check output types
    assert type(X_sol) == numpy.ndarray, "State trajectory X_sol is not numpy.ndarray! Check implementation!"
    assert type(P_sol) == numpy.ndarray, "Parameter trajectory P_sol is not numpy.ndarray! Check implementation!"
    assert type(stats) == dict, "NLP solver statistics stats is not dict! Check implementation!"  
    assert X_sol.shape == (self._nx,self._Nx), "State trajectory X_sol doesn't have shape (nx,Nx)! Check implementation!"
    assert P_sol.shape == (self._np,self._Np), "Parameter trajectory P_sol doesn't have shape (np,Np)! Check implementation!"

    # Write back the solution
    self._X_mem = copy.deepcopy(X_sol)
    self._P_full_mem[:,:self._Np] = copy.deepcopy(P_sol)

    # Recompute and write back derivatives
    self._dP_full_mem[:,0] = self._P_full_mem[:,0] - self._P_full_mem[:,-1]
    self._dP_full_mem[:,1:] = numpy.diff(self._P_full_mem, axis=1)
    self._ddP_full_mem[:,0] = self._dP_full_mem[:,0] - self._dP_full_mem[:,-1]
    self._ddP_full_mem[:,1:] = numpy.diff(self._dP_full_mem, axis=1)
    
    # Fetch current state estimate
    x = copy.deepcopy( self._X_mem[:,-1] ).reshape((self._nx,1))
    P = numpy.roll( copy.deepcopy(self._P_full_mem), -self._N_mhe)

    # Check output type and shape
    assert type(x) == numpy.ndarray, "State vector x is not numpy.ndarray! Check implementation!"
    assert x.shape == (self._nx,1), "State vector x doesn't have shape (nx,1)! Check implementation!"

    # Return the results
    return x, P, stats
  

  def toString(self, with_constants=True):
    info = "\n"
    info += bcolors.HEADER + "============= MHE STATUS STRING =============" + bcolors.ENDC + "\n"
    info += bcolors.BOLD + "Sizes:" + bcolors.ENDC + "\n"
    info += bcolors.OKGREEN + "   nu = " + bcolors.ENDC + str(self._nu) + "\n"
    info += bcolors.OKGREEN + "   nx = " + bcolors.ENDC + str(self._nx) + "\n"
    info += bcolors.OKGREEN + "   ny = " + bcolors.ENDC + str(self._ny) + "\n"
    info += bcolors.OKGREEN + "   np = " + bcolors.ENDC + str(self._np) + "\n"
    info += bcolors.OKGREEN + " Nper = " + bcolors.ENDC + str(self._N_per) + "\n"
    info += bcolors.OKGREEN + " Nmhe = " + bcolors.ENDC + str(self._N_mhe) + "\n"
    info += bcolors.BOLD + "Internal Memory:" + bcolors.ENDC + "\n"
    info += bcolors.OKGREEN + "    U = " + bcolors.ENDC +  str(self._U_mem) + "\n"
    info += bcolors.OKGREEN + "    X = " + bcolors.ENDC + str(self._X_mem) + "\n"
    info += bcolors.OKGREEN + "    Y = " + bcolors.ENDC + str(self._Y_mem) + "\n"
    info += bcolors.OKGREEN + "    P = " + bcolors.ENDC + str(self._P_full_mem) + "\n"
    info += bcolors.OKGREEN + "   dP = " + bcolors.ENDC + str(self._dP_full_mem) + "\n"
    info += bcolors.OKGREEN + "  ddP = " + bcolors.ENDC + str(self._ddP_full_mem) + "\n"
    if with_constants:
      info += bcolors.BOLD + "Internal Memory (Constants): " + bcolors.ENDC + "\n"
      info += bcolors.OKGREEN + " P (prior) = " + bcolors.ENDC + str(self._P_prior_full_mem) + "\n"
    info += bcolors.HEADER + "============================================" + bcolors.ENDC + "\n"
    info += "\n"
    return info
  
  def getResultString(self, result: dict):
    pass

###########################################################################
###                                                                     ###
###                          END OF MHE CLASS                           ###
###                                                                     ###
###########################################################################