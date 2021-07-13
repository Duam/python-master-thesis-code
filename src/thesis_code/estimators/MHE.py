import casadi as cas
from casadi import Function, jacobian, DM, SX, horzcat
from typing import Callable
import numpy as np
from thesis_code.utils.ParametricNLP import ParametricNLP


class MHE:

  ##
  # @brief Initializes the MHE
  # @param f The discretized ode (function)
  # @param g The measurement function (function)
  # @param N The estimation horizon (int)
  # @param dt The sample time (float/double)
  # @param nu The size of the control vector (TODO: needed?) (int)
  # @param nw The size of the disturbance vector (TODO: needed?) (int)
  # @param nw The size of the measurement vector (TODO: needed?) (int)
  ##
  def __init__(self, F: Callable[[SX,SX,SX],SX], 
                     g: Callable[[SX,SX,SX],SX], 
                     N: int, 
                     dt: float, 
                     nx: int, 
                     nu: int, 
                     nw: int, 
                     ny: int ):

    print("====================================")
    print("===         Creating MHE         ===")
    print("====================================")


    ################################################
    #                                              #
    #      OPTIMIZATION PROBLEM PREPARATION        #
    #                                              #
    ################################################
    ################################################
    ###             FETCH PARAMETERS             ###

    ## Fetch the given parameters as casadi expressions ##
    # Input sizes
    self.nx = nx    # State vector size
    self.nu = nu    # Control vector size
    self.nw = nw    # Disturbance vector size
    self.ny = ny    # Output vector size

    # Trajectory lenghts
    self.Nx = N     # State trajectory length
    self.Nu = N-1   # Control trajectory length 
    self.Nw = N-1   # Disturbance trajectory length
    self.Ny = N-1   # Output trajectory length

    # The other parameters
    self.N = N      # Estimation horizon
    self.dt = dt    # Sample time

    print("Parameters fetched.")

    ################################################
    ###        PREPARE CASADI EXPRESSIONS        ###

    # Create casadi variables
    x = SX.sym('x', self.nx, 1)
    w = SX.sym('w', self.nw, 1)
    u = SX.sym('u', self.nu, 1)
    y = SX.sym('y', self.ny, 1)

    # Create handles for the user to add constraints manually
    self.x = x
    self.w = w
    self.u = u
    self.y = y

    # Discretized ODE
    self.F = Function('F', [x,u,w], [F(x,u,w)], ['x','u','w'], ['xnext'])

    # Measurement function
    self.g = Function('h', [x,u,w], [g(x,u,w)], ['x','u','w'], ['y'])

    # Create jacobians
    self.dFdx = Function('dFdx', [x,u,w], [jacobian(F(x,u,w),x)], ['x','u','w'], ['dFdx'])
    self.dgdx = Function('dgdx', [x,u,w], [jacobian(g(x,u,w),x)], ['x','u','w'], ['dgdx'])

    print("Casadi objects created.")

    ################################################
    ###          OPTIMIZATION VARIABLES          ###

    # Create the parametric NLP
    self.nlp = ParametricNLP('MHE')

    # Define decision variables
    self.nlp.add_decision_var('X', (self.nx,self.Nx))
    self.nlp.add_decision_var('W', (self.nw,self.Nw))

    # Define parameters
    self.nlp.add_parameter('U',  (self.nu,self.Nu))
    self.nlp.add_parameter('Y',  (self.ny,self.Ny))
    self.nlp.add_parameter('x0', (self.nx,1))
    self.nlp.add_parameter('P0', (self.nx,self.nx))

    ## Create aliases for readability ##
    self.X = self.nlp.get_decision_var('X')
    self.W = self.nlp.get_decision_var('W')
    self.U = self.nlp.get_parameter('U')
    self.Y = self.nlp.get_parameter('Y')
    self.x0 = self.nlp.get_parameter('x0')
    self.P0 = self.nlp.get_parameter('P0')

    print("Optimization variables created.")

    print("====================================")
    print("===       MHE creation done      ===")
    print("====================================")

  ##
  # @brief Init function. To be called after equality
  #        constraints have been added
  # @param x0 The initial state estimate (np.array)
  # @param P0 Covariance matrix for the initial state estimate (np.array)
  # @param Q The State noise covariance (np.array)
  # @param R The measurement noise covariance (np.array)
  ##
  def init(self, x0, P0, Q, R):

    print("====================================")
    print("===       Initializing MHE       ===")
    print("====================================")

    ################################################
    #                                              #
    #         OPTIMIZATION PROBLEM SETUP           #
    #                                              #
    ################################################
    ################################################
    ###             FETCH PARAMETERS             ###

    # Tuning parameters
    self.R = R
    self.Q = Q

    # Initial measurement output
    y0 = self.g(x0, np.zeros((self.nu,1)), np.zeros((self.nw,1)))

    # Problem parameters:
    self.parameters = self.nlp.parameters(0)
    self.parameters['U']  = DM.zeros((self.nu,self.N-1))   # Controls
    self.parameters['Y']  = np.repeat(y0, self.N-1, axis=1)  # Measurements
    self.parameters['x0'] = DM(x0)
    self.parameters['P0'] = DM(P0)

    # Buffer objects to hold the horizon information
    self.shooting = self.nlp.shooting(0)
    self.shooting['X'] = np.repeat(DM(x0),self.N, axis=1)  # States
    self.shooting['W'] = DM.zeros((self.nw,self.N-1))       # Disturbances

    ################################################
    ###          CREATE COST FUNCTION            ###
    
    self.nlp.set_cost(self.createCostFunction())
    print("Cost function created.")

    ################################################
    ###           CREATE CONSTRAINTS             ###

    self.createConstraints()
    print ("Constraints created.")

    ################################################
    ###            CREATE NLP SOLVER             ###

    # Specify solver
    nlpsolver = 'ipopt'

    # Specify solver options
    opts = {}
    opts['ipopt.print_info_string'] = 'yes'
    opts['ipopt.print_level'] = 0
    opts['ipopt.max_iter'] = 1000

    # Initialize the solver
    self.nlp.init(nlpsolver, opts)
    print("Solver created.")

    print("====================================")
    print("===   MHE initialization done    ===")
    print("====================================")

  ##
  # @brief Creates the MHE cost function
  # @return The cost function as a CasADi SX
  ##
  def createCostFunction(self):

    # Create cost term ##
    J = 0

    ## Add arrival cost ##
    # = Error in initial state vs. a-priori guess for initial state
    #   Weighted with inverse of covariance matrix of the guess
    J += cas.mtimes([ 
      (self.X[:,0] - self.x0).T,
      self.P0, 
      (self.X[:,0] - self.x0),
    ])

    ## Add measurement cost ##
    # = Error in predicted measurement vs. actual measurement
    #   Weighted with measurement covariance matrix
    for k in range(self.N-1):
      x_k = self.X[:,k]
      #x_k = self.shooting['X', k+1] # Which one is correct?
      u_k = self.U[:,k]
      w_k = self.W[:,k]
      y_k = self.Y[:,k]

      # Compute the mismatch
      y_error = self.g(x_k, u_k, w_k) - y_k
      
      # Weight the mismatch and add it to the cost
      J += cas.mtimes([ 
        y_error.T, 
        self.R, 
        y_error 
      ])

    ## Add process noise cost ##
    for k in range(self.N-1):
      w_k = self.W[:,k]
      
      # Weight the process noise and add it to the cost
      J += cas.mtimes([ 
        w_k.T, 
        self.Q, 
        w_k 
      ])

    '''
    ## Compute pseudo measurement error cost of current state
    x_last = self.shooting['X',self.N-1]
    u_pseudo = np.zeros((self.nu))
    w_pseudo = np.zeros((self.nw))
    y_last = self.parameters['Y',self.N-2]
    y_error = self.g(x_last, u_pseudo, w_pseudo) - y_last

    # Weight the mismatch and add it to the cost
    J += cas.mtimes([
      y_error.T,
      self.R,
      y_error
    ])
    '''

    return J

  ##
  # @brief Creates the MHE equality constraints
  # @return A list of constraints
  ##
  def createConstraints(self):

    # Fill container with constraints
    for k in range(self.N-1):
      # Fetch needed variables
      x_k = self.X[:,k]
      x_next = self.X[:,k+1]
      u_k = self.U[:,k]
      w_k = self.W[:,k]
      y_k = self.Y[:,k]

      # Create multiple-shooting constraints
      # Each state, when simulated forward 
      # using some controls and disturbances,
      # has to be equal to the next state
      shoot_k = x_next - self.F(x_k, u_k, w_k)
      self.nlp.add_equality('shoot_'+str(k), shoot_k)


  ## 
  # @brief User function to add new equality constraints
  # @param expr A expression that resembles the LHS
  # of the constraint equation g = 0. g must utilize 
  # the casadi SX objects defined in the MHE, 
  # such as MHE.X or MHE.Y
  ##
  def add_equality(self, name:str, expr:SX):
    self.nlp.add_equality(name, expr)

  ## 
  # @brief User function to add new inequality constraints
  # @param expr A expression that resembles the LHS
  # of the constraint equation h >= 0. h must utilize 
  # the casadi SX objects defined in the MHE, 
  # such as MHE.X or MHE.Y
  ##
  def add_inequality(self, name:str, expr:SX):
    self.nlp.add_inequality(name, expr)

  ##
  # @brief Setter for the covariances Q and R
  # @param Q State noise covariance
  # @param R Measurement noise covariance
  ##
  def setTuningParams(self, Q, R):
    self.Q = Q # State noise covariance
    self.R = R # Measurement noise covariance

    print("Parameters set by hand.")

  ## 
  # @brief Estimates the current state (Filter mode)
  # @param u The current control
  # @param y The current measurement
  # @return The state estimate at the current time
  ##
  def call(self, u, y):

    print("MHE called")

    ################################################
    ###    ESTIMATE x0 and P0 OF ARRIVAL COST    ###

    x0, P0 = self.estimateArrivalCost()

    ################################################
    ###              SHIFT HORIZON               ###

    # Fetch trajectories
    N = self.N
    X = self.shooting['X'][0]
    W = self.shooting['W'][0]
    U = self.parameters['U'][0]
    Y = self.parameters['Y'][0]

    ## Shift the horizon of us,ys and add the new u,y
    U = horzcat( U[:,1:], u )
    Y = horzcat( Y[:,1:], y )

    ## Shift the horizon of xs and ws
    # new w = 0
    W = horzcat( W[:,1:], DM.zeros((self.nw, 1)) )

    # new x = forward simulation
    # Note: xs[:,N-1] is the most recent state
    #       us[:,N-2] is the most recent control
    #       ws[:,N-2] is the most recent disturbance
    X = horzcat( X[:,1:], self.F( X[:,N-1], U[:,N-2], W[:,N-2]) )

    ################################################
    ###       SET OPTIMIZATION VARIABLES         ###
    
    # Set the optimization parameters
    self.parameters['U']  = U
    self.parameters['Y']  = Y
    self.parameters['x0'] = x0
    self.parameters['P0'] = P0
    
    # Set the initial guess
    self.shooting['X'] = X
    self.shooting['W'] = W

    ################################################
    ###            SOLVE THE PROBLEM             ###

    # Optimize!
    result, stats = self.nlp.solve(self.shooting, self.parameters)

    # Grab the solution and store it
    self.shooting['X'] = result['X']
    self.shooting['W'] = result['W']

    # Return the numerical values
    x_ret = result['X'][0][:,N-1].full().flatten()
    w_ret = result['W'][0][:,N-2].full().flatten()
    return x_ret, w_ret, stats
  
  ## 
  # @brief 
  # @return x0hat, P0hat
  ##
  def estimateArrivalCost(self):

    # Test with implementation from papaer
    # 2017_05_Adaptive_arrival_cost_update_for_improving_MHE_performance
    alpha = 0.5
    beta = 1

    new_x0 = self.shooting['X'][0][:,1]
    new_P0 = alpha*self.parameters['P0'][0] + beta * np.outer(new_x0, new_x0)

    return new_x0, new_P0


  ##
  # @brief Prints the sparsity pattern of the hessian
  #        of the lagrangian
  # @param fignum The figure number
  # @return The figure handle
  ##
  def spy(self, fignum):
    return self.nlp.spy(fignum)


###########################################################################
###                                                                     ###
###                          END OF MHE CLASS                           ###
###                                                                     ###
###########################################################################

##
# Test area to see if there are any errors
##
if __name__ == "__main__":
  f = lambda x, u, w: x+u+w
  g = lambda x, u, w: x

  x0 = np.array([1,2])

  N = 10
  dt = 0.05

  P0 = np.array([
    [0.001]
  ])

  Q = np.array([
    [0.001]
  ])

  R = np.array([
    [0.001]
  ])

  nx = 2
  nu = 1
  nw = 1
  ny = nx

  # Initialize MHE
  mhe = MHE(f, g, N, dt, nx, nu, nw, ny)
  mhe.init(x0, P0, Q, R)
  
  # Simulate the system for a bit
  W = np.zeros((N, 1))
  X0 = 0
  U = np.array([0,0,1,0,1,0,1,2,0,-1])

  x,w,stat = mhe.call(U[0], [1.00,2.00])

  print(x)
  print(w)  


