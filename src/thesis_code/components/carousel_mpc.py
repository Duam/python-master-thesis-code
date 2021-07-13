from thesis_code.models.carousel_whitebox import CarouselWhiteBoxModel
import casadi as cas
from casadi import Function, jacobian, DM, mtimes, vertcat
from thesis_code.utils.CollocationHelper import simpleColl
from thesis_code.utils.ParametricNLP import ParametricNLP


class Carousel_MPC:

  def __init__(self, model: CarouselWhiteBoxModel, N: int, dt: float, verbose:bool=False, do_compile:bool=False, expand:bool=False):
    """Carousel_MPC A nonlinear model predictive controller for the carousel whitebox model

    Args:
      model[CarouselWhiteBoxModel] -- A model instance
      N[int] -- The prediction horizon
      dt[float] -- The timestep
      verbose[bool] -- Verbosity flag (default False)

    """

    print("====================================")
    print("===         Creating MPC         ===")
    print("====================================")

    self.verbose = verbose
    self.do_compile = do_compile
    self.expand = expand
    self.model = model

    ## Fetch the given parameters as casadi expressions ##
    # Input sizes
    NX, NZ, NU, NY, NP = (model.NX(), model.NZ(), model.NU(), model.NY(), model.NP())
    self.NX, self.NZ, self.NU, self.NY, self.NP = (NX, NZ, NU, NY, NP)
    
    # The other parameters
    self.N = N      # Estimation horizon
    self.dt = dt    # Sample time
    print("Horizon length: ", N)
    print("Timestep: ", dt)

    print("Parameters fetched.")

    ## Prepare casadi expressions ##
    # Get carousel symbolics
    sym_args = x_sym, z_sym, u_sym, p_sym = [model.x_aug_sym, model.z_sym, model.u_aug_sym, model.p_sym]
    ode_sym = model.ode_aug_fun(*sym_args)
    alg_sym = model.alg_aug_fun(*sym_args)
    out_sym = model.out_aug_fun(*sym_args)
    self.ode_fun = Function('ode', sym_args, [ode_sym]).expand()
    self.alg_fun = Function('alg', sym_args, [alg_sym]).expand()
    print("Casadi objects created.")

    ## Create collocation integrator ##
    ncol = 3          # Number of collocation points per interval
    Ncol = N*ncol # Total number of collocation points
    tau_root = [0] + cas.collocation_points(ncol, 'radau')
    self.collocation_dt = [ dt * (tau_root[i+1] - tau_root[i]) for i in range(len(tau_root)-1) ]
    print("Collocation timesteps: ", self.collocation_dt)
    dae_dict = {'x': x_sym, 'z': z_sym, 'p': cas.vertcat(p_sym, u_sym), 'ode': ode_sym, 'alg': alg_sym}
    # Integrator for use in OCP
    G = simpleColl(dae_dict, tau_root, dt)
    self.G = G
    print("Collocation equations created.")

    # Integrators for use in forward simulation
    dt_sym = cas.MX.sym('dt')
    int_opts = {'number_of_finite_elements': 1, 'tf': 1.0, 'expand': True, 'jit': True, 'rootfinder':'kinsol', 
      'jit_options':{'flags':['-O3']}}   
    dae_dict = {'x': x_sym, 'z': z_sym, 'p': cas.vertcat(dt_sym, p_sym, u_sym), 'ode': dt_sym*ode_sym, 'alg': alg_sym}
    self.integrator = cas.integrator('xnext','collocation',dae_dict, int_opts)
    print("Integrator created.")   

    # Create the parametric NLP
    ocp = ParametricNLP('Carousel_MPC_N'+str(N))
    ocp.add_decision_var('X', (NX,N+1))   # Differential states
    ocp.add_decision_var('Vx', (ncol*NX,N))  # Differential states at collocation points
    ocp.add_decision_var('Vz', (ncol*NZ,N))  # Algebraic states at collocation points
    ocp.add_decision_var('U',  (NU,N)) # Past controls
    ocp.add_decision_var('s', (1,1)) # Slack variable for AoA constraints
    ocp.add_parameter('x0', (NX,1)) # Current differential state estimate
    ocp.add_parameter('Xref', (NX,N+1)) # Reference trajectory: States
    ocp.add_parameter('Uref', (NU,N)) # Reference trajectory: Controls
    ocp.add_parameter('Q', (NX,1)) # Running cost weight matrix for states
    ocp.add_parameter('R', (NU,1)) # Running cost weight matrix for controls
    ocp.add_parameter('S', (NX,1)) # End cost weight matrix
    ocp.add_parameter('P', (NP,1)) # Model parameters
    ocp.bake_variables()

    # Fetch OCP symbolics
    X_sym = ocp.get_decision_var('X')
    Vx_sym = ocp.get_decision_var('Vx')
    Vz_sym = ocp.get_decision_var('Vz')
    U_sym = ocp.get_decision_var('U')
    s_sym = ocp.get_decision_var('s')
    x0_sym = ocp.get_parameter('x0')
    Xref_sym = ocp.get_parameter('Xref')
    Uref_sym = ocp.get_parameter('Uref')
    Q_sym = ocp.get_parameter('Q')
    R_sym = ocp.get_parameter('R')
    S_sym = ocp.get_parameter('S')
    P_sym = ocp.get_parameter('P')

    # Set up optimization problem
    xnext = cas.MX.sym('xnext', NX)
    xref =cas.MX.sym('xref', NX)
    uref = cas.MX.sym('uref', NU)
    x = cas.MX.sym('x', NX)
    vx = cas.MX.sym('vx', ncol * NX)
    vz = cas.MX.sym('vz', ncol * NZ)
    u = cas.MX.sym('u', NU)
    s = cas.MX.sym('s', 1, 1)
    p = cas.MX.sym('p', NP)
    Q = cas.MX.sym('Q', NX, 1)
    R = cas.MX.sym('R', NU, 1)


    # Create symbolic expressions
    g = G(x, vx.reshape((NX, ncol)), vz.reshape((NZ, ncol)), vertcat(p, u))[1]
    xf = G(x, vx.reshape((NX, ncol)), vz.reshape((NZ, ncol)), vertcat(p, u))[0]
    residual_xN = Xref_sym[:,N] - X_sym[:,N]
    residual_x = xref - x
    residual_u = uref - u

    # Create functions
    g_fun = Function('xf_k', [x, vx, vz, u, p], [g])
    cont_fun = Function('cont', [xnext, x, vx, vz, u, p], [xnext - xf])
    residual_xu_fun = Function('residual', [xref, uref, x, u], [vertcat(residual_x, residual_u)])
    weight_xu_fun = Function('weight', [Q, R], [vertcat(Q, R)])

    aoa_stall = model.constants['AoA_stall']
    aux_funs_arg_x = vertcat(x[:2], 0.0, x[2:4], self.model.constants['carousel_speed'], x[4])
    aoa_A_max_fun = Function('aoa_A_max', [x, p, s], [aoa_stall - model.alpha_A(aux_funs_arg_x, p) + s])
    aoa_A_min_fun = Function('aoa_A_min', [x, p, s], [aoa_stall + model.alpha_A(aux_funs_arg_x, p) + s])
    aoa_E_max_fun = Function('aoa_E_max', [x, p, s], [aoa_stall - model.alpha_E(aux_funs_arg_x, p) + s])
    aoa_E_min_fun = Function('aoa_E_min', [x, p, s], [aoa_stall + model.alpha_E(aux_funs_arg_x, p) + s])
    u_max_fun = Function('u_max', [u], [1.0 - u[0]])
    u_min_fun = Function('u_min', [u], [u[0] - 0.0])

    # Create mapped versions of the functions
    map_args = ['thread', 4]  # Effectively halves hessian computation time!
    g_map = g_fun.map(N, *map_args)
    cont_map = cont_fun.map(N, *map_args)
    alg_map = self.alg_fun.map(N + 1, *map_args)
    residual_xu_map = residual_xu_fun.map(N, *map_args)
    weight_xu_map = weight_xu_fun.map(N, *map_args)

    aoa_A_max_map = aoa_A_max_fun.map(N + 1, *map_args)
    aoa_A_min_map = aoa_A_min_fun.map(N + 1, *map_args)
    aoa_E_max_map = aoa_E_max_fun.map(N + 1, *map_args)
    aoa_E_min_map = aoa_E_min_fun.map(N + 1, *map_args)
    u_max_map = u_max_fun.map(N, *map_args)
    u_min_map = u_min_fun.map(N, *map_args)

    # Repeat "matrices" so that the threads use separate resources
    P_repN = cas.repmat(P_sym, 1, N)
    P_repNp1 = cas.repmat(P_sym, 1, N+1)
    Q_repN = cas.repmat(Q_sym, 1, N)
    R_repN = cas.repmat(R_sym, 1, N)
    s_repNp1 = cas.repmat(s_sym, 1, N+1)

    # Compute total residual and weights, weigh the residuals and set the cost
    residual_xu_size = (NX + NU) * N
    residual_xu_eval = residual_xu_map(Xref_sym[:,:-1], Uref_sym, X_sym[:, :-1], U_sym)
    weight_xu_eval = weight_xu_map(Q_repN, R_repN)
    total_residual = vertcat(residual_xN, residual_xu_eval.reshape((residual_xu_size, 1)))
    total_weight = cas.diag(vertcat(S_sym, weight_xu_eval.reshape((residual_xu_size, 1))))
    total_weight_sqrt = cas.sqrt(total_weight)
    total_weighted_residual = mtimes(total_weight_sqrt, total_residual)
    COST = 0.5 * mtimes(total_weighted_residual.T, total_weighted_residual)
    COST += 1e4 * s_sym + 1e4 * s_sym * s_sym
    ocp.set_cost(COST)

    # Set the constraints
    ocp.add_equality('init', x0_sym - X_sym[:,0])
    ocp.add_equality('coll', g_map(X_sym[:, :-1], Vx_sym, Vz_sym, U_sym, P_repN))
    ocp.add_equality('cont', cont_map(X_sym[:,1:], X_sym[:, :-1], Vx_sym, Vz_sym, U_sym, P_repN))
    ocp.add_inequality('u <= u_max', u_max_map(U_sym))
    ocp.add_inequality('u >= u_min', u_min_map(U_sym))
    ocp.add_inequality('aoa_A <= aoa_max + s', aoa_A_max_map(X_sym, P_repNp1, s_repNp1))
    ocp.add_inequality('aoa_A + s >= aoa_min', aoa_A_min_map(X_sym, P_repNp1, s_repNp1))
    ocp.add_inequality('aoa_E <= aoa_max + s', aoa_E_max_map(X_sym, P_repNp1, s_repNp1))
    ocp.add_inequality('aoa_E + s >= aoa_min', aoa_E_min_map(X_sym, P_repNp1, s_repNp1))
    ocp.add_inequality('s >= 0', s_sym )

    print("Optimization variables created.")
    
    # Jacobian of residuals
    w_ocp_sym = ocp.struct_w
    p_ocp_sym = ocp.struct_p
    GNJ = jacobian(total_weighted_residual, w_ocp_sym)
    # Gauss-Newton hessian
    GNH = cas.triu(mtimes(GNJ.T, GNJ))
    # Create hessian approximation functor
    sigma = cas.MX.sym('sigma')
    lamb = cas.MX.sym('lambda',0,1)
    self.hess_lag = Function('GNH', [w_ocp_sym, p_ocp_sym, sigma, lamb], [sigma * GNH])

    ## Initialize solver
    ocp.init(
      nlpsolver = 'ipopt',
      opts = {
        #'hess_lag': self.hess_lag,
        'ipopt.linear_solver': 'ma57',
        #'ipopt.ma57_automatic_scaling': 'no',
        'expand': self.expand,
        'ipopt.print_level': 5 if verbose else 0,
        'print_time': 1 if verbose else 0,
        'ipopt.print_timing_statistics': 'yes' if verbose else 'no',
        'ipopt.sb': 'yes',
        'jit': False,
        'jit_options': {'flags':['-O3']},
        'ipopt.max_cpu_time': 20 * 1e-3,
      },
      create_analysis_functors=False,
      compile_solver = do_compile
    )
    
    # Create aliases
    self.N = N
    self.ncol = ncol
    self.Ncol = Ncol
    self.ocp = ocp

    self.initialized = False
    print("====================================")
    print("===       MPC creation done      ===")
    print("====================================")

  """ ============================================================================================================ """

  def simulateModel(self, x:DM, z:DM, u:DM, p:DM, dt:DM):
    step = self.integrator(x0=x,z0=z,p=vertcat(dt,p,u))
    return step['xf'], step['zf']

  """ ============================================================================================================ """

  def init(self, x0:DM, Q:DM, R:DM, S:DM, Uref:DM):
    """init Initializes the component

    Args:
      x0[DM] -- The initial differential state estimate
      Q[DM] -- State residual weight (running cost)
      R[DM] -- Control residual weight (running cost)
      S[DM] -- State residual weight (end cost)
      Uref[DM] -- The initial control reference

    """
    print("Initializing MPC..")
    assert not self.initialized, "Already initialized! Called twice?"
    assert DM(x0).shape == (self.NX,1), "x0 shape = " + str(x0.shape)
    #assert DM(z0).shape == (self.NZ,1)
    assert Q.shape == (self.NX,1)
    assert R.shape == (self.NU,1)
    assert S.shape == (self.NX,1)
    assert Uref.shape == (self.NU,self.N), "Uref shape = " + str(Uref.shape)
    
    print("Filling buffers with dummy values:")
    #print("Simulating forward with\nx0 =\n",x0, "\nz0 =\n", z0, "\nUref =\n", Uref)

    # Simulate N steps forward to fill the buffers
    X = DM.zeros((self.NX,self.N+1))
    Vx = DM.zeros((self.ncol*self.NX,self.N))
    Vz = DM.zeros((self.ncol*self.NZ,self.N))
    U = DM(Uref)
    p = self.model.p0()

    X[:,0] = DM(x0)
    # Prepare collocation containers
    vx_k = DM.zeros(((1 + self.ncol) * self.NX,1))
    vz_k = DM.zeros(((1 + self.ncol) * self.NZ,1))
    vx_k[-self.NX:] = DM(x0)
    
    # Simulate full horizon
    for k in range(self.N):  
      #vx_k[:,0] = vx_k[:,-1]
      vx_k[:self.NX] = vx_k[-self.NX:]
      u0_k = U[:, k]
      # Simulate collocation nodes
      for i in range(self.ncol):
        dt = self.collocation_dt[i]
        vx_curr = vx_k[i * self.NX:(i + 1) * self.NX]
        vz_curr = vz_k[i * self.NZ:(i + 1) * self.NZ]
        vx_next, vz_next = self.simulateModel(vx_curr, vz_curr, u0_k, p, dt)
        vx_k[(i + 1) * self.NX:(i + 2) * self.NX] = vx_next
        vz_k[(i + 1) * self.NZ:(i + 2) * self.NZ] = vz_next

      # Write back new states
      X[:, k + 1] = vx_k[-self.NX:]
      Vx[:, k] = vx_k[self.NX:]
      Vz[:, k] = vz_k[self.NZ:]

    print("Setting problem parameters and initial guess..")
    # Problem parameters:
    self.params_x0 = DM(X[:,0])
    self.params_Xref = DM(X)
    self.params_Uref = DM(U)
    self.params_Q = DM(Q)
    self.params_R = DM(R)
    self.params_S = DM(S)
    self.params_P = DM(p)
    self.parameters = self.ocp.struct_p(0)
    self.parameters['x0'] = self.params_x0
    self.parameters['Xref'] = self.params_Xref
    self.parameters['Uref'] = self.params_Uref
    self.parameters['Q'] = self.params_Q
    self.parameters['R'] = self.params_R
    self.parameters['S'] = self.params_S
    self.parameters['P'] = self.params_P

    # Buffer objects to hold the horizon information
    self.guess_X = DM(X)
    self.guess_Vx = DM(Vx)
    self.guess_Vz = DM(Vz)
    self.guess_U = DM(U)
    self.guess_s = 1e3
    self.initial_guess = self.ocp.struct_w(0)
    self.initial_guess['X'] = self.guess_X
    self.initial_guess['Vx'] = self.guess_Vx
    self.initial_guess['Vz'] = self.guess_Vz
    self.initial_guess['U'] = self.guess_U
    self.initial_guess['s'] = self.guess_s

    self.initialized = True

  """ ============================================================================================================ """

  def call(self, x:DM, Xref:DM, Uref:DM):
    """call Calls the component and computes a new control

    Args:
      x[DM] -- The current differential state estimate

    Returns:
      u[DM] -- The optimal current control
      result[dict] -- The optimizer's result
      stats[dict] -- The optimizer's statistics

    """
    assert self.initialized
    assert x.shape == (self.NX,1)
    assert Xref.shape == (self.NX,self.N+1), "Xref shape = " + str(Xref.shape)
    assert Uref.shape == (self.NU,self.N), "Uref shape = " + str(Uref.shape)
    if self.verbose: print("MPC called!")

    # Fetch member data
    ncol, NX, NZ = [self.ncol, self.NX, self.NZ]

    # Fetch previous data
    X = self.guess_X
    Vx = self.guess_Vx
    Vz = self.guess_Vz
    U = self.guess_U
    x0 = self.params_x0
    P = self.params_P

    """
    Every time a new state estimate arrives, the trajectory arrays are left-shifted.
    The (N+1)^th state is then computed using the previously stored N^th state and control. The collocation points
    in between step (N) and (N+1) are computed along the way. The newly computed values are then pushed into the 
    right hand side of the trajectory containers.
    """
    
    # Compute a new state by applying the reference control
    u0_k = Uref[:,-1]
    vx_k = DM.zeros((NX,1+ncol))
    vz_k = DM.zeros((NZ,1+ncol))
    vx_k[:,0] = Vx[-NX:,-1]
    vz_k[:,0] = Vz[-NZ:,-1]

    # Simulate one step
    for i in range(self.ncol):
      dt = self.collocation_dt[i]   
      vx_curr = vx_k[i * NX:(i + 1) * NX]
      vz_curr = vz_k[i * NZ:(i + 1) * NZ]
      vx_next, vz_next = self.simulateModel(vx_curr, vz_curr, u0_k, P, dt)
      vx_k[(i + 1) * NX:(i + 2) * NX] = vx_next
      vz_k[(i + 1) * NZ:(i + 2) * NZ] = vz_next

    # Left-shift arrays
    X[:, :-1] = X[:, 1:]
    Vx[:, :-1] = Vx[:, 1:]
    Vz[:, :-1] = Vz[:, 1:]
    U[:, :-1] = U[:, 1:]

    # Shift new values into arrays
    X[:, -1] = vx_k[-NX:]
    Vx[:, -1] = vx_k[NX:]
    Vz[:, -1] = vz_k[NZ:]
    U[:, -1] = DM(u0_k)
    
    ################################################
    ###            SOLVE THE PROBLEM             ###
    # Assign guess and parameters
    
    self.initial_guess['X'] = X
    self.initial_guess['Vx'] = Vx
    self.initial_guess['Vz'] = Vz
    self.initial_guess['U']  = U
    
    self.parameters['x0'] = x
    self.parameters['Xref'] = Xref
    self.parameters['Uref'] = Uref

    initial_guess = self.initial_guess
    parameters = self.parameters

    # Optimize!
    result, stats, dum,dum,dum = self.ocp.solve(self.initial_guess, self.parameters)

    # Grab the solution and store it
    self.initial_guess = result['w']
    self.guess_X = result['w']['X']
    self.guess_Vx = result['w']['Vx']
    self.guess_Vz = result['w']['Vz']
    self.guess_U = result['w']['U']
    self.params_x0 = x

    # Fetch current control and return
    uk = result['w']['U'][:,0]
    return uk, result, stats, initial_guess, parameters
