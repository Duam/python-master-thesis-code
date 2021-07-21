from thesis_code.carousel_model import CarouselWhiteBoxModel
from thesis_code.components.carousel_ekf import Carousel_EKF
import casadi as cas
from casadi import Function, jacobian, DM, MX, mtimes, horzcat, vertcat
from thesis_code.utils.CollocationHelper import simpleColl
import numpy as np
import time
from thesis_code.utils.ParametricNLP import ParametricNLP


class Carousel_MHE:

    def __init__(self, model: CarouselWhiteBoxModel, N: int, dt: float, verbose:bool=False, do_compile:bool=False, expand=False):

        print("====================================")
        print("===         Creating MHE         ===")
        print("====================================")

        self.verbose = verbose
        self.do_compile = do_compile
        self.expand = expand
        self.model = model

        ## Fetch the given parameters as casadi expressions ##
        # Input sizes
        self.NX = NX = model.NX()  # Differential state vector size
        self.NZ = NZ = model.NZ()  # Algebraic state vector size
        self.NU = NU = model.NU()  # Control vector size
        self.NP = NP = model.NP()  # Parameter vector size
        self.NY = NY = model.NY()  # Output vector size

        # The other parameters
        self.N = N      # Estimation horizon
        self.dt = dt    # Sample time

        print("Parameters fetched.")

        ## Prepare casadi expressions ##
        # Fetch carousel symbolics
        """
        dae = model.dae
        x_sym = vertcat(*dae.x)
        z_sym = vertcat(*dae.z)
        u_sym = vertcat(*dae.u)
        p_sym = vertcat(*dae.p)
        ode_sym = vertcat(*dae.ode)
        alg_sym = vertcat(*dae.alg)
        quad_sym = vertcat(*dae.quad)
        out_sym = vertcat(*dae.ydef)
    
        # Create functionals
        args = [x_sym,z_sym,u_sym,p_sym]
        self.ode_fun = Function('ode', args, [ode_sym], ['x','z','u','p'], ['f']).expand()
        self.alg_fun = Function('alg', args, [alg_sym], ['x','z','u','p'], ['g']).expand()
        self.out_fun = Function('out', args, [out_sym], ['x','z','u','p'], ['h']).expand()
    
        # Augment carousel symbolics
        self.NX = NX = 5
        self.NU = NU = 1
        x_sym = MX.sym('x', NX, 1) # x = [roll,pitch,roll_rate,pitch_rate,deflection]
        u_sym = MX.sym('u', NU, 1) # u = [deflection_setpoint]
        args = [x_sym,z_sym,u_sym,p_sym]
        args_aug = [
          vertcat(x_sym[0], x_sym[1], 0.0, x_sym[2], x_sym[3], model.constants['carousel_speed'], x_sym[4]),
          z_sym,
          vertcat(u_sym[0], model.constants['carousel_speed']),
          p_sym
        ]
    
        ode = self.ode_fun(*args_aug)
        ode_sym = vertcat(ode[0],ode[1],ode[3],ode[4],ode[6])
        alg_sym = self.alg_fun(*args_aug)
        out_sym = self.out_fun(*args_aug)
    
        # Re-create functionals
        self.ode_fun = Function('ode', args, [ode_sym])
        self.alg_fun = Function('alg', args, [alg_sym])
        self.out_fun = Function('out', args, [out_sym])
        """
        NX, NZ, NU, NY, NP = (model.NX(), model.NZ(), model.NU(), model.NY(), model.NP())
        self.NX, self.NZ, self.NU, self.NY, self.NP = (NX, NZ, NU, NY, NP)
        x_sym, z_sym, u_sym, p_sym = (model.x_aug_sym, model.z_sym, model.u_aug_sym, model.p_sym)
        ode_sym = model.ode_aug_fun(x_sym, z_sym, u_sym, p_sym)
        alg_sym = model.alg_aug_fun(x_sym, z_sym, u_sym, p_sym)
        out_sym = model.out_aug_fun(x_sym, z_sym, u_sym, p_sym)
        print("Casadi objects created.")

        ## Create collocation integrator ##
        ncol = 3          # Number of collocation points per interval
        Ncol = N*ncol # Total number of collocation points
        tau_root = [0] + cas.collocation_points(ncol, 'radau')
        self.collocation_dt = [ dt * (tau_root[i+1] - tau_root[i]) for i in range(len(tau_root)-1) ]
        dae_dict = {
            'x': x_sym, 'z': z_sym, 'p': cas.vertcat(p_sym, u_sym),
            'ode': ode_sym, 'alg': alg_sym#, 'quad': quad_sym
        }
        # Integrator for use in OCP
        G = simpleColl(dae_dict, tau_root, dt)
        print("Collocation equations created.")

        # Create the parametric NLP
        ocp = ParametricNLP('Carousel_MHE_N'+str(N)+"_NY"+str(NY))
        ocp.add_decision_var('X', (NX,N+1))   # Differential states
        ocp.add_decision_var('Z', (NZ,N+1)) # Algebraic states
        ocp.add_decision_var('Vx', (ncol*NX,N))  # Differential states at collocation points
        ocp.add_decision_var('Vz', (ncol*NZ,N))  # Algebraic states at collocation points
        #ocp.add_decision_var('z0', (NZ,1))
        ocp.add_parameter('U',  (NU,N)) # Past controls
        ocp.add_parameter('Y',  (NY,N)) # Past measurements
        ocp.add_parameter('x0', (NX,1)) # Arrival cost state estimate

        #ocp.add_parameter('z0', (NZ,1))
        ocp.add_parameter('S', (NX,NX)) # Arrival cost covariance
        ocp.add_parameter('R', (NY,1)) # Measurement noise covariance
        ocp.add_parameter('Q', (NX,1)) # Process noise covariance
        ocp.add_parameter('P', (NP,1)) # Model parameters
        ocp.bake_variables()

        # Fetch OCP symbolics
        X_sym = ocp.get_decision_var('X')
        Z_sym = ocp.get_decision_var('Z')
        Vx_sym = ocp.get_decision_var('Vx')
        Vz_sym = ocp.get_decision_var('Vz')
        #z0_sym = ocp.get_decision_var('z0')
        U_sym = ocp.get_parameter('U')
        Y_sym = ocp.get_parameter('Y')
        x0_sym = ocp.get_parameter('x0')
        S_sym = ocp.get_parameter('S')
        R_sym = ocp.get_parameter('R')
        Q_sym = ocp.get_parameter('Q')
        P_sym = ocp.get_parameter('P')
        print("Optimization variables created.")

        # Set up optimization problem
        xnext = cas.MX.sym('xnext', NX)
        x = cas.MX.sym('x', NX)
        z = cas.MX.sym('z', NZ)
        vx = cas.MX.sym('vx', ncol * NX)
        vz = cas.MX.sym('vz', ncol * NZ)
        u = cas.MX.sym('u', NU)
        p = cas.MX.sym('p', NP)
        y = cas.MX.sym('y', NY)
        Q = cas.MX.sym('Q', NX, 1)
        R = cas.MX.sym('R', NY, 1)

        # Create symbolic expressions
        g = G(x, vx.reshape((NX, ncol)), vz.reshape((NZ, ncol)), vertcat(p, u))[1]
        xf = G(x, vx.reshape((NX, ncol)), vz.reshape((NZ, ncol)), vertcat(p, u))[0]
        residual_x0 = X_sym[:,0] - x0_sym
        residual_x = xnext - xf
        residual_y = self.model.out_aug_fun(x, z, u, p) - y

        # Create functions
        g_fun = Function('xf_k', [x, vx, vz, u, p], [g])
        residual_xy_fun = Function('residual', [xnext, x, z, vx, vz, u, y, p], [vertcat(residual_x, residual_y)])
        weight_xy_fun = Function('weight', [Q, R], [vertcat(Q, R)])

        max_pitch = 53 * (2*np.pi/360)
        min_pitch = -53 * (2*np.pi/360)
        max_roll = 135 * (2*np.pi/360)
        min_roll = -135 * (2*np.pi/360)

        max_roll_fun = Function('max_roll', [x,p], [max_roll - x[0]])
        min_roll_fun = Function('min_roll', [x,p], [(x[0] - min_roll)])
        max_pitch_fun = Function('max_pitch', [x,p], [max_pitch - x[1]])
        min_pitch_fun = Function('min_pitch', [x,p], [(x[1] - min_pitch)])
        max_trim_fun = Function('max_trim', [x], [1 - x[4]])
        min_trim_fun = Function('min_trim', [x], [x[4] - 0])

        # Create mapped versions of the functions
        #map_args = ['thread', N]  # Effectively halves hessian computation time!
        map_args = ['openmp']
        #map_args = []
        g_map = g_fun.map(N, *map_args)
        alg_map = self.model.alg_aug_fun.map(N + 1, *map_args)
        residual_xy_map = residual_xy_fun.map(N, *map_args)
        weight_xy_map = weight_xy_fun.map(N, *map_args)

        self.max_roll_map = max_roll_map = max_roll_fun.map(N + 1, *map_args)
        self.min_roll_map = min_roll_map = min_roll_fun.map(N + 1, *map_args)
        self.max_pitch_map = max_pitch_map = max_pitch_fun.map(N + 1, *map_args)
        self.min_pitch_map = min_pitch_map = min_pitch_fun.map(N + 1, *map_args)
        self.max_trim_map = max_trim_map = max_trim_fun.map(N + 1, *map_args)
        self.min_trim_map = min_trim_map = min_trim_fun.map(N + 1, *map_args)

        # Repeat "matrices" so that the threads use separate resources
        P_repN = cas.repmat(P_sym, 1, N)
        P_repNp1 = cas.repmat(P_sym, 1, N+1)
        Q_repN = cas.repmat(Q_sym, 1, N)
        R_repN = cas.repmat(R_sym, 1, N)

        NALL = N * (NX + NY) + NX
        DUMMY = cas.SX.sym('DUMMY', NALL, NALL)
        cholesky = cas.Function('chol', [DUMMY], [cas.chol(DUMMY)])

        # Compute total residual and weight, weight residuals and set cost
        residual_xy_size = (NX + NY) * N
        residual_xy_eval = residual_xy_map(X_sym[:, 1:], X_sym[:, :-1], Z_sym[:, :-1], Vx_sym, Vz_sym, U_sym, Y_sym, P_repN)
        weight_xy_eval = weight_xy_map(Q_repN, R_repN)
        total_residual = vertcat(residual_x0, residual_xy_eval.reshape((residual_xy_size, 1)))
        total_weight = cas.diag(vertcat(cas.diag(S_sym), weight_xy_eval.reshape((residual_xy_size, 1))))
        #total_weight = cas.diag(vertcat(DM.ones((NX,1)), weight_xy_eval.reshape((residual_xy_size, 1))))
        total_weight[0:NX,0:NX] = S_sym
        #total_weight_sqrt = cas.sqrt(total_weight)
        #total_weight_sqrt = cas.chol(total_weight)
        total_weight_sqrt = cholesky(total_weight)
        total_weighted_residual = mtimes(total_weight_sqrt, total_residual)
        COST = 0.5 * mtimes(total_weighted_residual.T, total_weighted_residual)
        ocp.set_cost(COST)

        # Equality constraints
        ocp.add_equality('alg', alg_map(X_sym, Z_sym, horzcat(U_sym, U_sym[:, -1]), P_repNp1))
        ocp.add_equality('coll', g_map(X_sym[:, :-1], Vx_sym, Vz_sym, U_sym, P_repN))

        ocp.add_inequality('max_roll', max_roll_map(X_sym, P_repNp1))
        ocp.add_inequality('min_roll', min_roll_map(X_sym, P_repNp1))
        ocp.add_inequality('max_pitch', max_pitch_map(X_sym, P_repNp1))
        ocp.add_inequality('min_pitch', min_pitch_map(X_sym, P_repNp1))
        ocp.add_inequality('max_trim', max_trim_map(X_sym))
        ocp.add_inequality('min_trim', min_trim_map(X_sym))

        # Jacobian of residuals
        w_ocp_sym = ocp.struct_w
        p_ocp_sym = ocp.struct_p
        GNJ = jacobian(total_weighted_residual, w_ocp_sym)
        # Gauss-Newton hessian
        GNH = cas.triu(mtimes(GNJ.T, GNJ))
        # Create hessian approximation functor
        sigma = MX.sym('sigma')
        lamb = MX.sym('lambda',0,1)
        self.hess_lag = Function('GNH', [w_ocp_sym, p_ocp_sym, sigma, lamb], [sigma * GNH])

        ## Initialize solver
        ocp.init(
            nlpsolver = 'ipopt',
            opts = {
                #'hess_lag': self.hess_lag,
                #'ipopt.linear_solver': 'ma27',
                'ipopt.linear_solver': 'ma57',
                #'ipopt.linear_solver': 'ma77',
                #'ipopt.linear_solver': 'ma86',
                #'ipopt.linear_solver': 'ma97',
                #'ipopt.ma57_automatic_scaling':'no',
                'expand': self.expand,
                'jit': False,# do_compile, # See do_compile flag
                'ipopt.print_level': 5 if verbose else 0,
                'print_time': 1 if verbose else 0,
                'ipopt.print_timing_statistics': 'yes' if verbose else 'no',
                'ipopt.sb': 'yes',
                #'ipopt.max_iter': 20,
                #'ipopt.tol': 1e-4,
                'jit_options': {'flags':['-O3']},
                #'ipopt.max_cpu_time': 20 * 1e-3,
                'ipopt.max_cpu_time': 30 * 1e-3,
            },
            create_analysis_functors=False,
            compile_solver=do_compile
        )

        print("Solver initialized.")

        # Integrators for use in forward simulation
        dt = cas.MX.sym('dt')

        int_opts = {'number_of_finite_elements': 1, 'tf': 1.0, 'expand': True, 'jit': True, 'rootfinder':'kinsol',
                    'jit_options':{'flags':['-O3']}}
        #int_opts = {'tf': 1.0, 'expand': True, 'jit': True, 'jit_options':{'flags':['-O3']}}

        #int_opts = {'tf': 1.0, 'expand': True, 'jit': True, 'jit_options':{'flags':['-O3']}}
        dae_dict = {
            'x': x_sym, 'z': z_sym, 'p': cas.vertcat(dt, p_sym, u_sym),
            'ode': dt*ode_sym, 'alg': alg_sym#, 'quad': dt*quad_sym
        }
        self.integrator = cas.integrator('xnext','collocation',dae_dict, int_opts)
        #self.integrator = cas.integrator('xnext','idas',dae_dict, int_opts)

        # Create an EKF
        self.ekf = Carousel_EKF(self.model, self.dt, self.verbose, do_compile)

        # Create aliases
        self.N = N
        self.ncol = ncol
        self.Ncol = Ncol
        self.ocp = ocp

        self.initialized = False
        print("====================================")
        print("===       MHE creation done      ===")
        print("====================================")

    def simulateModel(self, x:DM, z:DM, u:DM, p:DM, dt:DM):
        step = self.integrator(x0=x,z0=z,p=vertcat(dt,p,u))
        return step['xf'], step['zf']

    ##
    # @brief Init function. To be called after equality
    #        constraints have been added
    # @param x0 The initial state estimate (np.array)
    # @param P0 Covariance matrix for the initial state estimate (np.array)
    # @param Q The State noise covariance (np.array)
    # @param R The measurement noise covariance (np.array)
    ##
    def init(self, x0_est:DM, z0_est:DM, Q:DM, R:DM, S0:DM):
        print("Initializing MHE..")
        assert not self.initialized, "Already initialized! Called twice?"

        assert Q.shape == (self.NX,1), "Q shape = " + str(Q.shape)
        assert R.shape == (self.NY,1), "R shape = " + str(R.shape)
        assert S0.shape == (self.NX,1), "S0 shape = " + str(S0.shape)

        # Initialize EKF
        self.ekf.init(x0 = x0_est, P0 = cas.diag(1/S0), Q = cas.diag(1/Q), R = cas.diag(1/R))

        print("\t Filling buffers with dummy values..")
        # Simulate N steps forward to fill the buffers
        X = DM.zeros((self.NX,self.N+1))
        Z = DM.zeros((self.NZ,self.N+1))
        Vx = DM.zeros((self.ncol*self.NX,self.N))
        Vz = DM.zeros((self.ncol*self.NZ,self.N))
        U = DM(np.repeat(self.model.u0()[0],self.N,axis=1))
        Y = DM.zeros((self.NY,self.N))
        p = self.model.p0()

        # Prepare forward simulation
        vx_k = DM.zeros(((1 + self.ncol)*self.NX,1))
        vz_k = DM.zeros(((1 + self.ncol)*self.NZ,1))
        vx_k[-self.NX:] = X[:,0] = x0_est
        vz_k[-self.NZ:] = Z[:,0] = z0_est

        # Simulate to fill the horizon
        for k in range(self.N):
            vx_k[:self.NX] = vx_k[-self.NX:]
            vz_k[:self.NZ] = vz_k[-self.NZ:]
            u0_k = U[:,k]

            # Simulate one collocation interval
            for i in range(self.ncol):
                dt = self.collocation_dt[i]
                vx_curr = vx_k[i * self.NX:(i + 1) * self.NX]
                vz_curr = vz_k[i * self.NZ:(i + 1) * self.NZ]
                vx_next, vz_next = self.simulateModel(vx_curr, vz_curr, u0_k, p, dt)
                vx_k[(i + 1) * self.NX:(i + 2) * self.NX] = vx_next
                vz_k[(i + 1) * self.NZ:(i + 2) * self.NZ] = vz_next

            # Write back new states
            X[:, k + 1] = vx_k[-self.NX:]
            Z[:, k + 1] = vz_k[-self.NZ:]
            Y[:, k] = self.model.out_aug_fun(X[:, k], Z[:, k], u0_k, p)
            Vx[:, k] = vx_k[self.NX:]
            Vz[:, k] = vz_k[self.NZ:]

        print("\t Setting problem parameters and initial guess..")
        # Problem parameters:
        self.params = {
            'U': DM(U),
            'Y': DM(Y),
            'x0': DM(x0_est),
            'S': cas.diag(S0),
            'R': DM(R),
            'Q': DM(Q),
            'P': DM(self.model.p0())
        }
        self.params_U = DM(U)
        self.params_Y = DM(Y)
        self.params_x0 = DM(x0_est)
        self.params_Q = DM(Q)
        self.params_R = DM(R)
        self.params_S = cas.diag(S0)
        self.params_P = DM(self.model.p0())

        self.parameters = self.ocp.struct_p(0)
        self.parameters['U']  = self.params_U
        self.parameters['Y']  = self.params_Y
        self.parameters['x0'] = self.params_x0
        self.parameters['Q'] = self.params_Q
        self.parameters['R'] = self.params_R
        self.parameters['S'] = self.params_S
        self.parameters['P'] = self.params_P

        # Buffer objects to hold the horizon information
        self.guess = {
            'X': DM(X),
            'Z': DM(Z),
            'Vx': DM(Vx),
            'Vz': DM(Vz)
        }
        self.guess_X = DM(X)
        self.guess_Z = DM(Z)
        self.guess_Vx = DM(Vx)
        self.guess_Vz = DM(Vz)

        self.initial_guess = self.ocp.struct_w(0)
        self.initial_guess['X'] = self.guess_X
        self.initial_guess['Z'] = self.guess_Z
        self.initial_guess['Vx'] = self.guess_Vx
        self.initial_guess['Vz'] = self.guess_Vz

        self.initialized = True
        self.callCount = 0

    ##
    # @brief Estimates the current state (Filter mode)
    # @param u The current control
    # @param y The current measurement (caused by current control)
    # @return The state estimate at the current time
    ##
    def call(self, u, y, verbose:bool=False):
        startTime = time.time()
        assert self.initialized
        u = DM(u)
        y = DM(y)
        assert u.shape == (self.NU,1), "u shape = " + str(u.shape)
        assert y.shape == (self.NY,1), "y shape = " + str(y.shape)

        if self.verbose: print("MHE called! ", self.callCount)
        veryVerbose = self.verbose and verbose

        if veryVerbose: print("\t Fetching data..")
        k = self.callCount

        # Fetch previous data
        X = self.guess_X
        Z = self.guess_Z
        Vx = self.guess_Vx
        Vz = self.guess_Vz
        U = self.params_U
        Y = self.params_Y
        x0 = self.params_x0
        P = self.params_P
        S = self.params_S
        Q = self.params_Q

        print(self.params_Y)
        print(self.params_R)

        """
        X = self.initial_guess['X']
        Z = self.initial_guess['Z']
        Vx = self.initial_guess['Vx']
        Vz = self.initial_guess['Vz']
        U = self.parameters['U']
        Y = self.parameters['Y']
        x0 = self.parameters['x0']
        S = self.parameters['S']
        P = self.parameters['P']
        """

        if veryVerbose:
            print("X (prior) =", X)
            print("Z (prior) =", Z)
            print("Vx (prior) =", Vx)
            print("Vz (prior) =", Vz)

        # While the buffer is still filling up, use simulated dummy values
        if k < self.N:
            """
            While the MHE's horizon buffer is not full yet, we have to supply some dummy values.
            Ideally, one would simply solve the full-information problem N times (N being the horizon
            size), increase the problem size until the buffer is full, and then continue with
            solving the moving-horizon problem. 
            Here we go a different route: Every time the mhe is called and the buffer is not full,
            we use the most recent state and control to simulate the rest of the horizon, including
            pseudo-measurements. This gives us perfectly consistent trajectories that do not lead to
            an increase in the cost function. In the language of optimization, this means that the cost
            is optimal along the directions of the dummy-variables. Only the variables that have been
            pushed into the queue in the past should be varied.
            """
            if self.verbose:
                print("\t Buffer filling up.. ", k+1, " of ", self.N)
                print("u(k) = " + str(u))
                print("x(k) = " + str(X[:,k]))


            # Overwrite current dummy input with current actual input
            U[:,k] = DM(u)

            # Prepare collocation containers
            vx_k = DM.zeros(((1 + self.ncol) * self.NX,1))
            vz_k = DM.zeros(((1 + self.ncol) * self.NZ,1))
            vx_k[-self.NX:] = X[:, k]
            vz_k[-self.NZ:] = Z[:, k]
            for kbar in range(k,self.N):
                # Set interval's initial state
                vx_k[:self.NX] = vx_k[-self.NX:]
                vz_k[:self.NZ] = vz_k[-self.NZ:]
                u0_k = U[:,kbar]

                # Simulate collocation nodes
                for i in range(self.ncol):
                    dt = self.collocation_dt[i]
                    vx_curr = vx_k[i * self.NX:(i + 1) * self.NX]
                    vz_curr = vz_k[i * self.NZ:(i + 1) * self.NZ]
                    vx_next, vz_next = self.simulateModel(vx_curr, vz_curr, u0_k, P, dt)
                    vx_k[(i + 1) * self.NX:(i + 2) * self.NX] = vx_next
                    vz_k[(i + 1) * self.NZ:(i + 2) * self.NZ] = vz_next

                # Write back new states
                X[:, k + 1] = vx_k[-self.NX:]
                Z[:, k + 1] = vz_k[-self.NZ:]
                #Y[:, k] = self.out_fun(X[:, k], Z[:, k], u0_k, P)
                Y[:, k] = self.model.out_aug_fun(X[:, k], Z[:, k], u0_k, P)
                Vx[:, k] = vx_k[self.NX:]
                Vz[:, k] = vz_k[self.NZ:]

                #print(Vx.shape)

            # Overwrite current simulated measurement with current actual measurement
            Y[:,k] = DM(y)

        else:
            """
            Once the buffer is full, we can finally start solving the moving-horizon problem.
            Every time a new control-measurement pair arrives, the trajectory arrays are left-shifted.
            The most recent control (the input to this function) and the most recent state estimate (indexed 
            with N before the shift) are then used to compute a new state estimate and also all the collocation
            points along the way. The newly computed values are then pushed into the right hand side of the
            trajectory containers.
            """
            #print("Normal call!")
            u0_k = DM(u)

            # Update arrival cost
            x0, P0 = self.ekf(Z[:,0], U[:,0], Y[:,1], self.dt)
            S = cas.inv(P0)
            Q = (Q + cas.diag(S))/2

            # Prepare forward simulation
            vx_k = DM.zeros(((1 + self.ncol) * self.NX, 1))
            vz_k = DM.zeros(((1 + self.ncol) * self.NZ, 1))
            vx_k[:self.NX] = X[:, -1]
            vz_k[:self.NZ] = Z[:, -1]

            """
            # Simulate for one collocation interval
            for i in range(self.ncol):
              dt = self.collocation_dt[i]
              vx_curr = vx_k[i * self.NX:(i + 1) * self.NX]
              vz_curr = vz_k[i * self.NZ:(i + 1) * self.NZ]
              #print("Node " + str(i) + ": vx = " + str(vx_curr) + ", vz = " + str(vz_curr))
              vx_next, vz_next = self.simulateModel(vx_curr, vz_curr, u0_k, P, dt)
              vx_k[(i + 1) * self.NX:(i + 2) * self.NX] = vx_next
              vz_k[(i + 1) * self.NZ:(i + 2) * self.NZ] = vz_next
            """
            #"""
            # Simulate one step and interpolate to get the collocation nodes
            vx_k = DM.zeros((self.ncol * self.NX, 1))
            vz_k = DM.zeros((self.ncol * self.NZ, 1))
            x_next, z_next = self.simulateModel(X[:,-1], Z[:,-1], u0_k, P, self.dt)
            dXdt = (x_next - X[:,-1]) / self.dt
            dZdt = (z_next - Z[:,-1]) / self.dt
            for i in range(self.ncol):
                vx_k[i*self.NX:(i+1)*self.NX] = X[:,-1] + sum(self.collocation_dt[:i+1]) * dXdt
                vz_k[i*self.NZ:(i+1)*self.NZ] = Z[:,-1] + sum(self.collocation_dt[:i+1]) * dZdt
            #
            """
            x_next = X[:, -1]
            z_next = Z[:, -1]
            vx_k = Vx[:,-1]
            vz_k = Vz[:,-1]
            """

            # Left-shift arrays
            X[:,:-1] = X[:,1:]
            Z[:,:-1] = Z[:,1:]
            U[:,:-1] = U[:,1:]
            Y[:,:-1] = Y[:,1:]
            Vx[:, :-1] = Vx[:, 1:]
            Vz[:, :-1] = Vz[:, 1:]

            # Shift new values into arrays
            U[:,-1] = DM(u)
            Y[:,-1] = DM(y)
            #X[:, -1] = vx_k[-self.NX:]
            #Z[:, -1] = vz_k[-self.NZ:]
            #Vx[:, -1] = vx_k[self.NX:]
            #Vz[:, -1] = vz_k[self.NZ:]
            X[:, -1] = x_next
            Z[:, -1] = z_next
            Vx[:, -1] = vx_k
            Vz[:, -1] = vz_k

        ################################################
        ###            SOLVE THE PROBLEM             ###
        # Assign guess and parameters
        if veryVerbose: print("\t Assigning guess and parameters..")
        self.initial_guess['X'] = X
        self.initial_guess['Z'] = Z
        self.initial_guess['Vx'] = Vx
        self.initial_guess['Vz'] = Vz
        self.parameters['U']  = U
        self.parameters['Y']  = Y
        self.parameters['x0'] = x0
        self.parameters['Q'] = Q
        self.parameters['S'] = S
        """
        
        self.guess['X'] = X
        self.guess['Z'] = Z
        self.guess['Vx'] = Vx
        self.guess['Vz'] = Vz
        self.params['U'] = U
        self.params['Y'] = Y
        self.params['x0'] = x0
        self.params['Q'] = Q
        self.params['S'] = S
        """
        #print(self.guess['X'])
        #print(self.guess['Z'])
        #print(self.guess['Vx'])
        #print(self.guess['Vz'])

        #self.initial_guess = DM(self.guess.values())
        #guess = cas.vertcat(*[cas.vertcat(self.guess[key].reshape((np.prod(self.guess[key].shape),1))) for key in self.guess.keys()])
        #param = cas.vertcat(*[cas.vertcat(self.params[key].reshape((np.prod(self.params[key].shape),1))) for key in self.params.keys()])
        """
        print(guess)
        print(self.ocp.struct_w(guess)['X'])
        print(self.ocp.struct_w(guess)['Z'])
        print(self.ocp.struct_w(guess)['Vx'])
        print(self.ocp.struct_w(guess)['Vz'])
        """
        #quit(0)
        # Optimize!
        if veryVerbose: print("\t Starting optimization..")
        result, stats, dum,dum,dum = self.ocp.solve(self.initial_guess, self.parameters)
        #result, stats, dum,dum,dum = self.ocp.solve(guess, param)
        sol = result['w']
        Xsol = sol['X']
        Zsol = sol['Z']

        # Grab the solution and store it.
        # If the solver didn't succeed, just use the integrated state
        i = min([ k+1, self.N ])
        if stats['return_status'] == 'Solve_Succeeded':
            #xk = result['w']['X'][:,i]
            #zk = result['w']['Z'][:,i]
            xk = Xsol[:,i]
            zk = Zsol[:,i]
        else:
            xk = X[:,i]
            zk = Z[:,i]

        self.guess_X = Xsol
        self.guess_Z = Zsol
        self.guess_Vx = sol['Vx']
        self.guess_Vz = sol['Vz']
        self.params_U = U
        self.params_Y = Y
        self.params_x0 = x0
        self.params_S = S
        self.params_Q = Q

        # Increase call counter and return
        self.callCount += 1

        endTime = time.time()
        call_duration = endTime - startTime
        return xk, zk, result, stats, call_duration
