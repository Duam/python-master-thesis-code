from thesis_code.model import CarouselModel
import casadi as cas
import numpy as np
from casadi import Function, DM, mtimes, vertcat, horzcat
from thesis_code.utils.CollocationHelper import simpleColl
from thesis_code.utils.ParametricNLP import ParametricNLP


class CarouselIdentificator:

    def __init__(
            self,
            model: CarouselModel,
            N: int,
            dt: float,
            do_compile: bool = False,
            expand: bool = False,
            fit_imu: bool = False
    ):
        """Carousel_Identificator A module for identifying the carousel's model parameters.
        Args:
            model[CarouselWhiteBoxModel] -- An instance of the carousel's whitebox model
            N[int] -- The number of samples
            dt[float] -- The timestep
            do_compile[bool] -- Compilation flag
            expand[bool] -- Expands the computation graph if set (sometimes solve speedup)
        """

        print("Creating idenficator.")
        self.do_compile = do_compile
        self.expand = expand
        self.model = model
        self.initialized = False

        # Fetch the given parameters as casadi expressions
        # Input sizes
        NX, NZ, NU, NY, NP = (model.NX(), model.NZ(), model.NU(), model.NY(), model.NP())
        self.NX, self.NZ, self.NU, self.NY, self.NP = [NX, NZ, NU, NY, NP]
        self.N = N
        self.dt = dt
        print("Parameters fetched.")

        # Prepare casadi expressions
        # Get carousel symbolics
        x_sym,z_sym,u_sym,p_sym = (model.x_aug_sym, model.z_sym, model.u_aug_sym, model.p_sym)
        sym_args = [x_sym,z_sym,u_sym,p_sym]
        ode_sym = model.ode_aug_fun(*sym_args)
        alg_sym = model.alg_aug_fun(*sym_args)
        out_sym = model.out_aug_fun(*sym_args)
        print("Casadi objects created.")

        # Create collocation integrator
        self.ncol = ncol = 3  # Number of collocation points per interval
        self.Ncol = Ncol = N * ncol  # Total number of collocation points
        tau_root = [0] + cas.collocation_points(ncol, 'radau')
        self.collocation_dt = [dt * (tau_root[i + 1] - tau_root[i]) for i in range(len(tau_root) - 1)]
        dae_dict = {'x': x_sym, 'z': z_sym, 'p': cas.vertcat(p_sym, u_sym), 'ode': ode_sym, 'alg': alg_sym }
        # Integrator for use in OCP
        G = simpleColl(dae_dict, tau_root, dt)
        print("Collocation equations created.")

        # Integrators for use in forward simulation
        dt = cas.MX.sym('dt')
        int_opts = {'number_of_finite_elements': ncol, 'tf': 1.0, 'expand': self.expand, 'jit': True}
        dae_dict = {'x': x_sym, 'z': z_sym, 'p': cas.vertcat(dt, p_sym, u_sym), 'ode': dt * ode_sym, 'alg': alg_sym}
        self.integrator = cas.integrator('xnext', 'collocation', dae_dict, int_opts)

        # Create the parametric NLP
        self.ocp = ocp = ParametricNLP('Carousel_Identificator_N'+str(N))
        ocp.add_decision_var('X', (NX, N + 1))  # Differential states
        ocp.add_decision_var('Z', (NZ, N + 1))  # Algebraic states
        ocp.add_decision_var('Vx', (ncol*NX, N))  # Differential states at collocation points
        ocp.add_decision_var('Vz', (ncol*NZ, N))  # Algebraic states at collocation points
        ocp.add_decision_var('P', (NP, 1))  # Model parameters
        ocp.add_parameter('U', (NU, N))  # Controls
        ocp.add_parameter('Y', (NY, N))  # Measurements
        ocp.add_parameter('x0', (NX, 1))  # Initial state
        ocp.add_parameter('P0', (NP, 1)) # Prior guess for model parameters
        ocp.add_parameter('Q', (NX, 1))  # Model confidence
        ocp.add_parameter('R', (NY, 1))  # Sensor confidence
        ocp.add_parameter('S', (NP, 1))  # Parameter prior confidence
        ocp.add_parameter('Q0', (NX,1)) # Initial state confidence
        ocp.bake_variables()

        # Fetch OCP symbolics
        X_sym = ocp.get_decision_var('X')
        Z_sym = ocp.get_decision_var('Z')
        Vx_sym = ocp.get_decision_var('Vx')
        Vz_sym = ocp.get_decision_var('Vz')
        P_sym = ocp.get_decision_var('P')
        U_sym = ocp.get_parameter('U')
        Y_sym = ocp.get_parameter('Y')
        x0_sym = ocp.get_parameter('x0')
        P0_sym = ocp.get_parameter('P0')
        Q_sym = ocp.get_parameter('Q')
        R_sym = ocp.get_parameter('R')
        S_sym = ocp.get_parameter('S')
        Q0_sym = ocp.get_parameter('Q0')
        print("Optimization variables created.")

        # Set up optimization problem
        xnext = cas.MX.sym('xnext', NX)
        x = cas.MX.sym('x', NX)
        z = cas.MX.sym('z', NZ)
        vx = cas.MX.sym('vx', ncol*NX)
        vz = cas.MX.sym('vz', ncol*NZ)
        u = cas.MX.sym('u', NU)
        p = cas.MX.sym('p', NP)
        y = cas.MX.sym('y', NY)
        Q = cas.MX.sym('Q', NX, 1)
        R = cas.MX.sym('R', NY, 1)

        # Create symbolic expressions
        g = G(x, vx.reshape((NX, ncol)), vz.reshape((NZ, ncol)), vertcat(p, u))[1]
        xf = G(x, vx.reshape((NX, ncol)), vz.reshape((NZ, ncol)), vertcat(p, u))[0]
        residual_x0 = X_sym[:,0] - x0_sym
        residual_p = P_sym - P0_sym
        residual_x = xnext - xf
        residual_y = self.model.out_aug_fun(x, z, u, p) - y
        cost_x = 0.5 * mtimes([ residual_x.T, cas.diag(Q), residual_x ])
        cost_y = 0.5 * mtimes([ residual_y.T, cas.diag(R), residual_y ])

        g_fun = Function('xf_k', [x,vx,vz,u,p], [g])
        residual_xy_fun = Function('residual', [xnext,x,z,vx,vz,u,y,p], [vertcat(residual_x,residual_y)])
        #residual_xy_fun = Function('residual', [xnext,x,z,vx,vz,u,y,p], [vertcat(residual_x)])
        weight_xy_fun = Function('weight', [Q,R], [vertcat(Q,R)])
        #weight_xy_fun = Function('weight', [Q,R], [vertcat(Q)])
        cost_x_fun = Function('cost_x', [xnext,x,vx,vz,u,p,Q], [cost_x])
        cost_y_fun = Function('cost_y', [x,z,u,p,y,R], [cost_y])

        max_pitch = 53 * (2*np.pi/360)
        min_pitch = -53 * (2*np.pi/360)
        max_roll = 135 * (2*np.pi/360)
        min_roll = -135 * (2*np.pi/360)
        max_aoa = model.constants['AoA_stall']
        min_aoa = - max_aoa
        x_model_in = vertcat(x[0],x[1],0.0,x[2],x[3],model.constants['carousel_speed'],x[4])

        max_roll_fun = Function('max_roll', [x,p], [max_roll - x[0]])
        min_roll_fun = Function('min_roll', [x,p], [(x[0] - min_roll)])
        max_pitch_fun = Function('max_pitch', [x,p], [max_pitch - x[1]])
        min_pitch_fun = Function('min_pitch', [x,p], [(x[1] - min_pitch)])
        max_aoa_a_fun = Function('max_aoa', [x,p], [max_aoa - model.alpha_A(x_model_in,p)])
        min_aoa_a_fun = Function('max_aoa', [x,p], [model.alpha_A(x_model_in,p) - min_aoa])


        # Create mapped versions of the functions
        #map_args = ['thread', N] # Effectively halves hessian computation time!
        map_args = ['openmp'] # Not so good
        #map_args = ['serial']
        g_map = g_fun.map(N, *map_args)
        alg_map = self.model.alg_aug_fun.map(N+1, *map_args)
        residual_xy_map = residual_xy_fun.map(N, *map_args)
        weight_xy_map = weight_xy_fun.map(N, *map_args)
        cost_x_map = cost_x_fun.map(N, *map_args)
        cost_y_map = cost_y_fun.map(N, *map_args)

        self.max_roll_map = max_roll_map = max_roll_fun.map(N + 1, *map_args)
        self.min_roll_map = min_roll_map = min_roll_fun.map(N + 1, *map_args)
        self.max_pitch_map = max_pitch_map = max_pitch_fun.map(N + 1, *map_args)
        self.min_pitch_map = min_pitch_map = min_pitch_fun.map(N + 1, *map_args)
        self.max_aoa_a_map = max_aoa_a_map = max_aoa_a_fun.map(N + 1, *map_args)
        self.min_aoa_a_map = min_aoa_a_map = min_aoa_a_fun.map(N + 1, *map_args)

        # Repeat "matrices"
        P_repN = cas.repmat(P_sym, 1, N)
        P_repNp1 = cas.repmat(P_sym, 1, N+1)
        Q_repN = cas.repmat(Q_sym, 1, N)
        R_repN = cas.repmat(R_sym, 1, N)

        # Compute total residual and weight, weight the residuals and set the cost
        residual_xy_size = (NX + NY) * N
        #residual_xy_size = (NX) * N
        residual_xy_eval = residual_xy_map(X_sym[:,1:], X_sym[:,:-1], Z_sym[:,:-1], Vx_sym, Vz_sym, U_sym, Y_sym, P_repN)
        weight_xy_eval = weight_xy_map(Q_repN,R_repN)
        total_residual = vertcat(residual_x0, residual_p, residual_xy_eval.reshape((residual_xy_size,1)))
        total_weight = cas.diag(vertcat(Q0_sym, S_sym, weight_xy_eval.reshape((residual_xy_size,1))))
        total_weight_sqrt = cas.sqrt(total_weight)
        total_weighted_residual = mtimes(total_weight_sqrt, total_residual)
        COST = 0.5 * mtimes(total_weighted_residual.T, total_weighted_residual)
        ocp.set_cost(COST)

        # Set the constraints
        self.ocp.add_equality('alg', alg_map(X_sym, Z_sym, horzcat(U_sym, U_sym[:, -1]), P_repNp1))
        self.ocp.add_equality('coll', g_map(X_sym[:, :-1], Vx_sym, Vz_sym, U_sym, P_repN))
        
        self.ocp.add_inequality('max_roll', max_roll_map(X_sym, P_repNp1))
        self.ocp.add_inequality('min_roll', min_roll_map(X_sym, P_repNp1))
        self.ocp.add_inequality('max_pitch', max_pitch_map(X_sym, P_repNp1))
        self.ocp.add_inequality('min_pitch', min_pitch_map(X_sym, P_repNp1))
        self.ocp.add_inequality('max_aoa_a', max_aoa_a_map(X_sym, P_repNp1))
        self.ocp.add_inequality('min_aoa_a', min_aoa_a_map(X_sym, P_repNp1))
        
        # Parameter constraints
        if not fit_imu:
            fix_params = []
            fix_params += [13, 14, 15] # IMU position
            fix_params += [16, 17, 18] # IMU orientation
            for k in range(len(fix_params)):
                idx = fix_params[k]
                self.ocp.add_equality('p'+str(idx), P_sym[idx] - P0_sym[idx])

        # Jacobian of residuals
        w_ocp_sym = self.ocp.struct_w
        p_ocp_sym = self.ocp.struct_p
        GNJ = cas.jacobian(total_weighted_residual, w_ocp_sym)
        # Gauss-Newton hessian
        GNH = cas.triu(mtimes(GNJ.T, GNJ))
        # Create hessian approximation functor
        sigma = cas.MX.sym('sigma')
        lamb = cas.MX.sym('lambda')
        self.hess_lag = Function('GNH', [w_ocp_sym, p_ocp_sym, sigma, lamb], [sigma * GNH])

        ## Initialize solver
        print("Initializing solver..")
        ocp.init(
            nlpsolver='ipopt',
            opts={
                #'hess_lag': self.hess_lag,
                'ipopt.linear_solver': 'ma97',
                'verbose': True,
                'expand': self.expand,
                'jit': False,
                'ipopt.print_level': 5,
                'print_time': 1,
                'ipopt.print_timing_statistics': 'yes',
                'ipopt.sb': 'yes',
                'ipopt.max_iter': 100000,
                'verbose_init': True
            },
            create_analysis_functors = False,
            compile_solver = self.do_compile
        )
        print("Solver initialized.")

        print(self.ocp)

        # Compute covariance of solution
        jac_R = cas.jacobian(total_weighted_residual, w_ocp_sym)
        jac_g = cas.jacobian(self.ocp.nlp['g'][:self.ocp.num_eq_constraints], w_ocp_sym)
        print(jac_R.shape)
        print(jac_g.shape)
        print(w_ocp_sym.shape)
        n = w_ocp_sym.shape[0]
        m1 = jac_R.shape[0]
        m2 = jac_g.shape[0]
        mat = cas.inv(cas.blockcat([
            [cas.mtimes(jac_R.T, jac_R), jac_g.T],
            [jac_g, cas.DM.zeros((m2,m2)) ]    
        ]))
        self.cov_fun = cas.Function('Sigma', [w_ocp_sym, p_ocp_sym], [mat[:n,:n]])
        print("Identificator created.")

    def simulateModel(self, x: DM, z: DM, u: DM, p: DM, dt: DM):
        """simulateModel Simulates the model for one step
        Args:
            x[DM] -- The current differential state
            z[DM] -- A prior guess for the current algebraic state
            u[DM] -- The current control
            p[DM] -- The model parameters
            dt[DM] -- The timestep
        Returns:
            [xf,zf] -- The differential and algebraic state at the end of the integration interval
        """
        step = self.integrator(x0=x, z0=z, p=vertcat(dt, p, u))
        return step['xf'], step['zf']

    def init(
        self,
        x0: DM,
        U: DM,
        Y: DM,
        Q: DM,
        R: DM,
        S: DM,
        Q0: DM,
    ):
        """Initializes the identificator. Has to be called after creation and before call().
        :param x0: Initial state.
        :param U: Control input trajectory.
        :param Y: Measurement output trajectory.
        :param Q: Model confidence matrix.
        :param R: Sensor confidence matrix.
        :param S: Parameter's prior guess confidence matrix.
        :param Q0: Initial state confidence matrix.
        :returns: Nothing.
        """
        print("Initializing Identificator..")
        assert not self.initialized, "Already initialized! Called twice?"
        assert x0.shape == (self.NX,1), "x0 shape = " + str(x0.shape)
        assert U.shape == (self.NU, self.N), "U shape = " + str(U.shape)
        assert Y.shape == (self.NY, self.N), "Y shape = " + str(Y.shape)
        assert Q.shape == (self.NX, 1), "Q shape = " + str(Q.shape)
        assert Q0.shape == (self.NX, 1), "Q shape = " + str(Q0.shape)
        assert R.shape == (self.NY, 1), "R shape = " + str(R.shape)
        assert S.shape == (self.NP, 1), "S shape = " + str(S.shape)

        print("\t Filling buffers with dummy values..")
        # Simulate N steps forward to fill the buffers
        X = DM.zeros((self.NX, self.N + 1)) # DM.zeros(self.ocp.get_parameter('X').shape)
        Z = DM.zeros((self.NZ, self.N + 1))
        Vx = DM.zeros((self.ncol*self.NX, self.N))
        Vz = DM.zeros((self.ncol*self.NZ, self.N))
        P0 = self.model.p0()
        
        # Prepare forward simulation
        vx_k = DM.zeros(((1 + self.ncol)*self.NX, 1))
        vz_k = DM.zeros(((1 + self.ncol)*self.NZ, 1))
        vx_k[-self.NX:] = X[:, 0] = x0
        vz_k[-self.NZ:] = Z[:, 0] = DM.zeros((self.NZ,1))

        # Simulate to fill the horizon
        for k in range(self.N):
            vx_k[:self.NX] = vx_k[-self.NX:]
            vz_k[:self.NZ] = vz_k[-self.NZ:]
            u0_k = U[:, k]

            print("Sample " + str(k) + ": u = " + str(u0_k) + ", x = " + str(vx_k[:self.NX]) + ", z = " + str(vz_k[:self.NZ]))

            # Simulate one collocation interval
            for i in range(self.ncol):
                dt = self.collocation_dt[i]
                vx_curr = vx_k[i * self.NX:(i + 1) * self.NX]
                vz_curr = vz_k[i * self.NZ:(i + 1) * self.NZ]
                print("Node " + str(i) + ": vx = " + str(vx_curr) + ", vz = " + str(vz_curr))
                vx_next, vz_next = self.simulateModel(vx_curr, vz_curr, u0_k, P0, dt)
                vx_k[(i + 1) * self.NX:(i + 2) * self.NX] = vx_next
                vz_k[(i + 1) * self.NZ:(i + 2) * self.NZ] = vz_next

            # Write back new states
            X[:, k + 1] = vx_k[-self.NX:]
            Z[:, k + 1] = vz_k[-self.NZ:]
            Vx[:, k] = vx_k[self.NX:]
            Vz[:, k] = vz_k[self.NZ:]
        
        # Initialize X
        get_roll = lambda yr: self.model.constants['roll_sensor_offset'] - yr
        get_pitch = lambda yp: self.model.constants['pitch_sensor_offset'] - yp
        for k in range(self.N):
            X[0,k] = get_roll(Y[0,k])
            X[1,k] = get_pitch(Y[1,k])
        
        # Copy last state
        X[:,-1] = X[:,-2]

        for k in range(self.N):
            X[2,k] = (X[0,k+1] - X[0,k])/self.dt
            X[3,k] = (X[1,k+1] - X[1,k])/self.dt

        X[:,-1] = X[:,-2]
        
        # Solve algebraic constraint for Z
        for k in range(self.N+1):
            xk = X[:,k]
            uk = U[:,k] if k < self.N else U[:,self.N-1]
            p = P0
            zk = cas.SX.sym('z',self.NZ)
            fun = Function('g', [zk], [self.model.alg_aug_fun(xk,zk,uk,p)])
            rootfinder = cas.rootfinder('root', 'newton', fun)
            Z[:,k] = rootfinder(DM.zeros((self.NZ,1)))        

        # Interpolate X and Z to get Vx and Vz
        taxis = np.linspace(0,self.N*self.dt,self.N+1)
        dXdt = np.diff(X,axis=1) / self.dt
        dZdt = np.diff(Z,axis=1) / self.dt
        for k in range(self.N):
            xk = X[:,k]
            zk = Z[:,k]
            for i in range(self.ncol):
                Vx[i*self.NX:(i+1)*self.NX,k] = xk + sum(self.collocation_dt[:i+1]) * dXdt[:,k]
                Vz[i*self.NZ:(i+1)*self.NZ,k] = zk + sum(self.collocation_dt[:i+1]) * dZdt[:,k]

        print("\t Setting problem parameters and initial guess..")
        # Problem parameters:
        self.parameters = self.ocp.struct_p(0)
        self.parameters['U'] = DM(U)
        self.parameters['Y'] = DM(Y)
        self.parameters['x0'] = DM(x0)
        self.parameters['Q'] = DM(Q)
        self.parameters['Q0'] = DM(Q0)
        self.parameters['R'] = DM(R)
        self.parameters['S'] = DM(S)
        self.parameters['P0'] = DM(P0)

        # Buffer objects to hold the horizon information
        self.initial_guess = self.ocp.struct_w(0)
        self.initial_guess['X'] = DM(X)
        self.initial_guess['Z'] = DM(Z)
        self.initial_guess['Vx'] = DM(Vx)
        self.initial_guess['Vz'] = DM(Vz)
        self.initial_guess['P'] = DM(P0)
        self.initialized = True

    def call(self):
        """Solves the identification_imu problem and returns the result
        :returns: A tuple [P, X, Z, result, stats] with
            Parameters, Differential states, Algebraic states, original result, statistics
        """
        print("Identificator called! ")
        assert self.initialized

        # Optimize!
        result, stats, dum, dum, dum = self.ocp.solve(self.initial_guess, self.parameters)

        # Grab the solution and store it
        x = result['w']['X']
        z = result['w']['Z']
        p = result['w']['P']

        # Increase call counter and return
        return p, x, z, result, stats