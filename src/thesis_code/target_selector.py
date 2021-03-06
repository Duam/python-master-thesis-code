from thesis_code.model import CarouselModel
import casadi as cas
from casadi import Function, jacobian, MX, DM, mtimes, vertcat
from thesis_code.utils.CollocationHelper import simpleColl
from typing import Callable
import numpy as np
from scipy.interpolate import interp1d
from thesis_code.utils.ParametricNLP import ParametricNLP


class Carousel_TargetSelector:
    def __init__(
            self,
            model: CarouselModel,
            time_per_revolution: float,
            samples_per_revolution: int,
            roll_reference: Callable[[float], float],
    ):
        """An offline target selector for the carousel model.
        :param model: The model
        :param time_per_revolution: The time it takes the carousel to complete one revolution.
        :param samples_per_revolution: The number of discretized samples one revolution is divided into.
        :param roll_reference: The airplane's roll angle reference as a function of the carousel's yaw angle.
            Must be periodic wrt 2*pi
        """
        # Fetch model
        self.model = model

        ## Fetch parameters ##
        num_states, NZ, NU, NY, NP = (model.NX(), model.NZ(), model.NU(), model.NY(), model.NP())
        self.N = samples_per_revolution
        self.T = time_per_revolution
        self.dt = self.T / self.N # Time step between samples

        #T = 2*np.pi / abs(model.constants['carousel_speed']) # Time of one revolution

        # Get carousel symbolics
        x,z,u,p = (model.x_aug_sym, model.z_sym, model.u_aug_sym, model.p_sym)
        ode = model.ode_aug_fun(x,z,u,p)
        alg = model.alg_aug_fun(x,z,u,p)
        out = model.out_aug_fun(x,z,u,p)

        # The "output" function
        self.r_fun = Function('r', [x], [x[0]])
        self.reference = [ reference(k*self.dt) for k in range(self.N+1) ]

        ## Create collocation integrator ##
        ncol = 3          # Number of collocation points per interval
        Ncol = N*ncol # Total number of collocation points
        tau_root = [0] + cas.collocation_points(ncol, 'radau')
        self.collocation_dt = [ self.dt * (tau_root[i+1] - tau_root[i]) for i in range(len(tau_root)-1) ]
        dae_dict = {'x': x, 'z': z, 'p': vertcat(p, u), 'ode': ode, 'alg': alg}
        G = simpleColl(dae_dict, tau_root, self.dt)
        print("Collocation equations created.")

        # Create the parametric NLP
        ocp = ParametricNLP('Carousel_TargetSelector')
        ocp.add_decision_var('X', (num_states, N + 1))   # Differential states
        ocp.add_decision_var('Z', (NZ,N+1)) # Algebraic states
        ocp.add_decision_var('Vx', (num_states, Ncol))  # Differential states at collocation points
        ocp.add_decision_var('Vz', (NZ,Ncol))  # Algebraic states at collocation points
        ocp.add_decision_var('U',  (NU,N)) # Controls
        ocp.add_parameter('x0', (num_states, 1)) # Initial state
        ocp.add_parameter('z0', (NZ,1)) # Initial algebraic state
        ocp.add_parameter('P', (NP,1)) # Model parameters
        ocp.bake_variables()

        # Fetch OCP symbolics
        X_sym = ocp.get_decision_var('X')
        Z_sym = ocp.get_decision_var('Z')
        Vx_sym = ocp.get_decision_var('Vx')
        Vz_sym = ocp.get_decision_var('Vz')
        U_sym = ocp.get_decision_var('U')
        x0_sym = ocp.get_parameter('x0')
        z0_sym = ocp.get_parameter('z0')
        P_sym = ocp.get_parameter('P')

        # Set up optimization problem
        # Periodicity
        ocp.add_equality('x(0)=x(N)', X_sym[:,-1] - X_sym[:,0])
        ocp.add_equality('z(0)=z(N)', Z_sym[:,-1] - Z_sym[:,0])

        # State evolution and residual cost
        residual_r = []# [self.r_fun(X_sym[:,0])-self.ref_fun(X_sym[:,0])]
        residual_r_names = [] #['res_r_0']
        G_colloc_ode = []
        G_colloc_ode_names = []
        G_colloc_alg = []
        G_colloc_alg_names = []
        for k in range(N):
            # Fetch current state, collocation nodes, control
            x0_k = X_sym[:,k]
            z0_k = Z_sym[:,k]
            vx_k = Vx_sym[:,ncol*k:ncol*(k+1)]
            vz_k = Vz_sym[:,ncol*k:ncol*(k+1)]
            u0_k = U_sym[:,k]

            # Create collocation equation and integrate
            xf_k, g_k = G(x0_k, vx_k, vz_k, vertcat(P_sym,u0_k))

            # Add residuals
            residual_r_names.append('res_r_'+str(k+1))
            #residual_r.append(self.r_fun(X_sym[:,k+1]) - self.ref_fun(X_sym[:,k+1]))
            residual_r.append(self.r_fun(X_sym[:,k+1]) - self.reference[k+1])

            # Add collocation equations
            for i in range(ncol):
                base = i * (num_states + NZ)
                G_colloc_ode_names.append('coll_'+str(k)+'_ode_node_'+str(i))
                G_colloc_alg_names.append('coll_'+str(k)+'_alg_node_'+str(i))
                G_colloc_ode.append(g_k[base   :base + num_states])
                G_colloc_alg.append(g_k[base + num_states:base + num_states + NZ])

            # Enforce collocation equations
            ocp.add_equality('coll_'+str(k), g_k)

            # Enforce continuity of states at sample points
            #ocp.add_equality('x_cont_'+str(k), X_sym[:,k+1] - xf_k)
            ocp.add_equality('x_cont_'+str(k), X_sym[:,k+1] - vx_k[:,-1])
            ocp.add_equality('z_cont_'+str(k), Z_sym[:,k+1] - vz_k[:,-1])

            # Control box constraints
            ocp.add_inequality('u_max_'+str(k), 1.0 - u0_k[0])
            ocp.add_inequality('u_min_'+str(k), u0_k[0])
            # Angle of attack constraints
            aoa_stall = model.constants['AoA_stall'] * model.constants['safety_factor_stall']
            x_inp = vertcat(X_sym[0,k], X_sym[1,k], 0.0, X_sym[2,k], X_sym[3,k], model.constants['carousel_speed'], X_sym[4,k])
            ocp.add_inequality('aoa_A_max_'+str(k), aoa_stall - model.alpha_A(x_inp,P_sym))
            ocp.add_inequality('aoa_A_min_'+str(k), model.alpha_A(x_inp,P_sym) + aoa_stall)
            ocp.add_inequality('aoa_E_max_'+str(k), aoa_stall - model.alpha_E(x_inp,P_sym))
            ocp.add_inequality('aoa_E_min_'+str(k), model.alpha_E(x_inp,P_sym) + aoa_stall)

        res = vertcat(*residual_r)

        # Set cost
        COST = 0
        COST += 0.5 * mtimes([res.T,res])
        #COST += 1e-3 * sum([mtimes([Z_sym[:,k].T,Z_sym[:,k]]) for k in range(N)])
        ocp.set_cost(COST)
        #print(COST.shape)

        # Jacobian of residuals
        w_ocp_sym = ocp.struct_w
        p_ocp_sym = ocp.struct_p
        GNJ = jacobian(res, w_ocp_sym)
        # Gauss-Newton hessian
        GNH = mtimes(GNJ.T, GNJ)
        # Create hessian approximation functor
        sigma = MX.sym('sigma')
        lamb = MX.sym('lambda',0,1)
        hess_lag = Function('GNH', [w_ocp_sym, p_ocp_sym, sigma, lamb], [sigma * GNH])

        # Initialize solver
        ocp.init(
            nlpsolver = 'ipopt',
            opts = {
                'hess_lag': hess_lag,
                'expand': True,
                'ipopt.print_level': 5,
                'print_time': 5,
                'ipopt.print_timing_statistics': 'no',
                'ipopt.sb': 'no',
            }
        )

        # Fetch NLP symbolics for easy function creation and evaluation
        w_ocp_sym = ocp.struct_w
        p_ocp_sym = ocp.struct_p

        # Create function objects for later analysis
        self.residual_r_funs = [ Function(residual_r_names[k], [w_ocp_sym,p_ocp_sym], [residual_r[k]]) for k in range(len(residual_r)) ]
        self.G_colloc_ode_funs = [ Function(G_colloc_ode_names[k], [w_ocp_sym,p_ocp_sym], [G_colloc_ode[k]]) for k in range(len(G_colloc_ode)) ]
        self.G_colloc_alg_funs = [ Function(G_colloc_alg_names[k], [w_ocp_sym,p_ocp_sym], [G_colloc_alg[k]]) for k in range(len(G_colloc_alg)) ]

        print("Solver initialized.")

        # Integrators for use in forward simulation
        dt = cas.MX.sym('dt')
        int_opts = {'number_of_finite_elements': ncol, 'tf': 1.0, 'expand': True}
        dae_dict = {'x': x, 'z': z, 'p': vertcat(dt, p, u), 'ode': dt*ode, 'alg': alg}
        self.integrator = cas.integrator('xnext','collocation',dae_dict, int_opts)

        # Create aliases
        self.N = N
        self.ncol = ncol
        self.Ncol = Ncol
        self.ocp = ocp
        self.initialized = False


    def simulateModel(self, x: DM, z: DM, u: DM, p: DM, dt: DM):
        step = self.integrator(x0=x,z0=z,p=vertcat(dt,p,u))
        return step['xf'], step['zf']

    def init(self, x0: DM, z0: DM, W: DM):
        print("Initializing Target selector..")
        assert not self.initialized, "Already initialized! Called twice?"

        print("Filling buffers with dummy values..")
        # Simulate N steps forward to fill the buffers
        X = DM.zeros((self.model.NX(),self.N+1))
        Z = DM.zeros((self.model.NZ(),self.N+1))
        Vx = DM.zeros((self.model.NX(),self.Ncol))
        Vz = DM.zeros((self.model.NZ(),self.Ncol))
        U = DM(np.repeat(self.model.u0(),self.N,axis=1))
        p = self.model.p0()

        # Do one initial step to get from X[:,0] to Vx[:,0]
        X[:,0] = DM(x0)
        Z[:,0] = DM(z0)
        dt = self.collocation_dt[0]
        Vx[:,0], Vz[:,0] = self.simulateModel(X[:,0], Z[:,0], U[:,0], p, dt)

        # Simulate to fill the horizon
        for k in range(1,self.Ncol):
            # Fetch stuff
            vx_k = Vx[:,k-1]
            vz_k = Vz[:,k-1]
            u0_k = U[:,np.mod(k,self.ncol)]

            # Choose correct dt and simulate
            dt = self.collocation_dt[np.mod(k,self.ncol)]
            Vx[:,k], Vz[:,k] = self.simulateModel(vx_k, vz_k, u0_k, p, dt)

            # Also fill X's and Z's
            if np.mod(k,self.ncol) == 0:
                X[:,int(k/self.ncol)] = Vx[:,k-1]
                Z[:,int(k/self.ncol)] = Vz[:,k-1]

        # Fill last X and Z
        X[:,-1] = Vx[:,-1]
        Z[:,-1] = Vz[:,-1]

        # Problem parameters:
        self.parameters = self.ocp.struct_p(0)
        self.parameters['x0'] = DM(x0)
        self.parameters['z0'] = DM(z0)
        self.parameters['P'] = DM(self.model.p0())

        # Buffer objects to hold the horizon information
        self.initial_guess = self.ocp.struct_w(0)
        self.initial_guess['X'] = DM(X)
        self.initial_guess['Z'] = DM(Z)
        self.initial_guess['Vx'] = DM(Vx)
        self.initial_guess['Vz'] = DM(Vz)
        self.initial_guess['U']  = DM(U)
        self.initialized = True


    def call(self):
        print("Calling Target Selector!")
        assert self.initialized
        result = self.ocp.solve(self.initial_guess, self.parameters)

        # Fetch solution and remove duplicate elements
        sol = result[0]
        Xsol = sol['w']['X'][:,:-1]
        Zsol = sol['w']['Z'][:,:-1]
        Usol = sol['w']['U']
        N = Xsol.shape[1]

        # Massage arrays so they are nicely interpolatable
        Xref = cas.horzcat(Xsol, Xsol[:,0])
        Zref = cas.horzcat(Zsol, Zsol[:,0])
        Uref = cas.horzcat(Usol)

        # Create interpolation functions
        axis = [ self.T * (float(k)/N) for k in range(N) ]
        axis_extended = axis + [self.T]
        self.Uref_fun = interp1d(axis, Uref, fill_value="extrapolate", kind="zero")
        self.Xref_fun = interp1d(axis_extended, Xref, kind="cubic")
        self.Zref_fun = interp1d(axis_extended, Zref, kind="cubic")
        return Xsol, Zsol, Usol


    def get_new_reference(self, t: float, dt: float, N: int):
        """Returns a number of reference points that the system shall visit.
        :param psi: The current carousel angle.
        :param dt: The sampling time.
        :param N: The amount of reference points to be returned.
        :returns: Lists of length N with states x and controls u that start at
            carousel angle psi and are evenly spaced in time with spacing dt
        """
        # Create the yaw-values that we will evaluate the interpolation function with
        t0 = t
        tN = t0 + (N+1) * dt

        # Predict the yaw-angles for X, Z and U (U: shifted by half of a bin)
        tAxis_XZ = np.linspace(t0, tN, N+1, endpoint=False)
        tAxis_U = [ t + dt/2. for t in tAxis_XZ ]

        # Map the angles into the interval [0,2*pi)
        tAxis_XZ_mod = [ np.mod(t,self.T) for t in tAxis_XZ ]
        tAxis_U_mod = [ np.mod(t,self.T) for t in tAxis_U ]

        # Compute the reference
        Xref = self.Xref_fun(tAxis_XZ_mod)
        Zref = self.Zref_fun(tAxis_XZ_mod)
        Uref = self.Uref_fun(tAxis_U_mod)

        # Return the reference
        return DM(Xref), DM(Zref), DM(Uref)
