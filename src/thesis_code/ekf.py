from thesis_code.model import CarouselModel
import casadi as cas
from casadi import Function, jacobian, DM, MX, mtimes
from scipy import linalg


class CarouselEKF:

    def __init__(
            self,
            model: CarouselModel,
            dt: float,
            verbose: bool = False,
            do_compile: bool = False
    ):
        """<TODO>
        :param model:
        :param dt:
        :param verbose:
        :param do_compile:
        """

        # Fetch
        self.model = model
        self.dt = dt
        self.verbose = verbose
        self.do_compile = do_compile
        jit_args = {'jit': do_compile, 'jit_options':{'flags':['-O3']}}
        param = self.model.params

        # Fetch sizes
        self.NX = NX = self.model.NX()
        self.NU = NU = self.model.NX()
        self.NP = NP = self.model.NP()
        self.NY = NY = self.model.NY()

        # Fetch carousel symbolics
        x,z,u,p = (model.x_aug_sym, model.z_sym, model.u_aug_sym, model.p_sym)
        ode = model.ode_aug_fun(x,z,u,p)
        alg = model.alg_aug_fun(x,z,u,p)
        out = model.out_aug_fun(x,z,u,p)

        dt = MX.sym('dt')
        args = [x, z, u, p]

        # Create an integrator
        k1 = model.ode_aug_fun(x, z, u, p)
        k2 = model.ode_aug_fun(x + dt/2 * k1, z, u, p)
        k3 = model.ode_aug_fun(x + dt/2 * k2, z, u, p)
        k4 = model.ode_aug_fun(x + dt * k3, z, u, p)
        xnext = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.xnext = Function('xnext', args+[dt], [xnext], jit_args)
        xnext = self.xnext(*args, dt)

        # Compute derivatives
        self.dFdx_fun = Function('dFdx', args + [dt], [jacobian(xnext, x)], jit_args)
        self.dhdx_fun = Function('dhdx', args, [jacobian(out, x)], jit_args)

        # Set some parameters
        self.uk = 0.5
        self.zk = cas.DM.zeros((3,1))
        self.params = self.model.p0()
        self.I = DM.eye(NX)
        self.initialized = False

    def init (self, x0: DM, P0: DM, Q: DM, R: DM):
        assert not self.initialized
        assert DM(x0).shape == (self.NX,1), "x0 shape = " + str(DM(x0).shape)
        assert DM(P0).shape == (self.NX,self.NX), "P0 shape = " + str(DM(P0).shape)
        assert DM(Q).shape == (self.NX,self.NX), "Q shape = " + str(DM(Q).shape)
        assert DM(R).shape == (self.NY,self.NY), "R shape = " + str(DM(R).shape)
        self.xest = x0
        self.Pest = P0
        self.Q = Q
        self.R = R
        self.initialized = True

    def predict(self, z: DM, u: DM, dt: DM):
        assert self.initialized
        assert DM(u).shape == (1,1), "u shape = " + str(DM(u).shape)
        P = self.Pest
        xest = self.xest
        params = self.params
        args = [xest, z, u, params]

        # State sensitivity
        F = self.dFdx_fun(*args, dt)

        # Predict state
        self.xnext_pred = self.xnext(*args, dt)

        # Predict covariance
        self.Pnext_pred = mtimes([F, P, F.T]) + self.Q

        # Store control
        self.uk = u
        self.zk = z

    def update(self, y: DM):
        assert self.initialized
        assert DM(y).shape == (self.NY,1), "y shape = " + str(DM(y).shape)
        xpred = self.xnext_pred
        Ppred = self.Pnext_pred
        p = self.params

        # Output jacobian
        H = self.dhdx_fun(xpred, self.zk, self.uk, p)

        # Innovation (measurement pre-fit residual)
        res = y - self.model.out_aug_fun(xpred, self.zk, self.uk, p)

        # Innovation covariance
        S = mtimes([H, Ppred, H.T]) + self.R
        Sinv = linalg.inv(S.full())

        # Optimal kalman gain
        K = mtimes([Ppred, H.T, Sinv])

        # Updated state estimate
        self.x = xpred + mtimes(K, res)

        # Updated state estimate covariance
        self.P = mtimes(self.I - mtimes(K,H), Ppred)

        return self.x, self.P

    def __call__(self, z: DM, u:DM, y:DM, dt:DM):
        if False:
            print("z = " + str(z))
            print("u = " + str(u))
            print("y = " + str(y))

        self.predict(z, u, dt)
        x, P = self.update(y)
        return x,P