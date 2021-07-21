import numpy as np
from numpy.random import multivariate_normal
from casadi import MX, DM, vertcat, integrator, Function
from thesis_code.carousel_model import CarouselWhiteBoxModel


class CarouselSimulator:
    def __init__(
            self,
            model: CarouselWhiteBoxModel,
            x0: DM,
            z0: DM = None,
            process_noise_mean: DM = None,
            process_noise_covar: DM = None,
            measurement_noise_mean: DM = None,
            measurement_noise_covar: DM = None,
            jit: bool = False,
            expand: bool = True) -> None:
        """Carousel_Simulator simulates the carousel whitebox model

          xdot = f(x,z,u,p) + w
             0 = g(x,z,u,p)
             y = h(x,z,u,p) + v

        :param model: The model to simulate
        :param x0: The initial (differential) state
        :param z0: The initial (algebraic) state (optional)
        :param process_noise_mean: The process noise mean (optional)
        :param process_noise_covar: The process noise covariance matrix (optional)
        :param measurement_noise_mean: The measurement noise mean (optional)
        :param measurement_noise_covar: The measurement noise covariance matrix (optional)
        :returns None:
        """

        # Fetch model
        self.model = model
        self.NX = self.model.NX()
        self.NZ = self.model.NZ()
        self.NU = self.model.NU()
        self.NP = self.model.NP()
        self.NY = self.model.NY()

        # Set initial stuff, noise parameters and normal parameters
        self.xk = x0
        self.zk = z0 if z0 is not None else DM.zeros((self.NZ, 1))
        self.w_mean = process_noise_mean if process_noise_mean is not None else np.zeros(self.NX)
        self.w_covar = process_noise_covar if process_noise_covar is not None else np.zeros((self.NX, self.NX))
        self.v_mean = measurement_noise_mean if measurement_noise_mean is not None else np.zeros(self.NY)
        self.v_covar = measurement_noise_covar if measurement_noise_covar is not None else np.zeros((self.NY, self.NY))
        self.params = vertcat(*[np.array(val).flatten() for val in self.model.params.values()])

        # Create an integrator
        x, z, u, p = (model.x_aug_sym, model.z_sym, model.u_aug_sym, model.p_sym)
        ode = model.ode_aug_fun(x, z, u, p)
        alg = model.alg_aug_fun(x, z, u, p)
        out = model.out_aug_fun(x, z, u, p)
        w = MX.sym('w', self.NX)
        v = MX.sym('v', self.NY)
        dt = MX.sym('dt')

        self.compute_next_state = integrator(
            'compute_next_state',
            'collocation',
            {
                'x': x,
                'z': z,
                'p': vertcat(dt, p, u, w),
                'ode': dt * (ode + w),
                'alg': alg,
            },
            {
                'number_of_finite_elements': 1,
                'tf': 1.0,
                'expand': expand,
                'jit': jit,
            }
        )

        self.compute_measurement = Function(
            'out',
            [x, z, u, p, v],
            [out + v],
            ['x', 'z', 'u', 'p', 'v'],
            ['y'],
            {'jit': jit}
        )

    def simulate_timestep(self, control: DM, timestep: float):
        """simulate_timestep Simulates one step,
        starting from the "current" time and moving the system forward until the "next" time.
        :param control: The applied control
        :param timestep: The duration to simulate
        :returns: A tuple containing [0] the differential state at the next time,
                                     [1] the algebraic state at the next time and
                                     [2] the measurement at the current time.
        """
        assert DM(control).shape == (1, 1)

        # Create process noise and measurement noise
        w = multivariate_normal(self.w_mean, self.w_covar)
        v = multivariate_normal(self.v_mean, self.v_covar)

        # Integrate!
        result = self.compute_next_state(
            x0=self.xk,
            p=vertcat(timestep, self.params, control, w),
            z0=self.zk
        )

        # Store results and return
        current_measurement = self.compute_measurement(self.xk, self.zk, control, self.params, v)
        next_differential_state = self.xk = result['xf']
        next_algebraic_state = self.zk = result['zf']
        return next_differential_state, next_algebraic_state, current_measurement
