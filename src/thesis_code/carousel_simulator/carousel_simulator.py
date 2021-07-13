import numpy as np
from numpy.random import multivariate_normal
from casadi import MX, DM, vertcat, integrator, Function
from thesis_code.models.carousel_whitebox import CarouselWhiteBoxModel

class Carousel_Simulator:
  def __init__(self, model:CarouselWhiteBoxModel, 
                     x0:DM, 
                     z0:DM=None, 
                     process_noise_mean:DM=None, 
                     process_noise_covar:DM=None,
                     measurement_noise_mean:DM=None,
                     measurement_noise_covar:DM=None,
                     jit:bool=False,
                     expand:bool=True):

    """ Carousel_Simulator simulates the carousel whitebox model

      xdot = f(x,z,u,p) + w
         0 = g(x,z,u,p)
         y = h(x,z,u,p) + v

    Args:
      model[CarouselWhiteBoxModel] -- The model to simulate
      x0[DM] -- The initial (differential) state
      z0[DM] -- The initial (algebraic) state (optional)
      mean[DM] -- The noise mean (optional)
      covar[DM] -- The noise covariance matrix (optional)

    """

    print("Setting up carousel simulator ..")

    # Fetch model
    self.model = model
    self.NX = self.model.NX()
    self.NZ = self.model.NZ()
    self.NU = self.model.NU()
    self.NP = self.model.NP()
    self.NY = self.model.NY()

    # Set initial stuff, noise parameters and normal parameters
    self.xk = x0
    self.zk = z0 if z0 is not None else DM.zeros((self.NZ,1))
    self.w_mean = process_noise_mean if process_noise_mean is not None else np.zeros((self.NX))
    self.w_covar = process_noise_covar if process_noise_covar is not None else np.zeros((self.NX,self.NX))
    self.v_mean = measurement_noise_mean if measurement_noise_mean is not None else np.zeros((self.NY))
    self.v_covar = measurement_noise_covar if measurement_noise_covar is not None else np.zeros((self.NY,self.NY))
    self.params = vertcat(*[np.array(val).flatten() for val in self.model.params.values()])

    # Create an integrator
    x,z,u,p = (model.x_aug_sym, model.z_sym, model.u_aug_sym, model.p_sym)
    ode = model.ode_aug_fun(x,z,u,p)
    alg = model.alg_aug_fun(x,z,u,p)
    out = model.out_aug_fun(x,z,u,p)
    w = MX.sym('w', self.NX)
    v = MX.sym('v', self.NY)
    dt = MX.sym('dt')
    
    dae_dict = {'x': x, 'z': z, 'p': vertcat(dt,p,u,w), 'ode': dt * (ode + w), 'alg': alg }
    integrator_opts = {'number_of_finite_elements': 1, 'tf':1.0, 'expand': expand, 'jit': jit }
    self.integrate_step = integrator('xnext', 'collocation', dae_dict, integrator_opts)
    self.measure = Function('out', [x,z,u,p,v], [out+v], ['x','z','u','p','v'], ['y'], {'jit':jit})

    print("Carousel simulator set up.")


  def simstep(self, u:DM, dt:float):
    """ simstep Simulates one step

    Args:
      u[DM] -- The applied control
      dt[float] -- The timestep

    Result:
      (xf,zf,y0) -- The differential state, algebraic state at the next timestep and 
      the measurement at the current timestep
    
    """
    assert DM(u).shape == (1,1)

    # Create process noise and measurement noise
    w = multivariate_normal(self.w_mean, self.w_covar)
    v = multivariate_normal(self.v_mean, self.v_covar)

    # Integrate!
    result = self.integrate_step(
      x0 = self.xk, 
      p = vertcat(dt, self.params, u, w), 
      z0 = self.zk
    )


    #print(self.integrate_step)

    # Store results and return
    ycurr = self.measure(self.xk, self.zk, u, self.params, v)
    xnext = self.xk = result['xf']
    znext = self.zk = result['zf']
    return xnext, znext, ycurr


  