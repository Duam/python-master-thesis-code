#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/06/24
  Brief: This file tests the nonlinear moving horizon estimator (NMHE)
          for iterative learning of periodic disturbance parameters. In this
          version, the periodic parameters are represented in the time-domain.
          It is tested against the whitebox carousel model.
"""

import casadi as cas
from thesis_code.carousel_model import get_steady_state
from thesis_code.carousel_model import CarouselWhiteBoxModel
import thesis_code.models.carousel_whitebox_viz as cplot
import numpy as np
np.set_printoptions(linewidth=np.inf)
import matplotlib.pyplot as plt
import pprint
from thesis_code.utils.signals import rectangle

""" =========================================       MODEL SETUP       ============================================= """

# Get defaul model parameters
print("Creating model parameters..")
model_params = CarouselWhiteBoxModel.getDefaultParams()

# Create a carousel model
print("Creating model..")
model = CarouselWhiteBoxModel(model_params)

# Print parameters
print(" ---------------------- Model Parameters ---------------------- ")
pprint.pprint(model.params)
print(" -------------------------------------------------------------- ")

# Fetch system sizes
print("Creating system handles..")
NX = model.NX() # Differential states
NZ = model.NZ() # Algebraic states
NU = model.NU() # Control inputs
NP = model.NP() # Parameters (constant in time)
NY = model.NY() # Measurement outputs

# The semi-explicit DAE
dae = model.dae

# All symbolics
x_sym = cas.vertcat(*dae.x)
z_sym = cas.vertcat(*dae.z)
u_sym = cas.vertcat(*dae.u)
p_sym = cas.vertcat(*dae.p)
ode_sym = cas.vertcat(*dae.ode)
alg_sym = cas.vertcat(*dae.alg)
quad_sym = cas.vertcat(*dae.quad)

# Function objects
ode_fun = cas.Function('ode', [x_sym,z_sym,u_sym,p_sym], [cas.vertcat(*dae.ode)])
alg_fun = cas.Function('alg', [x_sym,z_sym,u_sym,p_sym], [cas.vertcat(*dae.alg)])
out_fun = cas.Function('out', [x_sym,z_sym,u_sym,p_sym], [model.out(x_sym,z_sym,p_sym)])

""" ===================================       SIMULATION SETUP       ======================================== """

# System meta parameters
dt = 0.10
T_set = np.pi # Settling time before any controls are applied
T_sim = T_set + 3 * np.pi
N_set = int(T_set/dt)
N_sim = int(T_sim/dt)
N_per = 30
N_mhe = 4

# Create integrator
dae_dict = {'x': x_sym, 'ode': ode_sym, 'alg': alg_sym, 'z': z_sym, 'p': cas.vertcat(p_sym, u_sym)}
int_opts = {'number_of_finite_elements': 1, 'output_t0': True, 'tf':dt}
integrator = cas.integrator('xnext', 'collocation', dae_dict, int_opts)

# Create control values
print("Creating inputs..")
Us_sim = np.repeat(model.u0(), N_sim, axis=1)
Us_sim[0,:] = [0.5 for k in range(N_set)] + [0.5+0.175*rectangle(k,N_per) for k in range(N_sim-N_set)]

# Create simulation containers
Xs_sim = np.zeros((NX,N_sim))
Zs_sim = np.zeros((NZ,N_sim))
Ys_sim = np.zeros((NY,N_sim))

# Set initial state (= steady state)
x_ss, z_ss = get_steady_state(model)
Xs_sim[:,0] = x_ss.full().flatten()
Zs_sim[:,0] = z_ss.full().flatten()
Ys_sim[:,0] = model.out(x_ss,z_ss,model.p0()).full().flatten()

# Create simulation handles
def simstep (x, u):
  # Fetch current state and controls
  xk = Xs_sim[:,k]
  pk = cas.vertcat(model.p0(), Us_sim[:,k])
  # Simulate one step
  step = integrator(x0=xk,p=pk)
  # Fetch result
  xf = step['xf'][:,1].full().flatten()
  zf = step['zf'][:,1].full().flatten()
  # Compute output
  yf = out_fun(xf,zf,Us_sim[:,k],model.p0()).full().flatten()
  return xf, zf, yf

# Start simulation
print("Starting simulation..")
print("x0 = x_ss =", Xs_sim[:,0])
print("z0 = z_ss =", Zs_sim[:,0])
print("y0 =", Ys_sim[:,0])
print("Go!")
for k in range(N_sim-1):
  print("Simulating step", k+1, " of", N_sim-1)
  xnext, znext, ynext = simstep(Xs_sim[:,k], Us_sim[:,k])
  Xs_sim[:,k+1] = xnext
  Zs_sim[:,k+1] = znext
  Ys_sim[:,k+1] = ynext

if True:
  cplot.plotStates(Xs_sim,Us_sim[:,:-1],dt,model)
  cplot.plotAerodynamics(Xs_sim,Us_sim[:,:-1],dt,model)
  cplot.plotMoments(Xs_sim,Zs_sim,Us_sim[:,:-1],dt,model)
  cplot.plotFlightTrajectory(Xs_sim,model)

plt.show()

""" =========================================       MHE SETUP       ============================================= """


""" TODO 
- How to put collocation stuff in MHE? Apparently do by hand -.-
- p are time-fixed parameters. w shall be periodic disturbances. there should also be periodic parameters? Check with notes
- Period should be discretized spatially in psi. Implement in MHE. Add modulus-module?
"""