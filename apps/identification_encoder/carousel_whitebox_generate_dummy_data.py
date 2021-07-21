#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/06/24
  Brief: This script creates dummy data for dynamic identification_imu
"""

""" =========================================       PREPARATION       ============================================= """

# For the system definition
import casadi as cas
from casadi import vertcat
# For the model
from thesis_code.model import CarouselModel
from thesis_code.model import get_steady_state
# For math
import numpy as np
np.set_printoptions(linewidth=np.inf)
# For plotting
# For colorful text
# For deepcopy
# For nice dictionary prints
import pprint
# For signal generation
from thesis_code.utils.signals import rectangle, triangle
# For everything else

""" =========================================       MODEL SETUP       ============================================= """

print("======================================= GENERATE DUMMY DATA =======================================")

# Get defaul model parameters
print("Creating model parameters..")
model_params = CarouselModel.getDefaultParams()
model_params['m'] = 1.22
model_params['C_LA_0'] = 0.2
model_params['mu_theta'] = 0.8

# Create a carousel model
print("Creating model..")
model = CarouselModel(model_params)

# Print parameters
print(" ---------------------- Model Parameters ---------------------- ")
pprint.pprint(model.params)
print(" -------------------------------------------------------------- ")

# Create the output function: Output elevation and pitch
x_sym = vertcat(*model.dae.x)
z_sym = vertcat(*model.dae.z)
out = cas.Function('out', [x_sym,z_sym], [vertcat(x_sym[0],x_sym[1],x_sym[2])])

""" =======================================       SIMULATION SETUP       =========================================== """

# The steady state will be our initial state
print("Computing steady state..")
x_ss, z_ss = get_steady_state(model)
print("x_ss =", x_ss)
print("z_ss =", z_ss)

print("Preparing simulation..")

# State data is published with a frequency of 20 hz
dt = 1./20.

# Create inputs for each experiment
N_exp = 1
N_per_exp = 20 #5120
Us_sine = np.repeat(model.u0(), N_exp*N_per_exp, axis=1)
Us_tri  = np.repeat(model.u0(), N_exp*N_per_exp, axis=1)
Us_rect = np.repeat(model.u0(), N_exp*N_per_exp, axis=1)

for i in range(N_exp):
  for k in range(N_per_exp):
    Us_sine[0,i*N_per_exp+k] = 0.5 + 0.5 * np.sin(2**(i+1) * np.pi * k/N_per_exp)
    Us_tri[0,i*N_per_exp+k] = triangle(k, N_per_exp/2**(i))
    Us_rect[0,i*N_per_exp+k] = rectangle(k, N_per_exp/2**(i))

# Collect inputs
Us = np.concatenate([Us_sine,Us_tri,Us_rect], axis=1)
N = Us.shape[1]
T = N * dt

# Fetch parameter vector
p = np.concatenate([np.array(val).flatten() for val in model.params.values()])

# Prepare simulation containers
Zs = np.zeros((model.NZ(),N+1))
Xs = np.zeros((model.NX(),N+1))
Ys = np.zeros((3,N+1))

# Initialize simulation
Xs[:,0] = x_ss.full().flatten()
Zs[:,0] = z_ss.full().flatten()
Ys[:,0] = out(Xs[:,0],Zs[:,0]).full().flatten()

""" ===========================================       SIMULATION       ============================================= """

# Simulate!
print("Simulating for", N, "steps.")
for k in range(N):
  print("Simulating step", k, "of", N)

  # Fetch controls, state
  u = Us[:,k]
  x = Xs[:,k]
  z = Zs[:,k]

  # Simulate one step forward
  step = model.simstep(x,u,dt,z0=z)
  xnext = step['xf'][:,1]
  znext = step['zf'][:,1]
  Xs[:,k+1] = xnext.full().flatten()
  Zs[:,k+1] = znext.full().flatten()

  # Compute output
  Ys[:,k+1] = out(Xs[:,k+1],Zs[:,k+1]).full().flatten()

print("Simulation done.")

""" =========================================       POST-PROCESSING       ========================================== """

import json, csv, time

timebase = int(time.time() * 1e9)

# Store parameters in json file
param_filename = 'TRUE_PARAMS.json'
with open(param_filename, 'w') as outfile:
  json.dump(model.params, outfile, indent=4)

print("Parameters written to", param_filename)

# Store inputs in csv file
input_filename = 'set_VESTIBULUS_2_KINEOS_4_SETTABLE_SETPOINT.csv'
with open(input_filename, 'w') as outfile:
  writer = csv.writer(outfile)
  writer.writerow(['timestamp','values'])
  for k in range(N):
    timestamp = timebase + int(k * dt * 1e9)
    value = Us[0,k]
    writer.writerow([timestamp,value])

print("Actuator signals written to", input_filename)

# Store elevation and pitch encoder positions in csv file
angles_filename = 'ANGLE_sampled.csv'
with open(angles_filename, 'w') as outfile:
  writer = csv.writer(outfile)
  writer.writerow(['timestamp','elevation','rotation'])
  for k in range(N):
    timestamp = timebase + int(k * dt * 1e9)
    elevation = Ys[0,k]
    rotation = Ys[1,k]
    writer.writerow([timestamp,elevation,rotation])

print("Elevation and rotation written to", angles_filename)

# Store carousel encoder position in csv file
carousel_encoder_filename = 'CAROUSEL_CAROUSELENCODERPOSITION.csv'
with open(carousel_encoder_filename, 'w') as outfile:
  writer = csv.writer(outfile)
  writer.writerow(['timestamp','position'])
  for k in range(N):
    timestamp = timebase + int(k * dt * 1e9)
    position = Ys[2,k]
    writer.writerow([timestamp,position])

print("Carousel encoder positions written to", carousel_encoder_filename)
