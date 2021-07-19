import numpy as np
import casadi as cas
from thesis_code.carousel_model import CarouselWhiteBoxModel
import thesis_code.models.carousel_whitebox_viz as viz
from thesis_code.carousel_simulator import Carousel_Simulator
from thesis_code.components.carousel_ekf import Carousel_EKF
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=np.inf)


# Load identified parameters
import json
param = {}
path = "../apps/identification/params_identified_dynamic_imu.json"
print("Loading identified parameter set " + path)
with open(path, 'r') as file:
    param = json.load(file)

# Create model
model = CarouselWhiteBoxModel(param)
x0_est,z0_est,u0_est = model.get_steady_state()
NX = model.NX() - 2
NU = model.NU() - 1
NY = model.NY()

# Simulation meta-parameters
T_per = 2*np.pi / abs(param['carousel_speed'])

M_sim = 2
T_sim = M_sim * T_per
dt_sim = 1. / 50.
N_sim = int(np.ceil(T_sim / dt_sim))
N_sim = 250
#dt_sim = T_sim / N_sim
jit = False

# Create an estimator
x0_est_inp = cas.vertcat(x0_est[0],x0_est[1],x0_est[3],x0_est[4],x0_est[6])
ekf = Carousel_EKF(model = model, dt = dt_sim, verbose = True, do_compile = jit)
Q = 1e-6 * cas.DM.eye(NX)
R = 1e-3 * cas.DM.eye(NY)
ekf.init(x0 = x0_est_inp, P0 = Q, Q = Q, R = R)

# Create evaluation containers
Us_sim = [ 0.5 for k in range(N_sim) ]
Xs_sim = [ x0_est + 0.1 * cas.DM.rand((7,1))]
Zs_sim = [ z0_est ]
Ys_sim = [ ]
Xs_est = [ x0_est ]

# Create simulator
Q_sim = 1e-8 * np.eye((NX+2))
Q_sim[2] = 0.0
Q_sim[5] = 0.0
R_sim = 1e-3 * np.eye((NY))
simulator = Carousel_Simulator(
  model = model, x0 = Xs_sim[0], z0 = Zs_sim[0],
  process_noise_mean = 0e0 * np.ones((NX+2)),
  process_noise_covar = Q_sim,
  measurement_noise_mean = 0e0 * np.ones((NY)),
  measurement_noise_covar = R_sim,
  jit = jit
)

print("Starting simulation. Initial conditions:")
print("x0 =", Xs_sim[0])
print("z0 =", Zs_sim[0])
for k in range(N_sim):
  print(" =========================", k+1, " of", N_sim, " ===========================")
  # Fetch data
  u0_k = Us_sim[k]

  # Simulate one step
  xf_k, zf_k, y0_k = simulator.simstep(u0_k, dt_sim)
  Xs_sim += [ xf_k ]
  Zs_sim += [ zf_k ]
  Ys_sim += [ y0_k ]

  print("x_sim(k) =", Xs_sim[-2])
  print("u_sim(k) =", Us_sim[-1])
  print("y_sim(k) =", Ys_sim[-1])

  # Estimate one step
  xf_k_est, Pf_k_est = ekf(u0_k, y0_k, dt_sim)
  Xs_est += [ cas.vertcat(xf_k_est[:2], -2*(k+1)*dt_sim, xf_k_est[2:4], -2., xf_k_est[4]) ]

  print(Pf_k_est)

# 
xs_est = cas.horzcat(*Xs_est)
xs_sim = cas.horzcat(*Xs_sim)
us_sim = cas.horzcat(*Us_sim)

viz.plotStates_withRef(
  Xs = xs_est, Xs_ref = xs_sim,
  Us = us_sim, Us_ref = us_sim,
  dt = dt_sim, model = model
)

""" 
# Compute root-mean-square of the estimation error
err = [ Xs_sim[k].full() - Xs_est[k].full() for k in range(N_sim+1) ]
rms = [ cas.sqrt(cas.mtimes([err[k].T, err[k]])) for k in range(N_sim+1) ]

fig,ax = plt.subplots(1,1)
kAxis = range(N_sim+1)
plt.plot(kAxis, rms, 'x-')
plt.gca().set_yscale('log')
"""
plt.show()