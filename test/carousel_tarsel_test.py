import numpy as np
import casadi as cas
from thesis_code.models.carousel_whitebox import CarouselWhiteBoxModel
from thesis_code.components.carousel_tarsel import Carousel_TargetSelector
from thesis_code.carousel_simulator import Carousel_Simulator
import thesis_code.models.carousel_whitebox_viz as viz
import thesis_code.utils.signals as signals
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

# Load parameters
import json
param = {}
path = "../../carousel_identification/params_identified_dynamic_imu_new.json"
print("Loading identified parameter set " + path)
with open(path, 'r') as file:
    param = json.load(file)


# Create default model
model = CarouselWhiteBoxModel(param, with_angle_output=True, with_imu_output=True)
x0, z0, u0 = model.get_steady_state()
constants = model.getConstants()


# Time per revolution
T_per = 2*np.pi / abs(constants['carousel_speed'])

# Simulation parameters
M_sim = 5
T_sim = M_sim * T_per
N_sim = M_sim * 160
#dt_sim = T_sim / float(N_sim)
dt_sim = 1.0 / 20.0
print("dt_sim =", dt_sim)

# Target selector parameters
N_ts = 20
dt_ts = T_per / N_ts

# Create a target selector and create a reference trajectory
ref_period = 10 # seconds
ref_period_shift = ref_period/2.
ref_signal = signals.rectangle
ref_amplitude = 0.4 # rad
ref_offset = 0.0
ref_fun = lambda t: x0[0] + ref_amplitude * (ref_signal(t-ref_period_shift, ref_period) - 0.5) - ref_offset
tarsel = Carousel_TargetSelector(model = model, T = ref_period, N = N_ts, reference = ref_fun, verbose=True)
tarsel.init(x0 = x0, z0 = z0, W = 1e3 * cas.DM.eye(1))
Xref_orig, Zref_orig, Uref_orig = tarsel.call()
#x0,z0,u0 = tarsel.get_new_reference(0.0,0.0,0)

""" ================================================================================================================ """
# Create a few plots that illustrate the process
dt_ts = ref_period/N_ts
tAxis = np.arange(0, ref_period, dt_ts)
tAxis2 = np.arange(0, ref_period, dt_sim)
color_reference = 'magenta'
color_target_x = 'red'
color_target_u = 'blue'

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.ylim(0.1, 0.6)
plt.xlim(-0.5, 10.5)

with PdfPages("../../../tex/presentations/figures/target_selector_plots.pdf") as pdf:
  # Periodic reference and discretization
  plt.step(tAxis, [ref_fun(t) for t in tAxis], '-', color=color_reference, label='Reference')
  plt.ylabel(r'Elevation $\phi$')
  plt.xlabel(r'Time $t$')
  plt.legend(loc='lower right')
  pdf.savefig()

  for t in tAxis:
    plt.axvline(t, linewidth=1, linestyle='--', color='grey')
  plt.plot(tAxis, [ref_fun(t) for t in tAxis], 'x', color=color_reference, label='Sampled Reference')
  plt.legend(loc='lower right')
  pdf.savefig()

  # Target Selector sampled solution
  plt.plot(tAxis, Xref_orig[0,:].T, 'o', label="Sampled Solution", color=color_target_x)
  plt.legend(loc='lower right')
  pdf.savefig()

  # Target Selector interpolated solution
  x, z, u = tarsel.get_new_reference(tAxis2[0], dt_sim, len(tAxis2)-1)
  phiTAR = x[0,:].full().flatten()
  plt.plot(tAxis2, phiTAR, label="Interpolated Solution", color=color_target_x)
  plt.legend(loc='lower right')
  pdf.savefig()

plt.show()

quit(0)

""" ================================================================================================================ """

# Fetch reference trajectories (once whole, once piecewise)
# !! with piecewise, the fetching should be stable against perturbations (in both directions)
# !! solution: we 'rotate' the angles assigned to the bins by one half of delta-psi
# !! (= the yaw angle that is swept over in one time-step). That way we're safe as long
# !! as the yaw-disturbance is no bigger than one half of delta-psi
#x0[2] += 0.5* (M_sim * 2*np.pi / N_sim)
psi0 = np.mod(x0[2], 2*np.pi)
Xref_whole, Zref_whole, Uref_whole = tarsel.get_new_reference(psi0, dt_sim, N_sim)
Xref_piece = [ x0 ]
Zref_piece = [ z0 ]
Uref_piece = [ ]
for k in range(N_sim):
  delta = 1e-3 * dt_sim
  x0_k = Xref_piece[-1]
  psi0_k = np.mod(x0_k[2], 2*np.pi)
  X,Z,U = tarsel.get_new_reference(psi0_k, dt_sim, 1)
  #X,Z,U = tarsel.get_new_reference(x0_k[2], dt_sim-delta, 1)
  Xref_piece += [ X[:,1] ]
  Zref_piece += [ Z[:,1] ]
  Uref_piece += [ U[:,0] ]

Xref_piece = cas.horzcat(*Xref_piece)
Zref_piece = cas.horzcat(*Zref_piece)
Uref_piece = cas.horzcat(*Uref_piece)

# Check if the two methods produce the same result (Only check input-to-deflector behaviour)
if False:
  tAxis_orig = np.linspace(0, T_per, N_ts+1)
  tAxis = np.linspace(0, T_sim, N_sim+1)
  fig,ax = plt.subplots(2,1,sharex='col')
  plt.sca(ax[0])
  plt.suptitle("Comparison of interpolated values")
  plt.title("Input")
  plt.plot(tAxis_orig[:-1], Uref_orig[0,:].T, 'o', markersize=10, color="cyan", label="original data")
  plt.step(tAxis, Uref_whole[0,:], 'o-', label="whole", where='post')
  plt.step(tAxis[:-1], Uref_piece[0,:].T, 'x--', label="piece", where='post')
  plt.legend()
  plt.grid()
  plt.sca(ax[1])
  plt.title("Response")
  plt.plot(tAxis_orig[:-1], Xref_orig[-1,:].T, 'o', markersize=10, color="cyan", label="original data")
  plt.plot(tAxis, Xref_whole[-1,:], 'o-', label="whole")
  plt.plot(tAxis, Xref_piece[-1,:].T, 'x--', label="piece")
  plt.legend()
  plt.grid()
  plt.show()
  quit(0)

""" ================================================================================================================ """
"""
  Check if the reference trajectory (that is fetched one by one) can be reproduced by simulation
"""

# Prepare simulation
x0,z0,u0 = tarsel.get_new_reference(0.0,0.0,0)
Xs_sim = [ x0 ]
Zs_sim = [ z0 ]
Xs_ref = [ x0 ]
Zs_ref = [ z0 ]
Us_ref = [ ]

# Create a simulator
NX = model.NX()
NY = model.NY()
simulator = Carousel_Simulator(
  model = model, x0 = Xs_sim[0], z0 = Zs_sim[0],
  process_noise_mean = 0e0 * np.ones((NX)),
  process_noise_covar = 0e0 * np.eye((NX)),
  measurement_noise_mean = 0e0 * np.ones((NY)),
  measurement_noise_covar = 0e0 * np.eye((NY)),
  jit = False
)

t = -dt_sim
for k in range(N_sim):
  t += dt_sim
  print(" =========================", k+1, " of", N_sim, " ===========================")
  # Fetch data
  x0_k = Xs_sim[-1]
  z0_k = Zs_sim[-1]

  print("t = " + str(t))
  print("t/T_per = " + str(t/T_per))

  # Choose reference and new control
  Xref, Zref, Uref = tarsel.get_new_reference(t/T_per*2*np.pi, dt_sim, 1)
  xf_k_ref = Xref[:,1]
  zf_k_ref = Zref[:,1]
  u0_k = Uref[:,0]

  # Simulate one step
  xf_k, zf_k, y0_k = simulator.simstep(u0_k, dt_sim)

  # Store states
  Xs_sim += [ xf_k.full() ]
  Zs_sim += [ zf_k.full() ]
  Xs_ref += [ xf_k_ref ]
  Zs_ref += [ zf_k_ref ]
  Us_ref += [ u0_k ]

  #if k == 3: break


Xs_sim = cas.horzcat(*Xs_sim)
Xs_ref = cas.horzcat(*Xs_ref)
Us_ref = cas.horzcat(*Us_ref)

if False:
  print("RESULT ======================================")
  print("Us_ref =", Us_ref)

if False:
  # Check if the reference and response are the same
  tAxis_orig = np.linspace(0, T_per, N_ts, endpoint=False)
  tAxis = np.arange(0, dt_sim*Xs_sim.shape[1], dt_sim)
  fig,ax = plt.subplots(2,1,sharex='col')
  plt.sca(ax[0])
  plt.title("Input vs response")
  where = 'post' # 'post', 'mid', 'pre'
  plt.plot(tAxis_orig, Uref_orig[0,:].T, 'o', color="magenta", label="Input (orig)")
  plt.step(tAxis[:-1], Us_ref[0,:].T, 'x', color="green", label="Input (interp)", where=where)
  plt.plot(tAxis_orig, Xref_orig[-1,:].T, 'o', color="cyan", label="response: reference (orig)")
  plt.plot(tAxis, Xs_ref[-1,:].T, 'x-', color="blue", label="response: reference (interp)")
  plt.plot(tAxis, Xs_sim[-1,:].T, 'x-', color="red", label="response: simulation")
  plt.legend()
  plt.grid()
  plt.sca(ax[1])
  plt.title("Yaw angle")
  plt.plot(tAxis, Xs_sim[2,:].T, 'x-', color="red", label="Yaw")
  plt.legend()
  plt.grid()

if True:
  print("Plotting states vs references.")
  viz.plotStates_withRef(
    Xs = Xs_sim, Xs_ref = Xs_ref,
    Us = Us_ref, Us_ref = Us_ref,
    model = model, dt = dt_sim
  )

plt.show()
