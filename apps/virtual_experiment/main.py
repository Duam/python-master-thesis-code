import numpy as np
import casadi as cas
from thesis_code.carousel_model import CarouselWhiteBoxModel
import matplotlib.pyplot as plt
import thesis_code.carousel_visualizer as viz
from thesis_code.optimal_control_problems.carousel_mhe import Carousel_MHE
from thesis_code.optimal_control_problems.carousel_mpc import Carousel_MPC
from thesis_code.optimal_control_problems.carousel_tarsel import Carousel_TargetSelector
from thesis_code.carousel_simulator import Carousel_Simulator
from thesis_code.utils.bcolors import bcolors

""" =========================== Simulation ============================= """
# Create a simulation model
param_sim = CarouselWhiteBoxModel.getDefaultParams()
# TODO: Change some parameters
model_sim = CarouselWhiteBoxModel(param_sim)

# Set initial state and noise properties
x_ss_sim, z_ss_sim, u_ss_sim = model_sim.get_steady_state()
v_mean_sim  =  0e0 * np.ones(model_sim.NY())
v_covar_sim =  1e-3 * np.eye(model_sim.NY())
v_covar_sim[6,6] = 1e-4
v_covar_sim[7,7] = 1e-4
w_mean_sim  =  0e0 * np.ones(model_sim.NX())
w_covar_sim =  1e-6 * np.eye(model_sim.NX())
w_covar_sim[2,2] = 1e-12
w_covar_sim[5,5] = 1e-12

# Create a simulator
simulator = Carousel_Simulator(
  model_sim, x0 = x_ss_sim, z0 = z_ss_sim,
  process_noise_mean = w_mean_sim,
  process_noise_covar = w_covar_sim,
  measurement_noise_mean = v_mean_sim,
  measurement_noise_covar = v_covar_sim,
)

""" ============================== Meta ============================= """

# For M revolutions
M = 2
T_sim = M * 2*np.pi / abs(model_sim.params['carousel_speed'])

# Sample frequency is 50 (?) Hz
dt = 1.0 / 50.0
N_sim = int(np.ceil(T_sim/dt))
#N_sim = 20

# Estimation and control horizons
N_ts = 20
N_est = 6
N_ctr = 6

# Flags
verbose = False
jit = False
expand = not jit

""" ===================== Target selection, Estimation and Control ========================== """
# Create a control/estimation model
param = CarouselWhiteBoxModel.getDefaultParams()
model = CarouselWhiteBoxModel(param)
x_ss_est, z_ss_est, u_ss_est = model.get_steady_state()

NX = model.NX()
NZ = model.NZ()
NU = model.NU()
NY = model.NY()
NP = model.NP()

# Create a target-selector
#roll_ref = lambda psi: x0_est[0]
roll_ref = lambda psi: x_ss_est[0] + 15 * (2*np.pi/360) * np.sin(psi)
#roll_ref = lambda psi: x_ss_est[0] + 15 * (2*np.pi/360) * rectangle(psi/(2*np.pi)*N_ts,N_ts)
target_selector = Carousel_TargetSelector(model, N_ts, roll_ref, verbose=True)
target_selector.init(x0 = x_ss_est, z0 = z_ss_est, W = 1e0 * np.eye(1))
target_selector.call()
Xref, Zref, Uref = target_selector.get_new_reference(x_ss_est[2], dt, N_ctr)

# Create an estimator
Q_mhe = 1e0 * np.ones((NX,1))
Q_mhe[0] = 1e6 # Model confidence: Roll
Q_mhe[1] = 1e6 # Model confidence: Pitch
Q_mhe[2] = 1e6 # Model confidence: Yaw
Q_mhe[3] = 1e6 # Model confidence: Roll rate
Q_mhe[4] = 1e6 # Model confidence: Pitch rate
Q_mhe[5] = 1e6 # Model confidence: Yaw rate
Q_mhe[6] = 1e6 # Model confidence: Trim tab angle
R_mhe = 1e0 * np.ones((NY,1))
R_mhe[0] = R_mhe[1] = R_mhe[2] = 1e3 # Sensor confidence: Accelerometer
R_mhe[3] = R_mhe[4] = R_mhe[5] = 1e3 # Sensor confidence: Gyroscope
R_mhe[6] = 1e4 # Sensor confidence: Carousel encoder position
R_mhe[7] = 1e4 # Sensor confidence: Carousel speed smoothed
S_mhe = Q_mhe
estimator = Carousel_MHE(model, N_est, dt, verbose=verbose, do_compile=jit, expand=expand)
estimator.init(x0_est = x_ss_est, z0_est = z_ss_est, Q = Q_mhe, R = R_mhe, S = S_mhe)
#estimator = Carousel_MHE(model, N_est, dt, verbose=verbose, jit=jit)
#estimator.init(x0_est = x_ss_est, z0_est = z_ss_est, Q = cas.diag(Q_mhe), R = cas.diag(R_mhe), S = cas.diag(S_mhe))
print(estimator.ocp)

# Create a controller
Q_mpc = 1e-3 * np.ones((NX,1))
Q_mpc[0] = 1e0 # Roll, don't deviate
Q_mpc[1] = 1e0 # Pitch, don't deviate
Q_mpc[2] = 1e-3 # Yaw regularization
Q_mpc[3] = 1e0 # Roll rate, please follow reference
Q_mpc[4] = 1e0 # Pitch rate, please follow reference
Q_mpc[5] = 1e-3 # Yaw rate regularization (constrained)
Q_mpc[6] = 1e-2 # Trim tab angle, it's OK to deviate a bit
R_mpc = 1e-3 * np.ones((NU,1))
R_mpc[0] = 1e-1 # Trim tab setpoint, it's OK to deviate a bit
R_mpc[1] = 1e-3 # Yaw rate setpoint regularization
S_mpc = Q_mpc
S_mpc[0] = 1e1
S_mpc[1] = 1e1
S_mpc[3] = 1e-1
S_mpc[4] = 1e-1
S_mpc[6] = 1e0
controller = Carousel_MPC(model, N_ctr, dt, verbose=verbose, do_compile=jit, expand=expand)
controller.init(x0 = x_ss_est, Q = Q_mpc, R = R_mpc, S = S_mpc, Uref = Uref[:,:-1])
#controller = Carousel_MPC(model, N_ctr, dt, verbose=verbose, jit=jit)
#controller.init(x0 = x_ss_est, Q = cas.diag(Q_mpc), R = cas.diag(R_mpc), S = cas.diag(S_mpc), Uref = Uref[:,:-1])
print(controller.ocp)

""" ========================== Visualization ========================== """
#visualizer = Carousel_Visualizer(model_sim)
"""
import matplotlib.pyplot as plt
plt.axis([0, 10, 0, 1])
for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)
plt.show()
quit(0)
"""

""" ========================== Online: Estimation and Control ========================== """
# Initial target point
x0_ref, z0_ref, u0_ref = target_selector.get_new_reference(x_ss_est[2], 0.0, 0)
Xs_ref = [ x0_ref ]
Zs_ref = [ z0_ref ]
Us_ref = [ ] 
# Initial simulation state
Xs_sim = [ x_ss_sim ]
Zs_sim = [ z_ss_sim ]
Us_sim = [ ]
Ys_sim = [ ]
# Initial estimate
Xs_est = [ x_ss_est ]
Zs_est = [ z_ss_est ]

# Start simulation
mhe_solve_time = []
mhe_iter_count = []
mpc_solve_time = []
mpc_iter_count = []
print("========================= START ===========================")
for k in range(N_sim):
  print(" =========================", k+1, " of", N_sim, " ===========================")

  # Fetch the current state estimate
  x0_k_est = Xs_est[-1]
  z0_k_est = Zs_est[-1]

  # Chose the next reference points
  Xref, Zref, Uref = target_selector.get_new_reference(x0_k_est[2], dt, N_ctr)
  Xs_ref += [ Xref[:,1] ]
  Zs_ref += [ Zref[:,1] ]
  Us_ref += [ Uref[:,0] ]
  print("\n===============\n")
  print("Chose new reference.")
  print("\n===============")

  # Compute the control, based on the state estimate
  u0_k, ctrl_result, ctrl_stats, ctrl_w0, ctrl_p0 = controller.call(x0_k_est, Xref, Uref[:,:-1])
  #u0_k, ctrl_result, ctrl_stats = controller.call(Xs_sim[-1], Xref, Uref[:,:-1])
  Us_sim += [ u0_k ]
  ctrl_return_status = ctrl_stats['return_status']
  mpc_solve_time += [ ctrl_stats['t_proc_'+controller.ocp.name] ]
  mpc_iter_count += [ ctrl_stats['iter_count'] ]
  if ctrl_return_status != 'Solve_Succeeded': print(bcolors.WARNING)
  else: print(bcolors.OKGREEN)
  print("Computed new control.")
  print("Return status:", ctrl_stats['return_status'])
  print("Solve() time:", ctrl_stats['t_proc_'+controller.ocp.name]*1e3, "ms")
  print("Iterations:", ctrl_stats['iter_count'])
  print("Final objective:", ctrl_stats['iterations']['obj'][-1])
  print("Final constraint violation:", ctrl_stats['iterations']['inf_pr'][-1])
  print("lam_x_init:", ctrl_result['lam_g']['init'])
  print(bcolors.ENDC)
  print("===============\n")

  # Apply the control and simulate
  xf_k, zf_k, y0_k = simulator.simstep(u0_k[0], dt)
  Xs_sim += [ xf_k ]
  Zs_sim += [ zf_k ]
  Ys_sim += [ y0_k ]
  print("Simulated step.")

  print("\n===============")

  # Estimate the next state
  xf_k_est, zf_k_est, est_result, est_stats = estimator.call(u0_k, y0_k)
  Xs_est += [ xf_k_est ]
  Zs_est += [ zf_k_est ]
  est_return_status = est_stats['return_status']
  mhe_solve_time += [ est_stats['t_proc_'+estimator.ocp.name] ]
  mhe_iter_count += [ est_stats['iter_count'] ]
  if est_return_status != 'Solve_Succeeded': print(bcolors.WARNING)
  else: print(bcolors.OKBLUE)
  print("Estimated new state.")
  print("Return status:", est_stats['return_status'])
  print("Solve() time:", est_stats['t_proc_'+estimator.ocp.name]*1e3, "ms")
  print("Iterations:", est_stats['iter_count'])
  print("Final objective:", est_stats['iterations']['obj'][-1])
  print("Final constraint violation:", est_stats['iterations']['inf_pr'][-1])
  print(bcolors.ENDC)

  ctrl_inf = ctrl_return_status == 'Infeasible_Problem_Detected'
  est_inf = est_return_status == 'Infeasible_Problem_Detected'
  if ctrl_inf or est_inf:
    print(bcolors.FAIL + "ABORT. INFEASIBLE PROBLEM DETECTED."+ bcolors.ENDC)
    if ctrl_inf:
      print("Controller information")
      g0 = controller.ocp.eval_g(ctrl_w0,ctrl_p0)
      h0 = controller.ocp.eval_h(ctrl_w0,ctrl_p0)
      w = ctrl_result['w']
      p = ctrl_result['p']
      lam_g = ctrl_result['lam_g']
      lam_h = ctrl_result['lam_h']
      for key in lam_g.keys(): print(lam_g[key])
      print("Equality constraints:")
      for key in g0.keys(): print(key + " = " + str(g0[key]))
      print("Inequality constraints:")
      for key in h0.keys(): print(key + " = " + str(h0[key]))

    print(bcolors.ENDC)
    break

print("========================= DONE ===========================")
print("Average MPC solve() time:", sum(mpc_solve_time) / float(N_sim))
print("Average MPC iter count: ", sum(mpc_iter_count) / float(N_sim))
print("Average MHE solve() time:", sum(mhe_solve_time) / float(N_sim))
print("Average MHE iter count: ", sum(mhe_iter_count) / float(N_sim))

print("Simulation parameters:")
print("v_mean_sim = " + str(v_mean_sim))
print("v_covar_sim = \n" + str(v_covar_sim))
print("w_mean_sim = " + str(w_mean_sim))
print("w_covar_sim = \n" + str(w_covar_sim))
print("MHE parameters:")
print("Q_mhe = \n" + str(Q_mhe))
print("R_mhe = \n" + str(R_mhe))
print("S_mhe = \n" + str(S_mhe))
print("MPC parameters:")
print("Q_mpc = \n" + str(Q_mpc))
print("R_mpc = \n" + str(R_mpc))
print("S_mpc = \n" + str(S_mpc))


print("Pack trajectory data..")
Xs_ref = cas.horzcat(*Xs_ref)
Xs_est = cas.horzcat(*Xs_est)
Xs_sim = cas.horzcat(*Xs_sim)
Zs_ref = cas.horzcat(*Zs_ref)
Zs_est = cas.horzcat(*Zs_est)
Zs_sim = cas.horzcat(*Zs_sim)
Us_ref = cas.horzcat(*Us_ref)
Us_sim = cas.horzcat(*Us_sim)
Ys_sim = cas.horzcat(*Ys_sim)

print("Plot trajectory data..")
viz.plotStates(
  model = model_sim, dt = dt,
  Xs_ref = Xs_ref, Us_ref = Us_ref,
  Xs_sim = Xs_sim, Us_sim = Us_sim,
  Xs_est = Xs_est
)
plt.show()