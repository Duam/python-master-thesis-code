#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/04/03
  Brief: This file tests the nonlinear moving horizon estimator (NMHE)
          for iterative learning of periodic disturbance parameters. In this
          version, the periodic parameters are represented in the time-domain.
          It is tested against a simple 1 dimensional discrete PT1 element.
          This is a case study to test the influence of W1, .., W4 on the 
          estimation quality.
          x(k+1) = 0.9 x(k) + u(k)
          y(k)   = x(k) + p(k) + u(k)
"""

# For the system definition
from casadi import Function, SX
# The for MHE
from thesis_code.estimators.IL_NMHE import IL_NMHE
# For math
import numpy
numpy.set_printoptions(linewidth=numpy.inf)
# For plotting
import matplotlib.pyplot as plt
# For colorful text
from thesis_code.utils.bcolors import bcolors

# For signal generation
from thesis_code.utils.signals import rectangle, from_fourier

########################################################################################################################
########################################################################################################################
#                                                       SETUP

# Define system variables
nx = nu = ny = np = 1
x = SX.sym('x', nu, 1)
p = SX.sym('p', np, 1)
u = SX.sym('u', nu, 1)
y = SX.sym('y', ny, 1)

# Define the system and function handles
xnext = 0.6 * x + u
y = x + u + p
F = Function('F', [x,u,p], [xnext], ['x','u','p'], ['xnext'])
g = Function('g', [x,u,p], [y], ['x','u','p'], ['y'])

# System meta-parameters
T = 3                 # Period length of system
N_per = 40             # Samples per period. Should be an even number for the input signal in this system.
dt = float(T)/N_per   # Sampling time
N_mhe = 3             # MHE horizon

#########################################################################
# Fixed parameters for the whole of the study
x0_sim = numpy.zeros((nx,1))
x0_est = 1e-3 * numpy.ones((nx,1))
P0_est = 1e-3 * numpy.ones((np,N_per))
R  = 1e0 * numpy.eye(ny)

# Case study trial numbers
M_warmup = 2            # Fill the memory with valid data for M_warmup trials (No disturbances)
M_settle = 3            # Do M_settle trials to account for the MHE's transient oscillation (with disturbances)
M_main_convergence = 2  # Do M_main_convergence trials to gather data for the convergence rate
M_main_error = 10       # Do M_main_error trials to gather data for the estimation error
M_total = M_warmup + M_settle + M_main_convergence + M_main_error

# Case study samples
N_warmup = N_per * M_warmup
N_settle = N_per * M_settle
N_main_convergence = N_per * M_main_convergence
N_main_error = N_per * M_main_error

# Number of samples with active disturbance
N = N_settle + N_main_convergence + N_main_error 
# Total number of samples
N_total= N_warmup + N

# Compute indices where a new trial begins
Ntrials = numpy.ceil(float(N)/float(N_per))
trial_indices = list()
for k in range(N_total+1):
  if numpy.mod(k,N_per) == 0:
    trial_indices.append(k)

# Create control values
U_sim = [0 for k in range(N_warmup)] + [ rectangle(k,N_per,1.0) for k in range(N) ]

# Choose true parameter
param_offs = 0.0
param_amps = [1.0, 0.1]
param_freqs = lambda alpha: [1/float(T), alpha/float(T)]
param = lambda t, alpha: from_fourier(t, param_offs, param_amps, param_freqs(alpha))

# Plot parameter trajectory for testing
if False:
  alpha_test = 3.3
  tAxis = numpy.linspace(0, N_settle*dt, N_settle+1)
  param_test = [param(t, alpha_test) for t in tAxis]
  plt.figure("TEST PARAMETER PLOT")
  
  plt.ylabel(r'Parameter $p$')
  plt.xlim(left=-1, right=tAxis[-1]+1)
  plt.axhline(color='grey', linestyle='--')
  [plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
  plt.plot(tAxis, param_test)
  plt.show()
  quit(0)

#                                                   SETUP COMPLETE
########################################################################################################################
########################################################################################################################

STUDY_STRINGS = [
  "W1: Penalizes deviation from prior",
  "W2: Penalizes deviation from past trial's estimate",
  "W3: Penalizes 1st derivative",
  "W4: Penalizes 2nd derivative"
]
NUM_STUDIES = len(STUDY_STRINGS)

alphas = [0.0] + [ numpy.around(k*10.0**i,abs(i)) for i in numpy.arange(-1,5) for k in numpy.arange(1,10) ]
NUM_CASES = len(alphas)

W_zero = numpy.zeros((np,np))
Ws = [0.0] + [ numpy.around(k*10.0**i,abs(i)) for i in numpy.arange(-3,1) for k in numpy.arange(1,10) ]
NUM_RUNS = len(Ws)

# Create data containers
controls             = numpy.array(U_sim).reshape(N_total,nu)
states_simulated     = numpy.zeros((NUM_CASES, N_total+1, nx))
outputs_simulated    = numpy.zeros((NUM_CASES, N_total, ny))
parameters_simulated = numpy.zeros((NUM_CASES, N_total, np))
states_estimated     = numpy.zeros((NUM_STUDIES, NUM_CASES, NUM_RUNS, N_total+1, nx))
parameters_estimated = numpy.zeros((NUM_STUDIES, NUM_CASES, NUM_RUNS, N_total, np))
solver_iters         = numpy.zeros((NUM_STUDIES, NUM_CASES, NUM_RUNS, N_total))

skip = False
if not skip:
  print(bcolors.HEADER + "Simulating." + bcolors.ENDC)

  # Simulate the system for all alphas
  for CASE_NUM in range(NUM_CASES):
    # Fetch the alpha. Disturbs parameters nonperiodically.
    alpha = alphas[CASE_NUM]

    # Compute disturbed parameters
    parameters_warmup_tmp = numpy.zeros((N_warmup,np))
    parameters_active_tmp = numpy.array([param(dt*k,alpha) for k in range(N)]).reshape((N,np))
    parameters_simulated[CASE_NUM,:,:]  = numpy.concatenate([parameters_warmup_tmp, parameters_active_tmp])

    # Set the initial state
    states_simulated[CASE_NUM,0,:] = x0_sim.flatten()

    # Simulate the system
    for k in range(N_total):
      # Fetch data
      xk = states_simulated[CASE_NUM,k,:]
      uk = controls[k,:]
      pk = parameters_simulated[CASE_NUM,k,:]

      # Simulate one step
      states_simulated[CASE_NUM,k+1,:] = F(xk,uk,pk)
      outputs_simulated[CASE_NUM,k,:]  = g(xk,uk,pk)

  print(bcolors.HEADER + "Simulation done. Starting case study now." + bcolors.ENDC)

  # Start the case study
  for STUDY_NUM in range(NUM_STUDIES):
    print(bcolors.WARNING + "Starting study " + str(STUDY_NUM+1) + ". " + STUDY_STRINGS[STUDY_NUM] + bcolors.ENDC)

    # Iterate over all alphas
    for CASE_NUM in range(NUM_CASES):
      print(bcolors.HEADER + "CASE_NUM = " + str(CASE_NUM) + ", alpha = " + str(alphas[CASE_NUM]) + bcolors.ENDC)

      # Iterate over all weight magnitudes
      for RUN_NUM in range(NUM_RUNS):

        W_test = Ws[RUN_NUM] * numpy.eye(np)
        print(bcolors.OKBLUE + "RUN_NUM = " + str(RUN_NUM) + ", W = " + str(W_test) + bcolors.ENDC)

        # Create the MHE
        mhe = IL_NMHE(F, g, dt, N_per, N_mhe, nx, nu, ny, np, verbose=False)
      
        # Initialize the MHE according to which case we're testing
        if    STUDY_NUM == 0: mhe.init(x0_est, P0_est, R, W_test, W_zero, W_zero, W_zero)
        elif  STUDY_NUM == 1: mhe.init(x0_est, P0_est, R, W_zero, W_test, W_zero, W_zero)
        elif  STUDY_NUM == 2: mhe.init(x0_est, P0_est, R, W_zero, W_zero, W_test, W_zero)
        elif  STUDY_NUM == 3: mhe.init(x0_est, P0_est, R, W_zero, W_zero, W_zero, W_test)
        else: exit(1)

        # Define initial state
        states_estimated[STUDY_NUM, CASE_NUM, RUN_NUM, 0, :] = x0_est.flatten()

        # estimate    
        for k in range(N_total):
          # Fetch current variables
          uk_sim = controls[k,:].reshape((nu,1))
          yk_sim = outputs_simulated[CASE_NUM,k,:].reshape((ny,1))

          # Call the mhe
          xnext_est, P_traj_est, stats = mhe.call(k=k, u=uk_sim, y=yk_sim)
          
          # Store estimated variables
          states_estimated[STUDY_NUM, CASE_NUM, RUN_NUM, k+1, :] = xnext_est
          parameters_estimated[STUDY_NUM, CASE_NUM, RUN_NUM, k, :] = P_traj_est[:,-1] # The kth p is at the end of the trajectory

          # Store some solver stats
          solver_iters[STUDY_NUM, CASE_NUM, RUN_NUM, k] = stats['iter_count']

  print(bcolors.HEADER + "Case study done. Writing data now." + bcolors.ENDC)

#                                                    DATA GATHERING COMPLETE
########################################################################################################################
########################################################################################################################

# Create dataset
import xarray as xr
dataset = xr.Dataset(
  data_vars = {
    'controls': (['sample','control'], controls),
    'init_state_sim': (['state'], x0_sim.flatten()),
    'states_sim': (['case','sample','state'], states_simulated[:,1:,:]),
    'outputs_sim': (['case','sample','output'], outputs_simulated),
    'parameters_sim': (['case','sample','parameter'], parameters_simulated),
    'alphas': (['case'], alphas),
    'R': (['Rrows','Rcols'], R),
    'weight_W': (['run'], Ws),
    'init_state_est': (['state'], x0_est.flatten()),
    'states_est': (['study','case','run','sample','state'], states_estimated[:,:,:,1:,:]),
    'init_parameter_est': (['trial_sample','parameter'], P0_est.transpose()),
    'parameters_est': (['study','case','run','sample','parameter'], parameters_estimated),
    'iter_count': (['study','case','run','sample'], solver_iters),
  },
  coords = {
    'study': range(NUM_STUDIES),
    'case': range(NUM_CASES),
    'run': range(NUM_RUNS),
    'sample': range(N_total),
    'trial_sample': range(N_per),
    'control': range(nu),
    'state': range(nx),
    'parameter': range(np)
  },
  attrs = {
    'xnext': str(xnext),
    'y': str(y),
    'T': T,
    'dt': dt,
    'N_per': N_per,
    'M_warmup': M_warmup,
    'M_settle': M_settle,
    'M_conv': M_main_convergence,
    'M_err': M_main_error,
    'N_mhe': N_mhe
  }
)

print(dataset)
filename = 'case_study_data_loglinW_raw.nc'
dataset.to_netcdf(filename)
print("Data written to " + filename)
print("Exiting.")
exit(0)

#controls_xr = xr.DataArray(controls, coords = [('sample', N_total), ('control', nu)])
#states_simulated_xr = xr.DataArray(states_simulated, coords=[()])



if False:
  for STUDY_NUM in range(NUM_STUDIES):
    for CASE_NUM in range(NUM_CASES):
      
      alpha = alphas[CASE_NUM]
      P_sim = parameters_simulated[CASE_NUM,:,:]

      for RUN_NUM in range(NUM_RUNS):
        tAxis = numpy.linspace(0, N_total*dt, N_total+1)
        plt.figure("Trajectories")

        U_sim = controls
        X_sim = states_simulated[CASE_NUM,:,:]
        Y_sim = outputs_simulated[CASE_NUM,:,:]

        plt.subplot(211)
        plt.ylabel(r'State $x$, Control $u$, Output $y$')
        plt.xlim(left=-1, right=N_total*dt+1)
        plt.axhline(color='grey', linestyle='--')
        [plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
        plt.plot(tAxis[:N_total], U_sim[:N_total], 'x--', color='red', label='Control', alpha=0.25)
        plt.plot(tAxis, X_sim.flatten(), 'x-', color='green', label='State', alpha=0.25)
        plt.plot(tAxis[:N_total], Y_sim.flatten(), 'x-', color='blue', label='Output')
        plt.legend(loc='best')

        P_est = parameters_estimated[STUDY_NUM,CASE_NUM,RUN_NUM,:,:]

        plt.subplot(212)
        plt.title(r"Study " + str(STUDY_NUM+1) + " (" + STUDY_STRINGS[STUDY_NUM] + r"), $\alpha = $" + str(alpha) + r"$, W = $" + str(Ws[RUN_NUM]))
        plt.ylabel(r'Parameter $p$')
        plt.xlabel(r'Time $t$')
        plt.xlim(left=-1, right=N_total*dt+1)
        plt.axhline(color='grey', linestyle='--')
        [plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
        plt.plot(tAxis[:N_total], P_sim[:N_total,:].flatten(), 'x-', color='blue', label='Simulated')
        plt.plot(tAxis[:N_total], P_est[:N_total,:].flatten(), 'x--', color='red', label='Estimated')
        plt.legend(loc='best')
        plt.show(block=True)


"""
max_convergence_rates = numpy.zeros((NUM_RUNS,))
max_estimation_errors = numpy.zeros((NUM_RUNS,))
for RUN_NUM in range(NUM_RUNS):
  
  # Convergence rate
  n_start = N_warmup + N_settle
  n_end   = n_start + N_main_convergence
  max_convergence_rates[RUN_NUM] = max([ err_p[k]/err_p[k-1] for k in range(n_start,n_end) ])

  # Estimation error
  n_start = N_warmup + N_settle + N_main_convergence
  max_estimation_errors[RUN_NUM] = max(err_p[n_start:])

  print(bcolors.OKGREEN + "==============================")
  print("RUN_NUM = " + str(RUN_NUM))
  print("W_"+ str(CASE+1) + " = " + str(W_test))
  print("max_convergence_rate = " + str(max_convergence_rates[RUN_NUM]))
  print("max_estimation_error = " + str(max_estimation_errors[RUN_NUM]))
  print("=============================" + bcolors.ENDC)

"""

#                                                    DATA ANALYSIS COMPLETE
########################################################################################################################
########################################################################################################################

"""
# Plot parameter trajectory
if True:
  tAxis = numpy.linspace(0, N_total*dt, N_total+1)
  plt.figure("True trajectories")
  
  plt.subplot(211)
  plt.ylabel(r'State $x$, Control $u$, Output $y$')
  plt.xlim(left=-1, right=N_total*dt+1)
  plt.axhline(color='grey', linestyle='--')
  [plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
  plt.plot(tAxis[:N_total], U_sim[:N_total], 'x--', color='red', label='Control', alpha=0.25)
  plt.plot(tAxis, X_sim.flatten(), 'x-', color='green', label='State', alpha=0.25)
  plt.plot(tAxis[:N_total], Y_sim.flatten(), 'x-', color='blue', label='Output')
  plt.legend(loc='best')
  
  plt.subplot(212)
  plt.title(r'$\alpha = $' + str(alpha))
  plt.ylabel(r'Parameter $p$')
  plt.xlabel(r'Time $t$')
  plt.xlim(left=-1, right=N_total*dt+1)
  plt.axhline(color='grey', linestyle='--')
  [plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
  plt.plot(tAxis[:N_total], P_inner[:N_total], 'x--', color='red', label='Inner sine', alpha=0.25)
  plt.plot(tAxis[:N_total], P_outer[:N_total], 'x--', color='green', label='Outer sine', alpha=0.25)
  plt.plot(tAxis[:N_total], P_true[:N_total], 'x-', color='blue', label='Combined sine')
  plt.legend(loc='best')
  plt.show(block=False)

if True:
  # Prepare plotting
  plt.figure("Case Study")
  nAxis = Ws

  plt.subplot(211)
  plt.title(r"$\alpha = $" + str(alpha))
  plt.gca().set_yscale('log')
  plt.ylabel(r'error')
  plt.xlim(left=-0.25, right=1.25)
  plt.plot(nAxis, max_estimation_errors, 'x-', color='blue')
  plt.legend(loc='best')

  plt.subplot(212)
  plt.gca().set_yscale('log')
  plt.ylabel(r'convergence rate')
  plt.xlim(left=-0.25, right=1.25)
  #[plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
  plt.plot(nAxis, max_convergence_rates, 'x-', color='blue')
  plt.legend(loc='best')
  plt.xlabel(r'W_' + str(CASE+1))
  plt.show()
"""