import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from thesis_code.models.overhead_crane import OverheadCrane
from thesis_code.rk4step import rk4step
from thesis_code.referenceGenerators.RefGenNLP import RefGenNLP
from thesis_code.controllers.LMPC_simple import LMPC

print("Overhead-Crane Test: Target selector + Linear MPC")

#########################################################################
#                                SETUP                                  #

print("Setting up test ..")

# System model
plant = OverheadCrane()

nx = 6
nu = 2
ny = 2

# System model in casadi
x = cas.MX.sym('x', nx, 1)
u = cas.MX.sym('u', nu, 1)
ode = cas.Function('ode', [x,u], [plant.ode(x,u)], ['x','u'], ['xdot'])
out = cas.Function('out', [x,u], [plant.output(x,u)], ['x','u'], ['y'])

# Simulation parameters
M = 5             # Number of trials
T = 200           # Time per trial
N = 50            # Samples per trial
N_mpc = 10        # MPC horizon
dt = T/float(N)   # Sample-time
nn = 10           # Subsamples per rk4-step

# Discretize system
xnext = rk4step(ode, x, u, dt, nn)
F = cas.Function('F', [x,u], [xnext], ['x','u'], ['xnext'])

# Create cyclic target selector
gen = RefGenNLP(F, out, N, nx, nu, ny)
gen.add_equality('cycle', gen.X[:,0] - gen.X[:,N-1])
Q_gen = 1e5 * cas.DM.eye(ny)
R_gen = 1e0 * cas.DM.eye(nu)
gen.init(Q_gen,R_gen)

# Create linear MPC
mpc = LMPC(F, N_mpc, nx, nu)
Q_mpc = 1e5 * cas.DM.eye(nx)
R_mpc = 1e0 * cas.DM.eye(nu)
mpc.init(Q_mpc,R_mpc)


#########################################################################
#                     COMPUTE REFERENCE AND TARGET                      #

print("Computing targets ..")

# Set the output reference (checkpoints)
ref = np.array([
  [ 0.0,  0.5, -0.5,  0.0],
  [-1.0, -2.0, -2.0, -1.0]
])
t_checkpoints = np.linspace(0, T, ref.shape[1])

# Compute the output reference trajectory
tAxis = np.linspace(0, T, N-1)
x_spline = CubicSpline(t_checkpoints, ref[0,:], bc_type='periodic')
y_spline = CubicSpline(t_checkpoints, ref[1,:], bc_type='periodic')
yref = np.array([x_spline(tAxis), y_spline(tAxis)])

# Create initial guess for the target selector
xinit = np.vstack([
  np.concatenate([yref[0,:], np.array([yref[0,0]])]), 
  np.zeros((1,N)),
  np.concatenate([yref[1,:], np.array([yref[1,0]])]),
  np.zeros((1,N)),
  np.zeros((2,N))
])

# Compute the state and control target for one cycle
xref, uref, gen_stats, gen_result = gen.run(yref, xinit)

# Remove the last state (duplicate of first one)
xref = xref[:,:N-1]

#########################################################################
#                          CLOSED-LOOP CONTROL                          #

print("Simulating closed loop ..")

# Initialize x,u,y containers
xsim = np.zeros((nx,N-1,M))
xsim[:,0,0] = xref[:,0].flatten()
usim = np.zeros((nu,N-1,M))
ysim = np.zeros((ny,N-1,M))

# Initialize x,y containers for the system without a controller
xsim_uncontrolled = np.zeros((nx,N-1,M))
xsim_uncontrolled[:,0,0] = xref[:,0].flatten()
ysim_uncontrolled = np.zeros((ny,N-1,M))

# Control loop for M trials with N samples each
for i in range(M):
  for k in range(N-1):

    # Current state
    xk = xsim[:,k,i]
    xk_uncontrolled = xsim_uncontrolled[:,k,i]

    # Call LMPC
    dx, du, stats, result = mpc.call(xk, xref[:,0:N_mpc], uref[:,0:N_mpc-1])

    #print(du[:,0])

    # Compute new control
    uk = uref[:,0] + du[:,0]      

    # Create some noise
    mu_w = 0.0 * np.ones((nx,))
    Q_w = 1e-12 * np.ones((nx,nx))
    wk = np.random.multivariate_normal(mu_w, Q_w)

    # Apply control to the plant
    xnext = F(xk, uk).full().flatten() + wk
    xnext_uncontrolled = F(xk_uncontrolled, uref[:,0]).full().flatten() + wk

    # Store x. If it overflows into the next trial, store it there
    if k+1 < N-1:
      xsim             [:,k+1,i] = xnext
      xsim_uncontrolled[:,k+1,i] = xnext_uncontrolled
    else:
      # Only store it if there are trials left
      if i+1 < M:
        xsim             [:,0,i+1] = xnext
        xsim_uncontrolled[:,0,i+1] = xnext_uncontrolled

    # Store u,y
    usim[:,k,i] = uk
    ysim[:,k,i] = out(xk,uk).full().flatten()
    ysim_uncontrolled[:,k,i] = out(xk_uncontrolled,uref[:,0]).full().flatten()

    # Left-rotate cyclic references
    xref = np.roll(xref, -1, 1)
    uref = np.roll(uref, -1, 1)

  # The last reference state is a duplicate of the first one, so roll it 
  # once more to get back to where we started.
  #xref = np.roll(xref, -1, 1)
  print(xref[:,0])

print(xref[:,0])
print(xsim[:,0,0])
#########################################################################
#                          PLOTTING & ANALYSIS                          #

print("Plotting results ..")

# Prepare output deviation plot
rms_fig = plt.figure('Output reference tracking performance')
rms_ax = plt.gca()
rms_nAxis = np.repeat(np.array(range(N-1)).reshape((N-1,1)), M, 1)

# Compute RMS
yref_repeated = np.repeat(yref.reshape((ny,N-1,1)), M, 2)
rms_uncontrolled = (np.sqrt(np.square(yref_repeated - ysim_uncontrolled))).mean(axis=0)
rms_controlled = (np.sqrt(np.square(yref_repeated - ysim))).mean(axis=0)

# Plot RMS
plt.xlabel('Sample')
plt.ylabel('RMS error')
plt.gca().set_yscale('log')
plt.plot(rms_nAxis, rms_uncontrolled, '-', color='r', label='uncontrolled')
plt.plot(rms_nAxis, rms_controlled, '--', color='b', label='LMPC')
rms_legend = rms_ax.legend(loc='best')

# Place trial-indicators in plot
for i in range(M):
  plt.text(rms_nAxis[N-2,i], rms_controlled[N-2,i], 'i='+str(i))
  plt.text(rms_nAxis[N-2,i], rms_uncontrolled[N-2,i], 'i='+str(i))

'''
# Prepare state deviation plot
rms_x_fig = plt.figure('State target tracking performance')
rms_x_ax = plt.gca()
rms_x_nAxis = np.repeat(np.array(range(N)).reshape((N,1)), M, 1)

# Compute RMS (state deviation)
xref_repeated = np.repeat(xref.reshape((ny,N-1,1)), M, 2)
rms_uncontrolled = (np.sqrt(np.square(yref_repeated - ysim_uncontrolled))).mean(axis=0)
rms_controlled = (np.sqrt(np.square(yref_repeated - ysim))).mean(axis=0)

# Plot RMS
plt.xlabel('Sample')
plt.ylabel('RMS error')
plt.gca().set_yscale('log')
plt.plot(rms_nAxis, rms_uncontrolled, '-', color='r', label='uncontrolled')
plt.plot(rms_nAxis, rms_controlled, '--', color='b', label='LMPC')
rms_legend = rms_ax.legend(loc='best')

# Place trial-indicators in plot
for i in range(M):
  plt.text(rms_nAxis[N-2,i], rms_controlled[N-2,i], 'i='+str(i))
  plt.text(rms_nAxis[N-2,i], rms_uncontrolled[N-2,i], 'i='+str(i))


'''
plt.show()