import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from thesis_code.models.overhead_crane.overhead_crane import OverheadCrane
from thesis_code.integrators.rk4step import rk4step
from thesis_code.referenceGenerators.RefGenNLP import RefGenNLP
from thesis_code.controllers.ilc_simple import ILC

print("Overhead-Crane Test: Target selector + Linear ILC")

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
M = 2             # Number of trials
T = 200           # Time per trial
N = 50            # Samples per trial
N_ilc = N -1      # ILC samples (one less than N b.c. repetitivity)
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
#                         INITIALIZE CONTROLLER                         #

print("Initializing controller ..")


# Cut off last element of yref (duplicate of first element)
yref = yref[:,:N-1]

print(xref.shape)
print(uref.shape)
print(yref.shape)

# Create linear ILC
ilc = ILC(F, out, N_ilc, nx, nu, ny)
ilc.init(yref, xref, uref)

#########################################################################
#                          CLOSED-LOOP CONTROL                          #

print("Simulating closed loop ..")

# Initialize x,u,y containers
usim = np.zeros((nu,N-1,M))
xsim = np.zeros((nx,N-1,M))
xcmp = np.zeros((nx,N-1,M))
ysim = np.zeros((ny,N-1,M))
ycmp = np.zeros((ny,N-1,M))
xsim[:,0,0] = xref[:,0].flatten()
xcmp[:,0,0] = xref[:,0].flatten()

# Initialize containers for ilc control
dus = np.zeros((nu,N-1,M))

gotcha = False
# Control loop for M trials with N samples each
for i in range(M):
  for k in range(N-1):

    # Fetch control target and states
    uref_k = uref[:,k]
    xsim_k = xsim[:,k,i]
    xcmp_k = xcmp[:,k,i]

    # Compute current (pseudo) output
    ysim_k = out(xsim_k, uref_k).full().flatten()

    # Call LILC
    du_k = ilc.call(ysim_k, xsim_k)
    
    # Compute new control
    usim_k = uref_k + du_k

    # Create some noise
    mu_w = 0.0 * np.ones((nx,))
    Q_w = 0.0 * np.ones((nx,nx))
    wk = np.random.multivariate_normal(mu_w, Q_w)

    # Apply control to the plant
    xnext     = F(xsim_k, usim_k).full().flatten() + wk
    xnext_cmp = F(xcmp_k, uref_k).full().flatten() + wk

    # Store x. If it overflows into the next trial, store it there
    if k+1 < N-1:
      xsim[:,k+1,i] = xnext
      xcmp[:,k+1,i] = xnext_cmp
    else:
      # Only store it if there are trials left
      if i+1 < M:
        xsim[:,0,i+1] = xnext
        xcmp[:,0,i+1] = xnext_cmp

    # Store u,y
    dus[:,k,i] = du_k
    usim[:,k,i] = usim_k
    ysim[:,k,i] = out(xsim_k, usim_k).full().flatten()
    ycmp[:,k,i] = out(xcmp_k, uref_k).full().flatten()

    #print('i='+str(i) + ', k='+str(k) + ': ||xdif|| = '+str(np.linalg.norm(xsim_k - xcmp_k)))


#########################################################################
#                          PLOTTING & ANALYSIS                          #

print("Plotting results ..")

# Prepare output deviation plot
rms_fig = plt.figure('Output reference tracking performance')
rms_ax = plt.gca()
rms_nAxis = np.repeat(np.array(range(N-1)).reshape((N-1,1)), M, 1)

# Compute RMS
yref_repeated = np.repeat(yref.reshape((ny,N-1,1)), M, 2)
rms_cmp = (np.sqrt(np.square(yref_repeated - ycmp))).mean(axis=0)
rms     = (np.sqrt(np.square(yref_repeated - ysim))).mean(axis=0)

# Plot RMS
plt.xlabel('Sample')
plt.ylabel('RMS error')
plt.gca().set_yscale('log')
plt.plot(rms_nAxis, rms_cmp, '-', color='r', label='uncontrolled')
plt.plot(rms_nAxis, rms, '--', color='b', label='LILC')
rms_legend = rms_ax.legend(loc='best')

# Place trial-indicators in plot
for i in range(M):
  plt.text(rms_nAxis[N-2,i], rms[N-2,i], 'i='+str(i))
  plt.text(rms_nAxis[N-2,i], rms_cmp[N-2,i], 'i='+str(i))

''' next figure '''

# Prepare control plot
du_fig = plt.figure('ILC control output (du)')
du_nAxis = np.repeat(np.array(range(N_ilc)).reshape((N_ilc,1)), M, 1)

# Plot control 0
plt.subplot(211)
du0_ax = plt.gca()
plt.ylabel('Control output')
for i in range(M):
  plt.plot(du_nAxis[:,i], dus[0,:,i], label='trial '+str(i))
du0_legend = du0_ax.legend(loc='best')

# Plot control 1
plt.subplot(212)
du1_ax = plt.gca()
plt.xlabel('Sample')
plt.ylabel('Control output')
for i in range(M):
  plt.plot(du_nAxis[:,i], dus[1,:,i], label='trial '+str(i))
du1_legend = du1_ax.legend(loc='best')


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
plt.plot(rms_nAxis, rms_controlled, '--', color='b', label='LILC')
rms_legend = rms_ax.legend(loc='best')

# Place trial-indicators in plot
for i in range(M):
  plt.text(rms_nAxis[N-2,i], rms_controlled[N-2,i], 'i='+str(i))
  plt.text(rms_nAxis[N-2,i], rms_uncontrolled[N-2,i], 'i='+str(i))


'''
plt.show()
