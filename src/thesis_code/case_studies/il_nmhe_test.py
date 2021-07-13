#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/03/15
  Brief: This file tests the nonlinear moving horizon estimator (NMHE)
          for iterative learning of periodic disturbance parameters. In this
          version, the periodic parameters are represented in the time-domain.
          It is tested against a simple 1 dimensional discrete integrator.
"""

from thesis_code.estimators.IL_NMHE import IL_NMHE
from casadi import Function, SX
import numpy
import matplotlib.pyplot as plt
from thesis_code.utils.bcolors import bcolors

# Define system (integrator)
nx = nu = ny = np = 1
x = SX.sym('x', nu, 1)
p = SX.sym('p', np, 1)
u = SX.sym('u', nu, 1)
y = SX.sym('y', ny, 1)

xnext = x + u
F = Function('F', [x,u,p], [xnext], ['x','u','p'], ['xnext'])

y = x + u + p
g = Function('g', [x,u,p], [y], ['x','u','p'], ['y'])

# System meta-parameters
T = 3                 # Period length of system
N_per = 5             # Samples per period
dt = float(T)/N_per   # Sampling time
N_mhe = 3             # MHE horizon

N_mhe_p = numpy.min([N_per, N_mhe])
P_shift = numpy.mod(N_mhe, N_mhe_p)
print(N_mhe_p)
print(P_shift)

M = 1
elim = numpy.zeros((M))
for i in range(M):

  # Create IL-NMHE
  mhe = IL_NMHE(F, g, dt, N_per, N_mhe, nx, nu, ny, np)
  print(mhe._Np)
  # Initial state, parameters and tuning matrices
  x0_est = 1e-1 * numpy.ones((nx,1))
  sqrtTest = numpy.sqrt(i/float(N_per))
  #P0_est = sqrtTest * numpy.ones((np,N_per))
  P0_est = 1e0 * numpy.ones((np,N_per))

  R = 1e0 * numpy.eye(ny)
  W1 = 0.0 * numpy.eye(np)
  W2 = 0e0 * numpy.eye(np)
  W3 = 1e-1 * numpy.eye(np)
  W4 = 0e0 * numpy.eye(np)
  print("P0_est = " + str(P0_est))
  # Initialize IL-NMHE
  mhe.init(x0_est, P0_est, R, W1, W2, W3, W4)

  # Prepare simulation for N steps
  N = 60
  x0 = numpy.zeros((nx,1))
  p0 = numpy.zeros((np,1))

  # Choose controls (triangular) for N steps
  U_sim = numpy.zeros((nu,N))
  up = True
  for k in range(N): 
    if numpy.mod(2*k,N_per) == 0: 
      up = not up
    U_sim[:,k] = numpy.ones((nu,1)) if up else (-1)*numpy.ones((nu,1))
    

  # Choose parameters (sinusoidal with frequency 1/T, shifted by pi/2) for N_per steps
  P_sim = numpy.zeros((np,N_per))
  for k in range(N_per):
    arg_sin = 2 * numpy.pi * k * dt / float(T) + numpy.pi / 2.0
    P_sim[:,k] = - numpy.ones((np,1)) * numpy.sin(arg_sin)
    

  print("P_sim = " + str(P_sim))

  # Create data containers for simulation
  X_sim = numpy.zeros((nx,N+1))
  X_sim[:,0] = x0.flatten()
  Y_sim = numpy.zeros((ny,N))
  P_rec = numpy.zeros((np,N))

  # Create data containers for estimation
  X_est = numpy.zeros((nx,N+1))
  X_est[:,0] = x0_est.flatten()
  P_est = numpy.zeros((np,N))

  # Simulate N steps
  for k in range(N):
    kmodN = numpy.mod(k,N_per)
    # Fetch current variables
    xk = X_sim[:,k]
    uk = U_sim[:,k].reshape((nu,1))
    pk = P_sim[:,kmodN]

    # Simulate one step
    xnext = F(xk, uk, pk).full()
    yk    = g(xk, uk, pk).full()

    #print('k = ' + str(k))

    # Call the mhe
    P_prio = mhe._P_full_mem
    xnext_est, P_traj_est, stats = mhe.call(k=k, u=uk, y=yk)
    P_post = mhe._P_full_mem

    # Print info
    print(bcolors.OKGREEN + "==================" + bcolors.ENDC)
    if k > 0: print('(Prio) k=' + str(k) + ', p=' + str(numpy.roll(P_prio,k-1)))
    else:     print('(Prio) k=' + str(k) + ', p=' + str(numpy.roll(P_prio,k)))
    print('(Sol ) k=' + str(k) + ', p=' + str(numpy.roll(P_post,k)))

    
    # Store variables
    X_sim[:,k+1] = xnext
    Y_sim[:,k] = yk
    P_rec[:,k] = pk
    X_est[:,k+1] = xnext_est
    P_est[:,k] = P_traj_est[:,-1] # The kth p is at the end of the trajectory

  elim[i] = P_traj_est[:,-1]

print(stats)


print(elim)
figelim = plt.figure("Error Limit vs. Initial p magnitude")
xAxis = range(M)
#xSqrtAxis = numpy.sqrt([x/float(N_per) for x in xAxis])
plt.plot(xAxis, numpy.square(elim), 'x-')
plt.title("N_per="+str(N_per) + ", N_mhe="+str(N_mhe) + r", $\tilde{p}_0 = \tilde{p}_1 = \dots = \tilde{p}_{N^p-1}$")
plt.xlabel(r'$|\tilde\mathbf{{p}}|_2^2$')
plt.ylabel(r'$\mathbf{e}^2$ after ' + str(N) + ' mhe calls')
plt.show()


# Prepare plotting
fig = plt.figure("IL-NMHE Test")
tAxis = numpy.linspace(0, N*dt, N+1)

# Compute indices where a new trial begins
Ntrials = numpy.ceil(float(N)/float(N_per))
trial_indices = list()
for k in range(N):
  if numpy.mod(k,N_per) == 0:
    trial_indices.append(k)

# Plot controls
plt.subplot(421)
plt.title("T="+str(T)+", N_per="+str(N_per) + ", N_mhe="+str(N_mhe) + ", dt="+str(dt))
plt.ylabel('u')
plt.xlim(left=-1, right=N*dt+1)
[plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
plt.plot(tAxis[:N], U_sim[0,:], 'x-')

# Plot states
plt.subplot(423)
plt.ylabel('x')
plt.xlim(left=-1, right=N*dt+1)
[plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
plt.plot(tAxis[:N+1], X_sim[0,:], 'x-')
plt.plot(tAxis[:N+1], X_est[0,:], 'x--')

# Plot measurements
plt.subplot(425)
plt.ylabel('y')
plt.xlim(left=-1, right=N*dt+1)
[plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
plt.plot(tAxis[:N], Y_sim[0,:], 'x-')

# Plot parameters
plt.subplot(427)
plt.ylabel('p')
plt.xlabel('t')
plt.xlim(left=-1, right=N*dt+1)
[plt.axvline(dt*k, color='g', linestyle='--') for k in trial_indices]
plt.plot(tAxis[:N], P_rec[0,:N], 'x-')
plt.plot(tAxis[:N], P_est[0,:N], 'x--')

# Plot state deviation each iteration
plt.subplot(424)
plt.gca().set_yscale('log')
kAxis = range(N_per)
k_end_modN = numpy.remainder(N-1, N_per)
for i in range(int(Ntrials)):
  k_start = int(trial_indices[i])
  k_end   = int(trial_indices[i+1]) if i+1 < int(Ntrials) else int(N-1)
  ks = kAxis if i+1 < Ntrials else kAxis[:k_end_modN]
  X_sim_i = X_sim[:,k_start:k_end].flatten()
  X_est_i = X_est[:,k_start:k_end].flatten()
  dx = numpy.sqrt(numpy.square(X_sim_i - X_est_i))
  alpha = 0.2 + i * (0.8 / (Ntrials-1))
  plt.plot(ks, dx, alpha=alpha, label='i='+str(i)) # TODO: ANNOTATE
  plt.legend().draggable()


plt.subplot(428)
plt.gca().set_yscale('log')
kAxis = range(N_per)
k_end_modN = numpy.remainder(N-1, N_per)
for i in range(int(Ntrials)):
  k_start = int(trial_indices[i])
  k_end   = int(trial_indices[i+1]) if i+1 < int(Ntrials) else int(N-1)
  ks = kAxis if i+1 < Ntrials else kAxis[:k_end_modN]
  p_rec_i = P_rec[:,k_start:k_end].flatten()
  p_est_i = P_est[:,k_start:k_end].flatten()
  dp = numpy.sqrt(numpy.square(p_rec_i - p_est_i))
  alpha = 0.2 + i * (0.8 / (Ntrials-1))
  plt.plot(ks, dp, alpha=alpha, label='i='+str(i))
  plt.legend().draggable()

plt.show()