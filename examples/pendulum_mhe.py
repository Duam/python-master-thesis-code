import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from thesis_code.models.pendulum import Pendulum
from thesis_code.estimators.MHE import MHE

## Create a pendulum
pendulum = Pendulum()
print(pendulum.toString())

## Create rk4step function
def rk4step (f, x, u, w, h):
  k1 = f(x,u,w)
  k2 = f(x + h/2.0 * k1, u, w)
  k3 = f(x + h/2.0 * k2, u, w)
  k4 = f(x + h * k3, u, w)

  return x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


################################################
###       DEFINE SIMULATION PARAMETERS       ###

# Simulation parameters
T = 10.0 # seconds
N = 100 # samples
dt = T/N # Sample time

# Initial state
x0 = np.array([-np.pi/2.0, 0])

# Process noise covariance
Q = cas.vertcat(
  cas.horzcat(0.0001, 0.000),
  cas.horzcat(0.000, 0.0001)
)

# Measurement noise covariance
R = cas.vertcat(
  cas.horzcat(0.001, 0.000),
  cas.horzcat(0.000, 0.001)
)

# Convenience functions
f = lambda x, u, w: pendulum.ode(x,u,w)
g = lambda x, u, w: pendulum.output(x,u,w)

print(f(x0,[0,0],[0,0]))

# Integrator
f_discr = lambda x, u, w: rk4step(f, x, u, w, dt)

print("Simulation parameters set.")

################################################
###          SIMULATE THE SYSTEM             ###

print("Simulating system..")

# Choose random controls and disturbances (k=0 to k=N-1)
Us = np.zeros((2,N))
Us[:,int(N/2):] = 3*np.ones((2,int(N/2)))

Ws = np.zeros((2,N))
for k in range(N):
  Ws[:,k] = np.random.multivariate_normal(np.zeros((2)), Q)

print("Us shape = ", Us.shape, " (inputs)")
print("Ws shape = ", Ws.shape, " (disturbances)")

# Start the simulation (k=0 to k=N)
Xs_sim = np.zeros((2,N+1))
Xs_sim[:,0] = x0
for k in range(N):
  Xs_sim[:,k+1] = f_discr(Xs_sim[:,k], Us[:,k], Ws[:,k]).full().flatten()

print("Xs shape = ", Xs_sim.shape, " (states)")

# Compute the measurements (k=0 to K=N-1) and add noise
Ys_sim = np.zeros((2,N))
for k in range(N):
  Ys_sim[:,k] = g(Xs_sim[:,k], Us[:,k], Ws[:,k])
  Ys_sim[:,k] += np.random.multivariate_normal(np.zeros((2)), R)


print("Ys shape = ", Ys_sim.shape, " (outputs)")

print("Simulation done.")

################################################
###     CREATE MOVING HORIZON ESTIMATOR      ###

# Set estimator parameters
x0hat = cas.vertcat([0.0, 0.0])
P0 = cas.vertcat(
  cas.horzcat(0.001, 0.000),
  cas.horzcat(0.000, 0.001)
)

# Create an estimator object
N_est = 10
dt_est = 0.1 # Needs to be the same as simulation sampling time
mhe = MHE(f_discr, g, N_est, dt_est, 2, 2, 2, 2)
mhe.init(x0hat, P0, Q, R)

# Print sparsity pattern
#sparsity_fig = mhe.printSparsityPattern(98)

################################################
###         ESTIMATE THE TRAJECTORY          ###

# Create container
Xs_est = np.zeros((2,N+1))

# Estimate the first N samples to test
#Xs_est[:,:N_est], Ws_est = mhe.estimateTrajectory(x0, P0, Us[:,:N_est], Ys_sim[:,:N_est])

#for k in range(N_est,N):
for k in range(N):
  #Xs_est[:,k], w_est, stats = mhe.call(Us[:,k], Ys_sim[:,k-1])
  pass

#print(mhe.nlp)

################################################
###         PLOT THE TRAJECTORIES            ###

# Prepare plotting
tAxis = np.linspace(0, T, N+1)
plt.figure(1)

start = 0
end = N

# Plot
plt.subplot(211)
plt.plot(tAxis[start:end], Us[0,start:end])
plt.ylabel('u')
plt.xlabel('t [s]')

plt.subplot(212)
plt.plot(tAxis[start:end], Xs_sim[0,start:end])
plt.plot(tAxis[start:end], Ys_sim[0,start:end], 'o')
#plt.plot(tAxis[:N_est], Xs_est[0,:].flatten())
plt.plot(tAxis[start:end], Xs_est[0,start:end].flatten(), linewidth=3)
plt.plot(tAxis[start:end], Xs_sim[1,start:end])
plt.plot(tAxis[start:end], Ys_sim[1,start:end], 'o')
plt.plot(tAxis[start:end], Xs_est[1,start:end].flatten())
#plt.plot(tAxis[:N_est], Xs_est[1,:].flatten())
plt.ylabel('x')
plt.xlabel('t [s]')
plt.legend([
  'x0 sim',
  'x0 meas',
  'x0 est',
  'x1 sim',
  'x1 meas',
  'x1 est'
])


'''
plt.subplot(312)
plt.plot(tAxis[:N], Ws[0,:])
plt.plot(tAxis[:N_est-1], Ws_est[0,:])
plt.plot(tAxis[:N], Ws[1,:])
plt.plot(tAxis[:N_est-1], Ws_est[1,:])
plt.ylabel('w')
plt.xlabel('t [s]')
plt.legend([
  'w0 sim',
  'w0 est',
  'w1 sim',
  'w1 est'
])'''

plt.show()
