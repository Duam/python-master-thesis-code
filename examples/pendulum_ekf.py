import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from thesis_code.estimators.EKF import EKF
from thesis_code.models.pendulum import Pendulum
from thesis_code.integrators.rk4step import rk4step

print("This script tests state estimation with an EKF on a pendulum model.")

pendulum = Pendulum()
print(pendulum.toString())

# Simulation parameters
T = 10.0 # seconds
N = 100 # samples
dt = T/N # Sample time
nn = 10 # integrator steps per step
dt_n = dt/nn 

# Initial state
x0 = cas.vertcat([-np.pi/2.0, 0])

# Create system model in CasADi
x = cas.MX.sym('x', 2, 1)
u = cas.MX.sym('u', 2, 1)
ode = cas.Function('ode', [x,u], [pendulum.ode(x,u,[0,0])], ['x','u'], ['xdot'])

# Discretize the system
Xk = x
for k in range(nn):
    Xk = rk4step(ode, Xk, u, dt_n)
f = cas.Function('f', [x,u], [Xk], ['x','u'], ['xnext'])

# Create F,h,H functions for the EKF
F = cas.Function('F', [x,u], [cas.jacobian(Xk, x)], ['x','u'], ['F'])
h = cas.Function('h', [x,u], [pendulum.output(x,u,[0,0])], ['x','u'], ['h'])
H = cas.Function('H', [x,u], [cas.jacobian(pendulum.output(x,u,[0,0]),x)], ['x','u'], ['H'])

# Set EKF parameters
x0hat = cas.vertcat([0.0, 0.0])
P0 = cas.vertcat(
  cas.horzcat(0.001, 0.000),
  cas.horzcat(0.000, 0.001)
)
Q = cas.vertcat(
  cas.horzcat(0.001, 0.000),
  cas.horzcat(0.000, 0.001)
)
R = cas.vertcat(
  cas.horzcat(0.001, 0.000),
  cas.horzcat(0.000, 0.001)
)

# Create an EKF object
ekf = EKF(f, F, h, H, x0hat, P0, Q, R)

# Choose controls (k=0 to k=N-1)
Us = cas.DM.zeros((2, N))

# Start the simulation (k=0 to k=N)
Xs_sim = cas.DM.zeros((2,N+1))
Xs_sim[:,0] = x0
for k in range(N):
  Xs_sim[:,k+1] = f(Xs_sim[:,k], Us[:,k])
  
Xs_sim = Xs_sim.full()

# Compute the measurements (k=0 to K=N-1)
Ys_sim = cas.DM.zeros((2,N))
for k in range(N):
  Ys_sim[:,k] = pendulum.output(Xs_sim[:,k], Us[:,k], [0,0])

# Estimate the state using the EKF
Xs_est = cas.DM.zeros((2,N+1))
Xs_est[:,0] = x0hat
for k in range(N-1):
  # Predict to k+1
  u_prev = Us[:,k]
  ekf.predict(Us[:,k], k)

  # Correct with values from k+1
  u_cur = Us[:,k+1]
  y_cur = Ys_sim[:,k+1]
  ekf.correct(u_cur, y_cur)

  # Save estimate
  Xs_est[:,k+1] = ekf.xhat

Xs_est = Xs_est.full()

# Prepare plotting
tAxis = np.linspace(0, T, N+1)
plt.figure(1)

# Plot
plt.subplot(211)
plt.plot(tAxis, Xs_sim[0,:])
plt.plot(tAxis[:N], Xs_est[0,:N], '.-')
plt.ylabel('Angle [rad]')
plt.xlabel('t [s]')

plt.subplot(212)
plt.plot(tAxis, Xs_sim[1,:])
plt.plot(tAxis[:N], Xs_est[1,:N], '.-')
plt.ylabel('Angular velocity [rad/s]')
plt.xlabel('t [s]')

plt.show()
