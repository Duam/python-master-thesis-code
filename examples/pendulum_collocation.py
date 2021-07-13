import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from thesis_code.models.pendulum import Pendulum
from thesis_code.integrators.collocation import Orthogonal_Collocation_Integrator as colint

## Create a pendulum
pendulum = Pendulum()
print(pendulum.toString())

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

# Model function (w is added in simulation)
f = lambda x, u: pendulum.ode(x,u,[0,0])
print(f(x0,[0,0]))

print("Simulation parameters set.")
################################################
###           CREATE AN INTEGRATOR           ###

model = colint(f, dt, 2, 2)
model.initSingleStepMode()

print("Collocation integrator created.")
################################################
## CREATE AN RK4 INTEGRATOR (FOR COMPARISON) ###

## Create rk4step function
def rk4step (f, x, u, h):
  k1 = f(x,u)
  k2 = f(x + h/2.0 * k1, u)
  k3 = f(x + h/2.0 * k2, u)
  k4 = f(x + h * k3, u)

  return x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

print("RK4 integrator created.")
################################################
###          SIMULATE THE SYSTEM             ###
print("Simulating system..")

# Choose controls and disturbances
Us = np.zeros((2,N))
Us[:,int(N/2):] = 3*np.ones((2,int(N/2)))

Ws = np.zeros((2,N))
for k in range(N):
  Ws[:,k] = np.random.multivariate_normal(np.zeros((2)), Q)

# Start the simulation (k=0 to k=N)
Xs_sim = np.zeros((2,N+1))
Xs_sim_comp = np.zeros((2,N+1))
Xs_sim[:,0] = x0
Xs_sim_comp[:,0] = x0
for k in range(N):
  xk = Xs_sim[:,k]
  uk = Us[:,k]
  wk = Ws[:,k]
  xnext = model.simulateStep(xk, uk).full().flatten()
  xnext += wk
  Xs_sim[:,k+1] = xnext

  xnext_comp = rk4step(f,xk,uk,dt).full().flatten()
  xnext_comp += wk
  Xs_sim_comp[:,k+1] = xnext_comp


print("Xs shape = ", Xs_sim.shape, " (states)")
print("Us shape = ", Us.shape, " (inputs)")
print("Ws shape = ", Ws.shape, " (disturbances)")

print("Simulation done.")

################################################
###         COMPUTE RMS DIFFERENCE           ###

# Create rms container
rms = np.zeros((2,N))

# Compute root mean square error
for k in range(N):
  for i in range(2):
    rms[i,k] = np.sqrt((Xs_sim[i,k] - Xs_sim_comp[i,k])**2)

print("Computed RMS difference.")

################################################
###         PLOT THE TRAJECTORIES            ###
print("Plotting trajectories...")

# Prepare plotting
tAxis = np.linspace(0,T,N+1)
plt.figure(1)

start = 0
end = N

# Controls
plt.subplot(321)
plt.title('Pendulum: Collocation method vs. RK4')
plt.plot(tAxis[start:end], Us[0,start:end])
plt.ylabel('u')

# Angle
plt.subplot(323)
plt.plot(tAxis[start:end], Xs_sim[0,start:end])
plt.plot(tAxis[start:end], Xs_sim_comp[0,start:end])
plt.legend(['collocation', 'rk4'])
plt.ylabel('ang.')

# Angle RMS
plt.subplot(324)
plt.gca().set_yscale('log')
plt.plot(tAxis[start:end], rms[0,start:end])
plt.ylabel('rms(ang.)')

# Angular velocity
plt.subplot(325)
plt.plot(tAxis[start:end], Xs_sim[1,start:end])
plt.plot(tAxis[start:end], Xs_sim_comp[1,start:end])
plt.legend(['collocation', 'rk4'])
plt.ylabel('ang. vel.')
plt.xlabel('t [s]')

# Angular velocity RMS
plt.subplot(326)
plt.gca().set_yscale('log')
plt.plot(tAxis[start:end], rms[1,start:end])
plt.ylabel('rms(ang. vel.)')
plt.xlabel('t [s]')

plt.show()