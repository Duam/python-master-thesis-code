import numpy as np
from thesis_code.models.carousel_greybox import Carousel
from thesis_code.controllers.pf_nmpc import NMPC


# Define the system
plant = Carousel()

# Define initial state
x0 = np.array([
  [0.0],
  [0.0],
  [0.0],
  [0.0]
])

# Define simulation parameters
# (The carousel turns with 2rad/s, so it takes PI seconds for one revolution)
T = np.pi   # Seconds per turn
N = 100
dt = T/N
M = 1       # Number of turns

# Create reference state trajectory
# (For now, it should just stay put)
xref = np.array([
  [0.0]
])
uref = np.array([
  [0.0]
])

# Create casadi function for mpc (state evolution)
import casadi as cas
nx = 4
nu = 1
x_sym = cas.SX.sym('x', nx)
u_sym = cas.SX.sym('u', nu)
ode_discr = cas.Function('F', [x_sym,u_sym], [plant.ode_discr(x_sym,u_sym,dt,nn=5)])

# Create reference function for mpc (states)
nrx = 1
Rx = cas.Function('Rx', [x_sym], [x_sym[0]])

# Create PF_NMPC
N_mpc = 20
Q_mpc = 1e1 * np.ones((nrx,nrx))
P_mpc = 1e1 * np.ones((nrx,nrx))
R_mpc = 1e-3 * np.ones((nu,nu))

ctrl = NMPC(ode_discr, N_mpc, nx, nu, Rx, nrx)
ctrl.init(Q_mpc, R_mpc, P_mpc)

# Assemble reference for MPC
xkref = np.repeat(xref, N_mpc, 1)
ukref = np.repeat(uref, N_mpc-1, 1)

# Simulate the plant for M turns
xs_sim = np.zeros((4,M*N))
xs_sim[:,0] = x0.flatten()
ys_sim = np.zeros((2,M*N))
us_sim = np.zeros((1,M*N-1))
es_sim = np.zeros((1,N*M))
for k in range(M*N-1):

  # Fetch current state
  xk = xs_sim[:,k]

  # Compute new control using LMPC
  x, u, stats, result = ctrl.call(xk, xkref, ukref)
  
  if k == 0: 
    print("Result at k = 0")
    print(result)
    print('lam_g: init = ' + str(result['lam_g']['init']))
    print('lam_g: shoot = ' + str(result['lam_g']['shoot']))
    print('g: init = ' + str(result['g']['init']))
    print('g: shoot = ' + str(result['g']['shoot']))
    print('lam_p: xest = ' + str(result['lam_p']['xest']))
    print('p: xest = ' + str(result['p']['xest']))
  us_sim[:,k] = u[:,0]

  

  print('k = ' + str(k) + ', f = ' + str(result['f']))

  # Apply new control
  xs_sim[:,k+1] = plant.ode_discr(xk, us_sim[:,k], dt, nn=10).full().flatten()


# Plot stuff
import matplotlib.pyplot as plt
tAxis = np.linspace(0,M*T,M*N)

xfig = plt.figure(1)
plt.subplot(411)
plt.plot(tAxis,xs_sim[0,:])
plt.subplot(412)
plt.plot(tAxis,xs_sim[1,:])
plt.subplot(413)
plt.plot(tAxis,xs_sim[2,:])
plt.subplot(414)
plt.plot(tAxis,xs_sim[3,:])

ufig = plt.figure(2)
plt.plot(tAxis[:M*N-1], us_sim[0,:])

plt.show()