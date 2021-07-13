import numpy as np
from thesis_code.models.carousel_greybox import Carousel

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
M = 3       # Number of turns

# Create reference state trajectory
# (Outputs: Elevation and pitch)
# (For now, it should just stay put)
yref = np.array([
  [0.0],
  [0.0]
])

# PID tuning parameters
ctrl_p = 3.5
ctrl_i = -0.02
ctrl_d = 0.0

# Simulate the plant for M turns
xs_sim = np.zeros((4,M*N))
xs_sim[:,0] = x0.flatten()
ys_sim = np.zeros((2,M*N))
us_sim = np.zeros((1,M*N-1))
es_sim = np.zeros((1,N*M))
e_int = 0.0
for k in range(M*N-1):

  # Compute current model output and error
  y = plant.out(xs_sim[:,k], us_sim[:,k])
  es_sim[:,k] = np.linalg.norm(y - yref)
  if np.linalg.norm(y) < np.linalg.norm(yref):
    es_sim[:,k] *= (-1)

  # Compute error integral and derivative
  e_int += es_sim[:,k]
  e_der = es_sim[:,k] - es_sim[:,k-1] if k > 0 else 0

  # Compute new control using PID
  us_sim[:,k] = ctrl_p * es_sim[:,k] + ctrl_i * e_int + ctrl_d * (es_sim[:,k] - es_sim[:,k-1])

  # Apply new control
  xs_sim[:,k+1] = plant.ode_discr(xs_sim[:,k], us_sim[:,k],dt,nn=10).full().flatten()


# Plot stuff
import matplotlib.pyplot as plt
tAxis = np.linspace(0,M*T,M*N)

fig,axes = plt.subplots(5,1)
axes = axes.flatten()

for k in range(axes.shape[0]):
  plt.sca(axes[k])
  plt.gca().grid()
  if k < 1: 
    plt.gca().plot(tAxis[:-1], us_sim[0,:])
    plt.ylabel('u')
  else:      
    plt.gca().plot(tAxis, xs_sim[k-1,:])
    plt.ylabel(r'$x_'+str(k-1)+r"$")

plt.xlabel('Time')

plt.show()