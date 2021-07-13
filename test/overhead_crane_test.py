import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from thesis_code.models.overhead_crane import OverheadCrane
from thesis_code.models.overhead_crane_viz import OverheadCrane_Visualizer
from thesis_code.integrators.rk4step import rk4step

print("This script tests the overhead-crane model and its visualizer")

# Create a model
plant = OverheadCrane()

# Simulation parameters
T = 10.0
N = 1000
dt = T/N
x0 = cas.vertcat(
  0.0,  # Cart position
  1e-6,  # Cart velocity
  -1.0,  # Cable length (!= 0)
  0.0,  # Cable length velocity
  0.0,  # Cable angle
  0.0   # Cable angular velocity
)

# Create system model in casadi
x = cas.MX.sym('x', 6, 1)
u = cas.MX.sym('u', 2, 1)
ode = cas.Function('ode', [x,u], [plant.ode(x,u)], ['x','u'], ['xdot'])
out = cas.Function('out', [x,u], [plant.output(x,u)], ['x','u'], ['y'])

# Create integrator
Xk = rk4step(ode, x, u, dt, 10)
F = cas.Function('F', [x,u], [Xk], ['x','u'], ['xnext'])

# Create a reference
Rs = cas.DM.zeros((2,4))
Rs[:,0] = cas.vertcat([-1.0,-1.0])
Rs[:,1] = cas.vertcat([1.0,-1.0])
Rs[:,2] = cas.vertcat([1.0,-2.0])
Rs[:,3] = cas.vertcat([-1.0,-2.0])

# Choose controls
Us = cas.DM.zeros((2,N))
Us[0,:] = cas.DM.ones((N)) * 1e-6 / plant.A
for k in range(int(N/2)-1):
  Us[0,k] = 1e-6 / plant.A + 100*np.sin(k/3.14)

# Simulate!
Xs = cas.DM.zeros((6,N+1))
Ys = cas.DM.zeros((2,N))
Xs[:,0] = x0
for k in range(N):
  Xs[:,k+1] = F(Xs[:,k], Us[:,k])
  Ys[:,k] = out(Xs[:,k], Us[:,k])

Xs = Xs.full()
Ys = Ys.full()

# Create a visualizer and run it
viz = OverheadCrane_Visualizer(Xs, Rs, dt)
anim = viz.createAnimation()
#anim.save("overhead_crane.mp4", fps=1/dt)
plt.show()
