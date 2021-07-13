#!/usr/bin/python3

import numpy as np
import casadi as cas
from models.simple.simple import Simple
from utils.signals import rectangle

model = Simple()

# Test evaluation
x0 = np.zeros(model._nx)
u0 = np.ones(model._nu)
p0 = np.zeros(model._np)
v0 = np.zeros(model._nv)

x1 = model.xnext(x0, u0, p0)
y0 = model.out(x0, u0, p0, v0)

print('x1 =', x1)
print('y0 =', y0)

# Create symbolics
print(model._fun_xnext_sym)

# Prepare simulation
Ntotal = 100
Nper = 10
u = lambda k: rectangle(k, Nper, 1.0)
p = lambda k: rectangle(k, Nper, 0.1, Nper/2)
v = lambda k: np.random.normal(0.0, 0e-1)

X = np.zeros(Ntotal+1)
X[0] = x0.flatten()
U = np.zeros(Ntotal)
P = np.zeros(Ntotal)
Y = np.zeros(Ntotal)

# Simulate
for k in range(Ntotal):
  U[k] = u(k)
  P[k] = p(k)
  X[k+1] = model.xnext(X[k], u(k), p(k))
  Y[k] = model.out(X[k], u(k), p(k), v(k))

# Plot
import matplotlib.pyplot as plt
# Create a figure with two subplots
fig, axes = plt.subplots(2,1,sharex='col')

# Populate the first subplot with state, control and output trajectories
plt.sca(axes[0])
plt.ylabel(r'State $x$, Control $u$, Output $y$')
plt.gca().plot(X, 'x-', color='green', label='State',  alpha=0.25)
plt.gca().plot(U, 'x--', color='red',   label='Control', alpha=0.25)
plt.gca().plot(Y, 'x-', color='blue',  label='Output')
plt.legend(loc='best')

# Populate the second subplot with parameters
plt.sca(axes[1])
plt.ylabel(r'Parameter $p$')
plt.gca().plot(P, 'x-', color='blue', label='Simulated')
plt.gca().legend(loc='best')

plt.show()