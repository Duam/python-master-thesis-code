#!/usr/bin/python3

##
# NOTES
# 

import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

t0 = 0
t1 = 1

d = 3

# Compute collocation times
xi_1 = 0.15505
xi_2 = 0.64495
xi_3 = 1.0

t0_0 = t0
t0_1 = t0 + (t1 - t0) * xi_1
t0_2 = t0 + (t1 - t0) * xi_2
t0_3 = t0 + (t1 - t0) * xi_3

# Create lagrange polynomials
l0_0 = lambda t: (t-t0_1)*(t-t0_2)*(t-t0_3) / ((t0_0-t0_1)*(t0_0-t0_2)*(t0_0-t0_3))
l0_1 = lambda t: (t-t0_0)*(t-t0_2)*(t-t0_3) / ((t0_1-t0_0)*(t0_1-t0_2)*(t0_1-t0_3))
l0_2 = lambda t: (t-t0_0)*(t-t0_1)*(t-t0_3) / ((t0_2-t0_0)*(t0_2-t0_1)*(t0_2-t0_3))
l0_3 = lambda t: (t-t0_0)*(t-t0_1)*(t-t0_2) / ((t0_3-t0_0)*(t0_3-t0_1)*(t0_3-t0_2))

# Create interpolation polynomial
n = 1
t = cas.SX.sym('t', 1)
v = cas.SX.sym('v', n*(d+1)) 
# COOL: each component of v determines the value of the overall
# interpolation polynomial at one of the collocation points
v0_0 = v[0:n]
v0_1 = v[n:2*n]
v0_2 = v[2*n:3*n]
v0_3 = v[3*n:4*n]
p_expr = v0_0*l0_0(t) + v0_1*l0_1(t) + v0_2*l0_2(t) + v0_3*l0_3(t)
p0 = cas.Function('p0', [t,v], [p_expr], ['t','v'], ['p0'])

# Create time-derivative of interpolation polynomial
pdot_expr = cas.jacobian(p_expr, t)
p0dot = cas.Function('p0dot', [t,v], [pdot_expr], ['t','v'], ['p0dot'])

# Define system dynamics (simple integrator)
x0 = cas.DM.ones(n)
x = cas.SX.sym('x', n)
xdot = cas.Function('xdot', [x], [x], ['x'], ['xdot'])

# Define system of equations (= 0)
eqsys = cas.vertcat(
  v0_0 - x0,
  p0dot(t0_1, v) - xdot(v0_1),
  p0dot(t0_2, v) - xdot(v0_2),
  p0dot(t0_3, v) - xdot(v0_3)
)
c = cas.Function('c', [v], [eqsys], ['v'], ['c'])

# Create a rootfinder to find v s.t. c = 0
rootfinder = cas.rootfinder('col', 'newton', c)
v_opt = rootfinder(cas.DM.zeros(n*(d+1)))

print(v_opt)

# Evaluate interpolation polynomial with the result
tAxis = np.linspace(t0,t1, 100)
p0_eval = np.zeros((n,tAxis.shape[0]))
for k in range(tAxis.shape[0]):
  p0_eval[:,k] = p0(tAxis[k], v_opt).full().flatten()

# Plot interpolation polynomial
plt.figure()
plt.plot(tAxis, np.exp(t0_0) * np.ones(tAxis.shape), '.', color='black')
plt.plot(tAxis, np.exp(t0_1) * np.ones(tAxis.shape), '.', color='black')
plt.plot(tAxis, np.exp(t0_2) * np.ones(tAxis.shape), '.', color='black')
plt.plot(tAxis, np.exp(t0_3) * np.ones(tAxis.shape), '.', color='black')
plt.plot(np.repeat(t0_0,50),np.linspace(0.5,3.0), '.', color='black')
plt.plot(np.repeat(t0_1,50),np.linspace(0.5,3.0), '.', color='black')
plt.plot(np.repeat(t0_2,50),np.linspace(0.5,3.0), '.', color='black')
plt.plot(np.repeat(t0_3,50),np.linspace(0.5,3.0), '.', color='black')
plt.plot(tAxis, p0_eval[0,:])
plt.show()



'''
# Evaluate lagrange polynomials
tAxis = np.linspace(t0,t1, 100)
l0_0_eval = np.zeros(tAxis.shape)
l0_1_eval = np.zeros(tAxis.shape)
l0_2_eval = np.zeros(tAxis.shape)
l0_3_eval = np.zeros(tAxis.shape)
for k in range(tAxis.shape[0]):
  l0_0_eval[k] = l0_0(tAxis[k])
  l0_1_eval[k] = l0_1(tAxis[k])
  l0_2_eval[k] = l0_2(tAxis[k])
  l0_3_eval[k] = l0_3(tAxis[k])

# Plot lagrange polynomials
plt.figure()
plt.gca().set_ylim([-0.5, 1.5])
plt.plot(tAxis, l0_0_eval)
plt.plot(tAxis, l0_1_eval)
plt.plot(tAxis, l0_2_eval)
plt.plot(tAxis, l0_3_eval)
plt.plot(tAxis, np.zeros(tAxis.shape), '.' ,color='black')
plt.plot(tAxis, np.ones(tAxis.shape), '.', color='black')
plt.plot(np.repeat(t0_0,50),np.linspace(-1.5,1.5), '.', color='black')
plt.plot(np.repeat(t0_1,50),np.linspace(-1.5,1.5), '.', color='black')
plt.plot(np.repeat(t0_2,50),np.linspace(-1.5,1.5), '.', color='black')
plt.plot(np.repeat(t0_3,50),np.linspace(-1.5,1.5), '.', color='black')
plt.legend(['l0','l1','l2','l3'])
plt.show()

# Evaluate interpolation polynomial
p0_eval = np.zeros((n,tAxis.shape[0]))
vs = np.array([1.0, -1.0, 0.0, 0.5]) 
for k in range(tAxis.shape[0]):
  p0_eval[:,k] = p0(tAxis[k], vs).full().flatten()

# Plot interpolation polynomial
plt.figure()
plt.plot(tAxis, np.zeros(tAxis.shape), '.' ,color='black')
plt.plot(tAxis, np.ones(tAxis.shape), '.', color='black')
plt.plot(np.repeat(t0_0,50),np.linspace(-1.5,1.5), '.', color='black')
plt.plot(np.repeat(t0_1,50),np.linspace(-1.5,1.5), '.', color='black')
plt.plot(np.repeat(t0_2,50),np.linspace(-1.5,1.5), '.', color='black')
plt.plot(np.repeat(t0_3,50),np.linspace(-1.5,1.5), '.', color='black')
plt.plot(tAxis, p0_eval[0,:])
plt.show()
'''