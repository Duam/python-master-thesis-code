#!/usr/bin/python3

import casadi as cas

# Create a dummy system
dae = cas.DaeBuilder()
p = dae.add_x('pos')
v = dae.add_x('vel')
a = dae.add_z('acc')
u = dae.add_u('input')
dae.add_ode('der_p', v)
dae.add_ode('der_v', a)
dae.add_alg('force', a - u)
dae.scale_variables()
dae.make_semi_explicit()

# Fetch system internals
dt = cas.MX.sym('dt')
dae_dict = {
  'x': cas.vertcat(*dae.x),
  'ode': dt * cas.vertcat(*dae.ode),
  'alg': cas.vertcat(*dae.alg),
  'z': cas.vertcat(*dae.z),
  'p': cas.vertcat(dt, *dae.p, *dae.u)
}

# Create integrator
int_dict = {'number_of_finite_elements': 1, 'tf': 1.0}
integrator = cas.integrator('xnext', 'collocation', dae_dict, int_dict)
T = 10

print(0.5 * 1 * T**2)

N = 20
dt = float(T)/N
Xs = cas.DM.zeros((2,N+1))
Us = cas.DM.ones((1,N))
for k in range(N):
  step = integrator(x0=Xs[:,k],p=cas.vertcat(dt,Us[:,k]))
  Xs[:,k+1] = step['xf'].full().flatten()

print(Xs[:,-1])



N = 40
dt = float(T)/N
Xs = cas.DM.zeros((2,N+1))
Us = cas.DM.ones((1,N))
for k in range(N):
  step = integrator(x0=Xs[:,k],p=cas.vertcat(dt,Us[:,k]))
  Xs[:,k+1] = step['xf'].full().flatten()

print(Xs[:,-1])