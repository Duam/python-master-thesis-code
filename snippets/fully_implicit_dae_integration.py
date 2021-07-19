#!/usr/bin/python3

import casadi as cas

dae = cas.DaeBuilder()

# ODE, DAE states
roll   = dae.add_x('roll')
(roll_vel, roll_acc) = dae.add_s('roll_vel')

# Velocity restoring force
z = dae.add_z('z')

# Meta information
dae.set_unit('roll', 'rad')
dae.set_unit('roll_vel', 'rad/s')
#dae.set_unit('der_roll_vel', 'rad/s^2')

dae.add_ode('roll_vel', roll_vel)
dae.add_dae('roll_acc', roll_acc - z*roll_vel)

dae.add_alg('roll_vel_setpoint', 0.5 * (roll_vel**2 - 4.0))

dae.disp(True)
dae.make_semi_explicit()
dae.disp(True)

# Reformulate dae
dae_dict = {
  'x': cas.vertcat(*dae.x),
  'ode': cas.vertcat(*dae.ode),
  'alg': cas.vertcat(*dae.alg),
  'z': z,
  'p': cas.vertcat(*dae.p)
}

print(dae_dict)

opts = {'tf':2.0}

I = cas.integrator('I', 'collocation', dae_dict, opts)
print(I)

x0 = cas.DM.zeros((2))
x0[1] = 1.0


res = I(x0=x0)

print(res)