import numpy as np
import matplotlib.pyplot as plt
from thesis_code.models.carousel_greybox import Carousel

x0 = np.array([0.0, 0.0, 0.0, 0.0])
u0 = np.array([0.0])

x1 = np.array([0.0, 1e-3, 0.0, 1e-3])
u1 = np.array([0.0])

model = Carousel(subsamples=25)


# Nonlinear
print("f(x,u) =", model.ode(x0,u0))
print("F(x,u,dt) =", model.ode_discr(x0,u0,0.1))

# Linear
print("f_lin(x,u) =", model.ode_linearized(x0,u0,x0,u0))
print("F_lin(x,u,dt) =", model.ode_linearized_discr(x0,u0,x0,u0,0.1))

print("F(x,u,dt) =", model.ode_discr(x0,u0,0.001))
print("F_lin(x,u,dt) =", model.ode_linearized_discr(x0,u0,x0,u0,0.001))

plt.figure()
plt.title("One-step integration difference: nonlin+rk4 vs. linearized+analytic")
plt.gca().set_ylabel("2-Norm of difference")
plt.gca().set_xlabel("Timestep in seconds")
for dt in np.logspace(-5, 2, 100):
  # Compute next state for both
  xnext_nonlin = model.ode_discr(x0,u0,dt)
  xnext_lin = model.ode_linearized_discr(x0,u0,x0,u0,dt)

  # Compute error between them
  err = xnext_nonlin - xnext_lin
  err_mag = np.sqrt(np.dot(err.transpose(),err))

  # Plot error
  plt.plot(dt,err_mag, 'x', color='blue')

plt.gca().set_yscale('log')
plt.gca().set_xscale('log')  
plt.show()