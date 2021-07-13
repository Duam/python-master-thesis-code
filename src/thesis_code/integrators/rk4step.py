##
# @brief One runge-kutta 4 step of an ordinary
#        differential equation xdot = f(x,u)
# @param f The time-continuous ODE to be integrated
# @param x The current state
# @param u The current control
# @param h The step size
# @return The state at the next timestep
##
def rk4step(f, x, u, dt, nn = 1):
  """ One Runge-Kutta 4 step for ODEs """
  h = dt / float(nn)
  xk = x

  for k in range(nn):
    k1 = f(xk,u)
    k2 = f(xk + h/2 * k1, u)
    k3 = f(xk + h/2 * k2, u)
    k4 = f(xk + h * k3, u)
    xk = xk + h/6 * (k1 + 2*k2 + 2*k3 + k4)

  return xk
