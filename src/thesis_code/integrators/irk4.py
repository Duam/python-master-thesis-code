from casadi import DaeBuilder, Function, MX, vertcat

def irk4step_equations(dt, dae:DaeBuilder):

  # All symbolics
  x_sym = vertcat(*dae.x)
  z_sym = vertcat(*dae.z)
  u_sym = vertcat(*dae.u)
  p_sym = vertcat(*dae.p)
  ode_sym = vertcat(*dae.ode)
  alg_sym = vertcat(*dae.alg)

  # Function objects
  f = Function('ode', [x_sym,z_sym,u_sym,p_sym], [ode_sym])
  g = Function('alg', [x_sym,z_sym,u_sym,p_sym], [alg_sym])

  # Model sizes
  nx = x_sym.shape[0]
  nz = z_sym.shape[0]
  nu = u_sym.shape[0]
  np = p_sym.shape[0]

  # Decision variables
  x0 = MX.sym('x0',nx)
  xf = MX.sym('xf',nx)
  z0 = MX.sym('z0',nz)
  zf = MX.sym('zf',nz)
  k = MX.sym('k',4*nx)
  u = MX.sym('u',nu)
  p = MX.sym('p',np)

  # Aliases
  k0, k1, k2, k3 = k[0:nx], k[nx:2*nx], k[2*nx:3*nx], k[3*nx:4*nx]

  # Create the rootfinding system G(*) = 0
  G = Function('G', [x0,z0,k,xf,zf,u,p], [vertcat(
    g(x0,z0,u,p),
    k0 - f(x0,               z0,u,p),
    k1 - f(x0 + k0 * dt/2.0, z0,u,p),
    k2 - f(x0 + k1 * dt/2.0, z0,u,p),
    k3 - f(x0 + k2 * dt,     z0,u,p),
    xf - (x0 + dt/6.0 * (k0 + 2*k1 + 2*k2 + k3)),
    g(xf,zf,u,p)
  )])

  return G
