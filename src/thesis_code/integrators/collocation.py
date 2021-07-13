import casadi as cas

##
# @class Orthogonal_Collocation_Integrator
# @brief An implementation of the orthogonal
# collocation method. It can be used as a 
# one-step integrator for closed-loop simulation,
# but it can also supply a large system of 
# equations for whole trajectories for usage
# in optimal control problems.
# TODO: For now, only Gauss-Radau (d=3) is supported
## 
class Orthogonal_Collocation_Integrator:

  ##
  # @brief Creates the integrator
  # @param f The ode xdot = f(x,u)
  # @param dt The sampling time (double)
  # @param nx The number of states (int)
  # @param nu The number of controls (int)
  # @param d The collocation-order
  ##
  def __init__(self, f, dt, nx, nu, d = 3):

    # Fetch parameters
    self.dt = dt
    self.nx = nx
    self.nu = nu
    self.d  = d

    # Set collocation times (Gauss-Radau, d=3)
    xi = [0.0, 0.15505, 0.64495, 1.0]

    tk0 = dt * xi[0]
    tk1 = dt * xi[1]
    tk2 = dt * xi[2]
    tk3 = dt * xi[3]

    # Create lagrange polynomials
    lk0 = lambda t: (t-tk1)*(t-tk2)*(t-tk3) / ((tk0-tk1)*(tk0-tk2)*(tk0-tk3))
    lk1 = lambda t: (t-tk0)*(t-tk2)*(t-tk3) / ((tk1-tk0)*(tk1-tk2)*(tk1-tk3))
    lk2 = lambda t: (t-tk0)*(t-tk1)*(t-tk3) / ((tk2-tk0)*(tk2-tk1)*(tk2-tk3))
    lk3 = lambda t: (t-tk0)*(t-tk1)*(t-tk2) / ((tk3-tk0)*(tk3-tk1)*(tk3-tk2))

    # Create casadi expressions
    t = cas.SX.sym('t',1)
    x = cas.SX.sym('x', nx)
    u = cas.SX.sym('u', nu)
    v = cas.SX.sym('v', nx*(d+1))

    # Create aliases for the polynome coefficients
    vk0 = v[0*nx:1*nx]
    vk1 = v[1*nx:2*nx]
    vk2 = v[2*nx:3*nx]
    vk3 = v[3*nx:4*nx]

    # Create interpolation polynomial
    pk_expr = vk0 * lk0(t) + vk1 * lk1(t) + vk2 * lk2(t) + vk3 * lk3(t)

    # Create time-derivative of interpolation polynomial
    pkdot_expr = cas.jacobian(pk_expr,t)

    # Create intermediate function objects
    f = cas.Function('f', [x,u], [f(x,u)], ['x','u'], ['xdot'])
    self.pk = cas.Function('pk', [t,v], [pk_expr], ['t','v'], ['pk'])
    pkdot = cas.Function('pkdot', [t,v], [pkdot_expr], ['t','v'], ['pk'])

    # Create collocation equations
    collocation = cas.vertcat(
      vk0 - x,
      pkdot(tk1,v) - f(vk1,u),
      pkdot(tk2,v) - f(vk2,u),
      pkdot(tk3,v) - f(vk3,u)
    )

    # Create function object for the collocation equation
    self.c = cas.Function('c', [v,x,u], [collocation], ['v','x','u'], ['0'])

    # Create aliases for class-wide access
    self.t = t
    self.v = v
    self.vk0 = vk0
    self.vk1 = vk1
    self.vk2 = vk2
    self.vk3 = vk3


  ##
  # @brief Initializes the integrator
  # @param x0 The initial state (np.array)
  # @param Q Process noise covariance (np.array)
  ##
  def initSingleStepMode(self):
    # Create a rootfinder to solve the collocation equations
    self.rootfinder = cas.rootfinder('collocation', 'newton', self.c)
    # Initialize resulting collocation coefficients
    self.v_res = cas.DM.zeros(self.nx * (self.d + 1))

  ## 
  # @brief Single step simulation function
  # @param x The current state at t(k)
  # @param u The current control at t(k)
  # @return The next state at t(k+1)
  ##
  def simulateStep(self, x, u):
    # Create an initial guess for the polynomial coefficients
    # (Use the previous one)
    v_guess = self.v_res

    # Call the rootfinder to find the polynomial coefficients
    self.v_res = self.rootfinder(v_guess,x,u)

    # Compute the next state at t(k+1) = t(k) + dt
    xnext = self.pk(self.dt, self.v_res)
    return xnext
    

  ##
  # @brief
  ##
  def init_multiStep(self, x0, Q, N):
    pass


  