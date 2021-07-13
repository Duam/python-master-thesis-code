import casadi as cas
import numpy as np

def simpleColl(dae_dict:dict, tau_root:list, h:float):
  """simpleColl Creates the collocation equations
  Args:
    dae[dict] -- A dict with the DAE symbolics
    tau_root[list] -- The collocation points
    h[float] -- The timestep
  Returns:
    CasADi function with the collocation and continuity equations
  """

  daefun = cas.Function("fun", dae_dict, ["x","z","p"], ["ode","alg","quad"])
  
  # Degree of interpolating polynomial
  d = len(tau_root)-1

  # Coefficients of the collocation equation
  C = np.zeros((d+1,d+1))

  # Coefficients of the continuity equation
  D = np.zeros(d+1)

  # Dimensionless time inside one control interval
  tau = cas.MX.sym("tau")

  # For all collocation points
  for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    L = 1
    for r in range(d+1):
      if r != j:
        L *= (tau-tau_root[r])/(tau_root[j]-tau_root[r])
    lfcn = cas.Function('lfcn', [tau],[L])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = lfcn(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    #tfcn = lfcn.tangent()
    #tfcn = lfcn.jacobian()
    tfcn = cas.Function('tfcn', [tau], [cas.jacobian(L,tau)])
    for r in range(d+1):
      #C[j,r], _ = tfcn([tau_root[r]])
      C[j,r] = tfcn([tau_root[r]])

  # State variable
  CVx  = cas.MX.sym("x",dae_dict["x"].size1(),1)

  # Helper state variables
  CVCx  = cas.MX.sym("x",dae_dict["x"].size1(),d)

  # Algebraic variables
  CVz  = cas.MX.sym("z",dae_dict["z"].size1(),d)

  # Fixed parameters (controls)
  CVp  = cas.MX.sym("p",dae_dict["p"].size1())

  X = cas.horzcat(CVx,CVCx)
  g = []

  # For all collocation points
  for j in range(1,d+1):

    # Get an expression for the state derivative at the collocation point
    xp_jk = 0
    for r in range (d+1):
      xp_jk += C[r,j]*X[:,r]

    # Add collocation equations to the NLP
    [ode,alg,_] = daefun(
      CVCx[:,j-1],CVz[:,j-1],CVp
    )
    g.append(h*ode - xp_jk)
    g.append(alg)

  # Get an expression for the state at the end of the finite element
  xf_k = 0
  for r in range(d+1):
    xf_k += D[r]*X[:,r]

  G = cas.Function("G",[CVx,CVCx,CVz,CVp],[xf_k,cas.vertcat(*g)])

  return G