#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/06/24
  Brief: This script finds identifies chosen carousel model parameters
"""

""" =========================================       PREPARATION       ============================================= """

import csv
import casadi as cas
from casadi import vertcat, symvar
from thesis_code.carousel_model import CarouselWhiteBoxModel
from thesis_code.utils.CollocationHelper import simpleColl
from scipy.interpolate import CubicSpline
import numpy as np
np.set_printoptions(linewidth=np.inf)
import pprint
from thesis_code.utils.ParametricNLP import ParametricNLP

def reassemble_P_MX(NP, Pvar, Pfix, var_idx):
  # Construct full parameter vector
  Pfull = cas.MX.zeros((NP,1))
  var_cnt = fix_cnt = 0
  for k in range(NP):
    if k in var_idx:
      Pfull[k] = Pvar[var_cnt]
      var_cnt += 1
    else:
      Pfull[k] = Pfix[fix_cnt]
      fix_cnt += 1
  return Pfull

def reassemble_P_DM(NP, Pvar, Pfix, var_idx):
  # Construct full parameter vector
  Pfull = cas.DM.zeros((NP,1))
  var_cnt = fix_cnt = 0
  for k in range(NP):
    if k in var_idx:
      Pfull[k] = Pvar[var_cnt]
      var_cnt += 1
    else:
      Pfull[k] = Pfix[fix_cnt]
      fix_cnt += 1
  return Pfull

""" =========================================       PREPROCESSING       ============================================ """
# 1. Load data (phi, theta, psi, psidot, delta_e) from files
controls_filename = 'set_VESTIBULUS_2_KINEOS_4_SETTABLE_SETPOINT.csv'
angles_filename = 'ANGLE_sampled.csv'
carousel_encoder_filename = 'CAROUSEL_CAROUSELENCODERPOSITION.csv'

# Load the controls
control = []
with open(controls_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      control.append([int(row[0]), float(row[1])])

# Load elevation and pitch angles
phi = []
theta = []
with open(angles_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      phi.append([int(row[0]), float(row[1])])
      theta.append([int(row[0]), float(row[2])])

# Load carousel yaw angles
psi = []
with open(carousel_encoder_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      psi.append([int(row[0]),float(row[1])])

# 2. Order data (ascending wrt timestamps)
control = np.array(sorted(control, key=lambda elem: elem[0]))
phi     = np.array(sorted(phi,     key=lambda elem: elem[0]))
theta   = np.array(sorted(theta,   key=lambda elem: elem[0]))
psi     = np.array(sorted(psi,     key=lambda elem: elem[0]))

# Convert from nanoseconds to seconds
control[:,0] *= 1e-9
phi[:,0] *= 1e-9
theta[:,0] *= 1e-9
psi[:,0] *= 1e-9

# Store earliest and latest timestamps
t_start_control = control[0,0]
t_start_phi     = phi[0,0]
t_start_theta   = theta[0,0]
t_start_psi     = psi[0,0]
t_end_control   = control[-1,0]
t_end_phi       = phi[-1,0]
t_end_theta     = theta[-1,0]
t_end_psi       = psi[-1,0]

# Rebase timestamp to 0
control[:,0] = control[:,0] - t_start_control
phi[:,0]     = phi[:,0]     - t_start_phi
theta[:,0]   = theta[:,0]   - t_start_theta
psi[:,0]     = psi[:,0]     - t_start_psi

# 3. Do a cubic spline fit
control_cs = CubicSpline(control[:,0], control[:,1])
phi_cs     = CubicSpline(phi[:,0], phi[:,1])
theta_cs   = CubicSpline(theta[:,0], theta[:,1])
psi_cs     = CubicSpline(psi[:,0], psi[:,1])

# Create a getter for the angle states
get_angles = lambda t_axis: np.array([
  phi_cs(t_axis),
  theta_cs(t_axis),
  psi_cs(t_axis)
])

get_angular_velocities = lambda t_axis: np.array([
  phi_cs(t_axis,1),
  theta_cs(t_axis,1),
  psi_cs(t_axis,1)
])

get_angular_accelerations = lambda t_axis: np.array([
  phi_cs(t_axis,2),
  theta_cs(t_axis,2),
  psi_cs(t_axis,2)
])

# Store latest first timestamp and earliest last timestamp, create time-axis
dt = 1./20.
t_start = max([t_start_control, t_start_phi, t_start_theta, t_start_psi])
t_end   = min([t_end_control,   t_end_phi,   t_end_theta,   t_end_psi])
duration = t_end - t_start

# Compute sample-numbers
# Number of time-grid-points
N = int(np.ceil(duration/dt)) 
# Collocation nodes per interval
ncol = 3
tau_root = [0] + cas.collocation_points(ncol, 'radau')
# Total number of collocation nodes
Ncol = (N-1)*ncol
# Total number of samples and nodes
Ntot = Ncol + 1


# Create time-grids
tAxis = np.array([ k*dt for k in range(N) ])
tAxis_coll = np.repeat(tAxis, ncol) + dt*np.tile(tau_root[1:],N)
tAxis_coll = tAxis_coll[:-ncol]
tAxis_full = np.concatenate(([0],tAxis_coll))

#tAxis  = np.arange(0., t_end-t_start, dt)
#tcAxis = np.array([0] + [ dt*int(k/ncol) + dt*tau_root[np.mod(k,ncol)+1] for k in range(Ncol-1) ])

# Create time-axis with collocation points in-between the discrete grid



print("dt:", dt)
print("Samples:", N)
print("Collocation nodes:", Ncol, " (", str(tau_root) + " per interval )")

""" =========================================       MODEL SETUP       ============================================= """

# Get defaul model parameters
print("Creating model parameters..")
model_params = CarouselWhiteBoxModel.getDefaultParams()

# Create a carousel model
print("Creating model..")
model = CarouselWhiteBoxModel(model_params)

# Print parameters
print(" ---------------------- Model Parameters ---------------------- ")
pprint.pprint(model.params)
print(" -------------------------------------------------------------- ")

# Sizes
NX = model.NX()
NZ = model.NZ()
NU = model.NU()
NP = model.NP()

# The semi-explicit DAE
dae = model.dae

# All symbolics
x_sym = cas.vertcat(*dae.x)
z_sym = cas.vertcat(*dae.z)
u_sym = cas.vertcat(*dae.u)
p_sym = cas.vertcat(*dae.p)
ode_sym = cas.vertcat(*dae.ode)
alg_sym = cas.vertcat(*dae.alg)
quad_sym = cas.vertcat(*dae.quad)

# Function objects
ode_fun = cas.Function('ode', [x_sym,z_sym,u_sym,p_sym], [cas.vertcat(*dae.ode)])
alg_fun = cas.Function('alg', [x_sym,z_sym,u_sym,p_sym], [cas.vertcat(*dae.alg)])

# Set up output function
out_fun = cas.Function('out', [x_sym,u_sym,p_sym], [x_sym[:6]])

""" ====================================       SYSTEM AUGMENTATION       =========================================== """
print("Preparing system..")

for k in range(p_sym.shape[0]):
  print(k, ":", p_sym[k])
  


quit(0)
# Select parameters that will not be subject to identification_imu
# Don't select vector subcomponents. Always the whole vector/matrix
idx_fix_parameters = []
idx_fix_parameters += [0,1,2,3] # Carousel speed and time constant, gravity, air density, mass
idx_fix_parameters += [5,6,7,8,9,10,11,12,13] # Inertia tensor
idx_fix_parameters += [14,15,16] # Center of mass
idx_fix_parameters += [17,18,19] # Wing aerodynamic center
idx_fix_parameters += [20,21,22] # Elevator aerodynamic center
idx_fix_parameters += [23,24,25] # IMU position
idx_fix_parameters += [26,27,28] # 4 wrt 3
idx_fix_parameters += [29,30,31] # 3 wrt 2
idx_fix_parameters += [32,33] # Aileron and wing surface areas
idx_var_parameters = [k for k in range(p_sym.rows()) if k not in idx_fix_parameters]

# Split up parameter vector into fixed and variable parameters (using MX-hack)
p_fix_sym = vertcat(*symvar(vertcat(*[p_sym[i] for i in idx_fix_parameters])))
p_var_sym = vertcat(*symvar(vertcat(*[p_sym[i] for i in idx_var_parameters])))

NP_fix = p_fix_sym.rows()
NP_var = p_var_sym.rows()

# Augment ode and alg to be able to exclude fixed parameters
ode_aug_fun = cas.Function('ode', [x_sym,z_sym,u_sym,p_var_sym,p_fix_sym], [ode_fun(x_sym,z_sym,u_sym,p_sym)])
alg_aug_fun = cas.Function('alg', [x_sym,z_sym,u_sym,p_var_sym,p_fix_sym], [alg_fun(x_sym,z_sym,u_sym,p_sym)])

""" ====================================       IDENTIFICATION SETUP       ========================================== """
print("Preparing identification_imu ocp..")
# Create collocation integrator
G = simpleColl({
    'x': x_sym,
    'z': z_sym,
    'p': cas.vertcat(p_sym, u_sym),
    'ode': ode_sym,
    'alg': alg_sym,
    'quad': quad_sym
  }, tau_root, dt
)

# Create the parametric NLP
ocp = ParametricNLP('ModelIdentification')
ocp.add_decision_var('Pvar', (NP_var, 1)) # Parameters to identify
ocp.add_decision_var('X', (NX, N)) # States
ocp.add_decision_var('Z', (NZ, Ncol)) # Algebraic states
ocp.add_decision_var('V', (NX, Ncol)) # States at collocation points
ocp.add_parameter('Pfix', (NP_fix,1))
ocp.add_parameter('Pvar_prior', (NP_var,1))
ocp.add_parameter('U', (NU, N))
ocp.add_parameter('Y', (out_fun.size1_out(0), N))
ocp.bake_variables()

# Fetch symbolics
Pvar_sym = ocp.get_decision_var('Pvar')
X_sym = ocp.get_decision_var('X')
Z_sym = ocp.get_decision_var('Z')
V_sym = ocp.get_decision_var('V')
Pfix_sym = ocp.get_parameter('Pfix')
Pvar_prior_sym = ocp.get_parameter('Pvar_prior')
U_sym = ocp.get_parameter('U')
Y_sym = ocp.get_parameter('Y')

# Construct full parameter vector
P_sym = reassemble_P_MX(NP, Pvar_sym, Pfix_sym, idx_var_parameters)

# Set up the residual function
residual_x = []
residual_y = [('res_y_0',Y_sym[:,0] - out_fun(X_sym[:,0],U_sym[:,0],P_sym))]
G_colloc_ode = []
G_colloc_alg = []
for k in range(N-1):
  x0_k = X_sym[:,k]
  vc_k = V_sym[:,ncol*k:ncol*(k+1)]
  z0_k = Z_sym[:,ncol*k:ncol*(k+1)]
  u0_k = U_sym[:,k] 
  xf_k, g_k = G(x0_k, vc_k, z0_k, vertcat(P_sym,u0_k))

  # Set up residual function (assume covariance 1)
  residual_x.append(('res_x_'+str(k+1), X_sym[:,k+1] - xf_k))
  residual_y.append(('res_y_'+str(k+1), Y_sym[:,k+1] - out_fun(X_sym[:,k+1],U_sym[:,k+1],P_sym)))
  
  # Set up collocation functions
  coll_name = 'coll_'+str(k)
  ocp.add_equality(coll_name, g_k)

  for j in range(ncol):
    base = j * (NX + NZ)
    G_colloc_ode.append((coll_name+'_ode_node_'+str(j), g_k[base   :base+NX]))
    G_colloc_alg.append((coll_name+'_alg_node_'+str(j), g_k[base+NX:base+NX+NZ]))


# Set up optimization problem
residual = cas.vertcat(
  *[res[1] for res in residual_x],
  *[res[1] for res in residual_y]
)

COST  = 0.5 * cas.mtimes([residual.T, residual])
COST += 1e-3 * cas.mtimes([(Pvar_sym-Pvar_prior_sym).T,(Pvar_sym-Pvar_prior_sym)])
#COST += 1e-3 * sum([ cas.mtimes([X_sym[:,k].T,X_sym[:,k]]) for k in range(X_sym.shape[1]) ])
#COST += 1e-3 * sum([ cas.mtimes([Z_sym[:,k].T,Z_sym[:,k]]) for k in range(Z_sym.shape[1]) ])
#COST += 1e-3 * sum([ cas.mtimes([V_sym[:,k].T,V_sym[:,k]]) for k in range(V_sym.shape[1]) ])
ocp.set_cost(COST)

print("Initializing ocp..")
ocp.init(
  nlpsolver = 'ipopt',
  opts = {
    #'ipopt.constr_viol_tol': 1e-5,
    'ipopt.max_iter': 1000,
    #'ipopt.hessian_approximation': 'limited-memory',
    'ipopt.hessian_approximation': 'exact',
    'ipopt.print_level':5,'print_time': 1, 'ipopt.print_timing_statistics': 'no', 'ipopt.sb': 'yes'}
)
print("Initialized.")
print(ocp)
print(len(G_colloc_alg))

""" ========================================      HELPER FUNCTIONS      ============================================ """
# Fetch symbolics
w_ocp_sym = ocp.struct_w
p_ocp_sym = ocp.struct_p

# Create getters for residual costs
residual_x_fun = [ cas.Function(res[0], [w_ocp_sym,p_ocp_sym], [res[1]]) for res in residual_x ]
residual_y_fun = [ cas.Function(res[0], [w_ocp_sym,p_ocp_sym], [res[1]]) for res in residual_y ]

# Create getters for collocation constraints
colloc_ode_fun = dict([ (coll[0], cas.Function(coll[0], [w_ocp_sym,p_ocp_sym], [coll[1]])) for coll in G_colloc_ode ])
colloc_alg_fun = dict([ (coll[0], cas.Function(coll[0], [w_ocp_sym,p_ocp_sym], [coll[1]])) for coll in G_colloc_alg ])

""" ==================================      INITIAL GUESS & PARAMETERS      ======================================== """
# The initial guesses for the deflection angle and the joint constraint moment
# can be obtained by forward simulation of the a-priori model
Usim = cas.vertcat(cas.DM(control_cs(tAxis_full)).T, -2*np.ones((1,Ntot)))
Xsim = cas.vertcat(get_angles(tAxis), get_angular_velocities(tAxis), 0.5*cas.DM.ones((1,N)))
Vsim = cas.vertcat(get_angles(tAxis_full), get_angular_velocities(tAxis_full), 0.5*cas.DM.ones((1,Ntot)))
Zsim = cas.vertcat(get_angular_accelerations(tAxis_full)[:2,:], cas.DM.zeros((1,Ntot)))

# Create one integrator for each collocation node (each with a different dt)
dae_dict = {'x': x_sym, 'ode': ode_sym, 'alg': alg_sym, 'z': z_sym, 'p': cas.vertcat(p_sym, u_sym)}
int_opts = [ {'number_of_finite_elements': 1, 'output_t0': True, 'tf': dt*(tau_root[k+1] - tau_root[k])} for k in range(ncol) ]
integrators = [ cas.integrator('xnext','collocation',dae_dict, opt) for opt in int_opts ]

# Simulate
print("Simulating..")
for k in range(Ntot-1):
  print("Step", k+1, "of", Ncol)
  # Fetch states, parameters and controls
  vk = Vsim[:,k]
  zk = Zsim[:,k]
  pk = cas.vertcat(model.p0(), Usim[:,k])

  # Choose the integrator with the correct dt
  integrator = integrators[np.mod(k,ncol)]

  # Simulate one step. For the differential states, we only update the deflection angle.
  # The other states were measured and are taken as fact. We update the algebraic states
  # in order to get consistent algebraic equations.
  simstep = integrator(x0=vk,p=pk,z0=zk)
  if k+1 < Ntot:
    Vsim[6,k+1] = simstep['xf'][6,1]
    Zsim[:,k+1] = simstep['zf'][:,1]
  
  if k == 0:
    print(Zsim[:,0], simstep['zf'][:,0])

  # Every ncol^th state is stored in a coarser grid
  if np.mod(k,ncol) == 0:
    Xsim[:,int(k/ncol)] = Vsim[:,k]

Xsim[:,-1] = Vsim[:,-1]

# Create initial guess
initial_guess = ocp.struct_w(0)
initial_guess['Pvar'] = cas.vertcat(*[model.p0()[i] for i in idx_var_parameters])
initial_guess['X'] = Xsim
initial_guess['Z'] = Zsim[:,1:]
initial_guess['V'] = Vsim[:,1:]

# Create parameters
parameters = ocp.struct_p(0)
parameters['Pfix'] = cas.vertcat(*[model.p0()[i] for i in idx_fix_parameters])
parameters['Pvar_prior'] = cas.vertcat(*[model.p0()[i] for i in idx_var_parameters])
parameters['U'] = cas.vertcat(cas.DM(control_cs(tAxis)).T, -2*np.ones((1,N)))
parameters['Y'] = Xsim[:-1,:]

# Construct full parameter vector
P_init = reassemble_P_DM(NP, initial_guess['Pvar'], parameters['Pfix'], idx_var_parameters)

""" =============================================     TEST     ===================================================== """
# Before we solve: Test evaluation
print("f_init =", ocp.getf()(initial_guess,parameters))
#print("G_init =", ocp.getG()(initial_guess,parameters))

"""
print("==================== COLLOCATION ODE RESIDUAL ========================")
for key in colloc_ode_fun.keys(): print(key, "=", colloc_ode_fun[key](initial_guess,parameters))
print("==================== COLLOCATION ALG RESIDUAL ========================")
for key in colloc_alg_fun.keys(): print(key, "=", colloc_alg_fun[key](initial_guess,parameters))
"""

"""
print("=== State transition residual values for initial guess ===")
for fun in residual_x_fun:
  res = fun(initial_guess,parameters).full()
  print(fun.name(), "=", res.T)

print("=== Measurement residual values for initial guess ===")
for fun in residual_y_fun:
  res = fun(initial_guess,parameters).full()
  print(fun.name(), "=", res.T)
"""

# Construct full parameter vector
"""
p = cas.DM.zeros((NP,1))
var_cnt = fix_cnt = 0
for k in range(NP):
  if k in idx_var_parameters:
    p[k] = initial_guess['Pvar'][var_cnt]
    var_cnt += 1
  else:
    p[k] = parameters['Pfix'][fix_cnt]
    fix_cnt += 1
"""
p = reassemble_P_DM(NP, initial_guess['Pvar'], parameters['Pfix'], idx_var_parameters)

#quit(0)

# Evalute all collocation nodes once
for k in range(N-1):
  print("=======================================================")
  print("COLLOCATION at k =", k)
  x0_k = initial_guess['X'][:,k].full()
  vc_k = initial_guess['V'][:,ncol*k:ncol*(k+1)].full()
  zc_k = initial_guess['Z'][:,ncol*k:ncol*(k+1)].full()
  u0_k = parameters['U'][:,k].full()
  p_k = vertcat(p.full(), u0_k)

  print("x0_k     =", x0_k[:,0])
  print("vc_k (0) =", vc_k[:,0])
  print("vc_k (1) =", vc_k[:,1])
  print("vc_k (2) =", vc_k[:,2])
  print("zc_k (0) =", zc_k[:,0])
  print("zc_k (1) =", zc_k[:,1])
  print("zc_k (2) =", zc_k[:,2])
  print("u0_k     =", u0_k[:,0])
  
  CVx = x0_k
  CVCx = vc_k
  CVz = z0_k
  CVp = cas.vertcat(p_k, u0_k)
  xf_k, g_k = G(x0_k,vc_k,zc_k,p_k)
  
  print("xf_COLL =", xf_k)
  print("xf_TRUE =", initial_guess['X'][:,k+1])

  for i in range(ncol):
    base = i * 10
    print("Collocation node",i)
    print("\t ode (res) =", g_k[base:base+NX])
    print("\t alg =", g_k[base+NX:base+NX+NZ])

  print("INFNORM =", np.max(g_k.full()))

""" ===========================================      SOLVING      ================================================== """

#quit(0)
# Solve!
sol, stats, sol_orig, lb, ub = ocp.solve(initial_guess, parameters)

# Fetch solution
pvar = sol['w']['Pvar']
pfix = parameters['Pfix']

pvar_prior = initial_guess['Pvar']

# Construct full parameter vector
p = cas.DM.zeros((NP,1))
var_cnt = fix_cnt = 0
for k in range(NP):
  if k in idx_var_parameters:
    p[k] = pvar[var_cnt]
    var_cnt += 1
  else:
    p[k] = pfix[fix_cnt]
    fix_cnt += 1


print("================ IDENT RESULT ==============")
cnt = 0
for k in range(P_sym.rows()):
  if k not in idx_fix_parameters:
    print(p_sym[k], ":", pvar_prior[cnt], "-->", pvar[cnt])
    cnt = cnt + 1
