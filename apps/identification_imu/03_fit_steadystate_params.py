from thesis_code.model import CarouselModel
from thesis_code.utils.ParametricNLP import ParametricNLP
import pprint, json
from thesis_code.utils.bcolors import bcolors
import casadi as cs
import numpy as np
from scipy import linalg
import pandas as pd
from pathlib import Path
import sys
np.set_printoptions(threshold=sys.maxsize)

# Set paths
dataset_path = "data_phys/CAROUSEL_ZCMLOG_2019_09_13_STEADY_STATE_CONTROL.zcmlog.PREPROCESSED.csv"
param_file_in = "params_initial.json"
param_file_out = "./params_identified_ss_new.json"
param_vars_file_out = "params_identified_ss_variances.json"

""" ================================================================================================================ """
# Check if input params exist. If not, create them
input_file = Path(param_file_in)
if not input_file.exists():
    print("Warning: " + param_file_in + " does not exist. Creating default params..")
    CarouselModel().writeParamsToJson(param_file_in)

# Create a model
model_params = {}
print("Loading parameter set " + param_file_in)
with open(param_file_in, 'r') as file:
    model_params = json.load(file)

print("Creating model..")
model = CarouselModel(model_params, with_angle_output=True, with_imu_output=True)
constants = model.getConstants()

print("===================================================================")
print("Parameters:")
pprint.pprint(model_params)
print("===================================================================")

NX = model.NX()
NZ = model.NZ()
NP = model.NP()
NY = model.NY()

print("NX = " + str(NX))
print("NZ = " + str(NZ))
print("NP = " + str(NP))
print("NY = " + str(NY))

""" ================================================================================================================ """

# Load data
print("Loading dataset " + dataset_path)
data = pd.read_csv(dataset_path, header=0, parse_dates=[0])

# Make sure we have actual data
N = len(data)
if N == 0:
    print(bcolors.FAIL + "FAIL! N = 0. Does the dataset contain samples?" + bcolors.ENDC)
    quit(0)

# Get mean value from series
u_data = data['VE2_KIN4_SET_0'].mean()
y_elevation_mean = data['VE1_SCAPULA_ELEVATION_0'].mean()
y_rotation_mean = data['VE1_SCAPULA_ROTATION_0'].mean()
y_acc_x_mean = data['VE2_MPU_ACC_0'].mean()
y_acc_y_mean = data['VE2_MPU_ACC_1'].mean()
y_acc_z_mean = data['VE2_MPU_ACC_2'].mean()
y_gyr_x_mean = data['VE2_MPU_GYRO_0'].mean()
y_gyr_y_mean = data['VE2_MPU_GYRO_1'].mean()
y_gyr_z_mean = data['VE2_MPU_GYRO_2'].mean()
y_mean_data = cs.vertcat(
    y_elevation_mean, y_rotation_mean, 
    y_acc_x_mean, y_acc_y_mean, y_acc_z_mean,
    y_gyr_x_mean, y_gyr_y_mean, y_gyr_z_mean,
)

# Get variance values from series
y_elevation_var = data['VE1_SCAPULA_ELEVATION_0'].var()
y_rotation_var = data['VE1_SCAPULA_ROTATION_0'].var()
y_acc_x_var = data['VE2_MPU_ACC_0'].var()
y_acc_y_var = data['VE2_MPU_ACC_1'].var()
y_acc_z_var = data['VE2_MPU_ACC_2'].var()
y_gyr_x_var = data['VE2_MPU_GYRO_0'].var()
y_gyr_y_var = data['VE2_MPU_GYRO_1'].var()
y_gyr_z_var = data['VE2_MPU_GYRO_2'].var()
y_covar_data = cs.diag(cs.vertcat(
    y_elevation_var, y_rotation_var,
    y_acc_x_var, y_acc_y_var, y_acc_z_var,
    y_gyr_x_var, y_gyr_y_var, y_gyr_z_var
))

# Get model parameter guess
p0_mean_data = model.p0()
p0_covar_data = 1e0 * cs.DM.eye(model.NP())

"""
print("Steady state information: ===")
print("control: " + str(u))
print("Roll: " + str(roll) + " rad (" + str(roll*360/(2*np.pi)) + "°)")
print("Pitch: " + str(pitch) + " rad (" + str(pitch*360/(2*np.pi)) + "°)")
print("State: " + str(x_ss))
print("State derivative: " + str(xdot_ss))
print("Accelerometer: " + str(y_acc_ss))
print("Gyroscope: " + str(y_gyr_ss))
print("=============================")
"""

# Create cholesky decomposition
DUMMY = cs.SX.sym('DUMMY', NP+NY, NP+NY)
cholesky = cs.Function('chol', [DUMMY], [cs.chol(DUMMY)])

# Create the NLP
ident = ParametricNLP('steady_state_identificator')
ident.add_decision_var('x', (NX,1))
ident.add_decision_var('z', (NZ,1))
ident.add_decision_var('p', (NP,1))
ident.add_parameter('u', (1,1))
ident.add_parameter('p0_mean', (NP,1))
ident.add_parameter('p0_conf', (NP,NP))
ident.add_parameter('y_mean', (NY,1))
ident.add_parameter('y_conf', (NY,NY))
ident.bake_variables()
x = ident.get_decision_var('x')
z = ident.get_decision_var('z')
p = ident.get_decision_var('p')
u = ident.get_parameter('u')
p0_mean = ident.get_parameter('p0_mean')
p0_conf = ident.get_parameter('p0_conf')
y_mean = ident.get_parameter('y_mean')
y_conf = ident.get_parameter('y_conf')

# Create residual terms
p_residual = p0_mean - p
y_residual = y_mean - model.out_aug_fun(x,z,u,p)
residual = cs.vertcat(p_residual, y_residual)

# Create residual weights
weights = cs.MX.zeros((NP+NY,NP+NY))
weights[:NP,:NP] = p0_conf
weights[NP:,NP:] = y_conf
weights_sqrt = cholesky(weights)

# Weight residuals and set cost
weighted_residual = cs.mtimes(weights_sqrt, residual)
COST = 0.5 * cs.mtimes(weighted_residual.T, weighted_residual)
ident.set_cost(COST)

# Set constraints:
ident.add_equality('0 = ode(x,z,u,p)', model.ode_aug_fun(x,z,u,p))
ident.add_equality('0 = alg(x,z,u,p)', model.alg_aug_fun(x,z,u,p))

# Initialize nlp
ident.init(
    nlpsolver='ipopt',
    opts={
        'verbose': True,
        'expand': True,
        'jit': False,
        'ipopt.print_level': 5,
        'print_time': 1,
        'ipopt.print_timing_statistics': 'yes',
        'ipopt.sb': 'no',
        'ipopt.max_iter': 2000,
        'verbose_init': True,
    },
    create_analysis_functors = False,
    compile_solver = False
)

# Create functors
w_sym = ident.struct_w
p_sym = ident.struct_p
R_fun = cs.Function('R', [w_sym, p_sym], [weighted_residual])
jac_R_fun = cs.Function('jac_R', [w_sym, p_sym], [cs.jacobian(weighted_residual, w_sym)])
print(jac_R_fun(w_sym,p_sym).shape)

g = ident.nlp['g'][:NX+NZ]
jac_g_fun = cs.Function('jac_g', [w_sym, p_sym], [cs.jacobian(g,w_sym)])
print(jac_g_fun(w_sym,p_sym).shape)

M_fun = cs.Function('M', [w_sym, p_sym], [
    cs.blockcat([
        [cs.mtimes(jac_R_fun(w_sym,p_sym).T, jac_R_fun(w_sym,p_sym)), jac_g_fun(w_sym,p_sym).T],
        [jac_g_fun(w_sym,p_sym), cs.DM.zeros((8,8)) ]
    ])
])

# Print info string
print(ident)

# Create initial guess for z
x_data = model.x0()
z_sym = cs.MX.sym('z', model.NZ(), 1)
fun = cs.Function('g', [z_sym], [model.alg_aug_fun(x_data,z_sym,u_data,p0_mean_data)])
rootfinder = cs.rootfinder('root', 'newton', fun)
z_data = rootfinder(cs.DM.zeros((model.NZ(),1)))  

# Create guess and parameters
guess = ident.struct_w(0)
guess['x'] = x_data
guess['z'] = z_data
guess['p'] = p0_mean_data
param = ident.struct_p(0)
param['u'] = u_data
param['p0_mean'] = p0_mean_data
param['p0_conf'] = cs.inv(p0_covar_data)
param['y_mean'] = y_mean_data
param['y_conf'] = cs.inv(y_covar_data)

# Solve
sol, stats, sol_orig, lb, ub = ident.solve(guess,param)
w_sol = sol['w']
p_sol = param
print("Solution:")
for key in w_sol.keys():
  print(key + " = " + str(w_sol[key]))

# Compute covariance of result, second try
# https://support.sas.com/documentation/cdl/en/statug/63962/HTML/default/viewer.htm#statug_nlin_sect027.htm
M = NX + NZ
"""
A = dgdp_fun(w_sol, p_sol).T.full()
B = hess_f_fun(w_sol, p_sol).full()
Q, R = linalg.qr(A)
Z = Q[:,NX+NZ:]
inner = cs.mtimes([Z.transpose(), B, Z])
outer = cs.mtimes([Z, linalg.inv(inner), Z.transpose()])
p_covar_diag = cs.diag(outer)[NX+NZ:]
print(p_covar_diag)
"""

# Compute rank of jacobians to check condition
print("n = " + str(w_sol.shape[0]))
print("m2 = " + str(g.shape[0]))
J1 = jac_R_fun(w_sol, p_sol)
J2 = jac_g_fun(w_sol, p_sol)
rank_J2 = np.linalg.matrix_rank(J2)
print("rank J1 = " + str(rank_J2))
rank_J = np.linalg.matrix_rank(cs.vertcat(J1,J2))
print("rank J = " + str(rank_J))

# Compute covariance matrix of result
M = cs.inv(M_fun(w_sol,p_sol))
print("M shape = " + str(M.shape))
C = M[:NX+NZ+NP,:NX+NZ+NP]
print("C shape = " + str(C.shape))
print(C)
Sigma = C[NX+NZ:, NX+NZ:]
p_covar = Sigma
print("Parameter covariance: " + str(Sigma))
print("Diagonals: " + str(cs.diag(Sigma)))

eigvals, eigvecs = linalg.eig(C)
print(eigvals)
#quit(0)

# Put parameters and its variances in a dict
idx = 0
identified_param = {}
identified_p = sol['w']['p']
identified_param_vars = {}
identified_p_vars = cs.diag(p_covar)
#identified_p_vars = cs.diag(p_covar)
#identified_p_vars = cs.mtimes(R.T, R) * p_covar_diag
for k in range(len(model.dae.p)):
    key = str(model.dae.p[k])
    shape = model.dae.p[k].shape
    len = np.prod(shape)
    identified_param[key] = identified_p[idx:idx+len].reshape(shape).full().squeeze().tolist()
    identified_param_vars[key] = identified_p_vars[idx:idx+len].reshape(shape).full().squeeze().tolist()
    idx += len


# Print out identified params
print("===============================================")
print("Identified parameters:")
for key in model_params.keys():
    info  = key + ": \n\tOLD:" + str(model_params[key]) 
    info +=       "  \n\tNEW:" + str(identified_param[key])
    info +=       " (variance = " + str(identified_param_vars[key]) + ")"
    print(info)

# Write identified parameters to json file
print("===============================================")
print("Writing identified parameters to", param_file_out)
with open(param_file_out, 'w') as outfile:
  json.dump(identified_param, outfile, indent=4)

# Write variances of parameters to other json file
print("===============================================")
print("Writing identified parameters to", param_vars_file_out)
with open(param_vars_file_out, 'w') as outfile:
  json.dump(identified_param_vars, outfile, indent=4)