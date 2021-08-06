from thesis_code.model import CarouselModel
from thesis_code.identificator import CarouselIdentificator
import json
from copy import deepcopy
from thesis_code.utils.bcolors import bcolors
import casadi as cs
import numpy as np
import pandas as pd
import time

fit_imu = True

#dataset_path = "./data_phys/CAROUSEL_ZCMLOG_2019_09_13_STEADY_STATE_CONTROL.zcmlog.PREPROCESSED.csv"
dataset_path = "data_phys/CAROUSEL_ZCMLOG_2019_09_13_RECT_CONTROL.zcmlog.PREPROCESSED.csv"


param_file_in = "params_identified_ss.json"
#param_file_in = "./params_identified_ss_new.json"
param_variances_file_in = "params_identified_ss_variances.json"
#param_file_in = "./params_identified_dynamic_imu.json"
param_file_out = "./params_identified_dynamic_imu_new.json"
param_vars_file_out = "params_identified_dynamic_imu_variances.json"

# Create a model
param = {}
print("Loading parameter set " + param_file_in)
with open(param_file_in, 'r') as file:
    param = json.load(file)
model = CarouselModel(param, with_angle_output=True, with_imu_output=fit_imu)
constants = CarouselModel.getConstants()

# Get parameter vairances
param_variances = {}
print("Loading parameter variance set " + param_variances_file_in)
with open(param_variances_file_in, 'r') as file:
    param_variances = json.load(file)

# Load data
print("Loading dataset " + dataset_path)
data = pd.read_csv(dataset_path, header=0, parse_dates=[0])

# Make sure data exists
N = len(data)
if N == 0:
    print(bcolors.FAIL + "FAIL! N = 0. Does the dataset contain samples?" + bcolors.ENDC)
    quit(0)

# Set horizon size
N = 2
N = 100
N = min([N,len(data)])

# Get mean value from series
Us = data['VE2_KIN4_SET_0'].head(N)
y_roll = data['VE1_SCAPULA_ELEVATION_0'].head(N)
y_pitch = data['VE1_SCAPULA_ROTATION_0'].head(N)
y_acc_0 = data['VE2_MPU_ACC_0'].head(N)
y_acc_1 = data['VE2_MPU_ACC_1'].head(N)
y_acc_2 = data['VE2_MPU_ACC_2'].head(N)
y_gyr_0 = data['VE2_MPU_GYRO_0'].head(N)
y_gyr_1 = data['VE2_MPU_GYRO_1'].head(N)
y_gyr_2 = data['VE2_MPU_GYRO_2'].head(N)

# Get the actual states from the measurements
roll = constants['roll_sensor_offset'] - y_roll
pitch = constants['pitch_sensor_offset'] - y_pitch

# Set sampling time (20 Hz)
dt = 1. / 20.

# Set initial state, derivatives computed by forward differences
x0 = cs.DM([roll[0], pitch[0], (roll[1]-roll[0])/dt, (pitch[1]-pitch[0])/dt, 0.5])
print("x0 = " + str(x0))

# Set controls
Us = cs.DM(Us).T

# Set measurements
if fit_imu:
    Ys = cs.horzcat(y_roll, y_pitch, y_acc_0, y_acc_1, y_acc_2, y_gyr_0, y_gyr_1, y_gyr_2).T
else:
    Ys = cs.horzcat(y_roll, y_pitch).T


# Choose initial state confidence matrix
init_confidence = {}
init_confidence['roll'] = 1e3
init_confidence['pitch'] = 1e3
init_confidence['roll_rate'] = 1e2
init_confidence['pitch_rate'] = 1e2
init_confidence['elevator_deflection'] = 1e3
init_confidence_vec = np.vstack(np.concatenate([np.array(val).flatten() for val in init_confidence.values()]))
Q0 = cs.DM(init_confidence_vec)

# Choose model confidence matrix <- Penalize much more so gaps are closed
model_confidence = {}
model_confidence['roll'] = 1e4
model_confidence['pitch'] = 1e4
model_confidence['roll_rate'] = 1e4
model_confidence['pitch_rate'] = 1e4
model_confidence['elevator_deflection'] = 1e4
model_confidence_vec = np.vstack(np.concatenate([np.array(val).flatten() for val in model_confidence.values()]))
Q = cs.DM(model_confidence_vec)

# Choose measurement confidence matrix
sensor_confidence = {}
sensor_confidence['roll'] = 1e2
sensor_confidence['pitch'] = 1e2
if fit_imu:
  acc_conf = 1e-1
  sensor_confidence['acc_x'] = acc_conf
  sensor_confidence['acc_y'] = acc_conf
  sensor_confidence['acc_z'] = acc_conf
  gyr_conf = 1e-1
  sensor_confidence['gyr_x'] = gyr_conf
  sensor_confidence['gyr_y'] = gyr_conf
  sensor_confidence['gyr_z'] = gyr_conf

sensor_confidence_vec = np.vstack(np.concatenate([np.array(val).flatten() for val in sensor_confidence.values()]))
R = cs.DM(sensor_confidence_vec)

# Choose prior-guess confidence matrix
param_confidence = deepcopy(param)
for key in param_confidence.keys():
  oldval = np.array(param_confidence[key])
  newval = (1e1 * np.ones(oldval.shape)).flatten().tolist()
  if fit_imu:
    newval = (1e1 * np.ones(oldval.shape)).flatten().tolist()
  param_confidence[key] = newval

if fit_imu:
  param_confidence['pos_imu_wrt_4'] = [1e0, 1e0, 1e0]
  param_confidence['rot_imu_wrt_4'] = [1e-1, 1e0, 1e0]
#else:
param_confidence['Ixx_COM'] = 1e-1
param_confidence['Iyy_COM'] = 1e-1
param_confidence['Izz_COM'] = 1e0
param_confidence['pos_center_of_mass_wrt_4'] = [1e-1, 1e0, 1e0]
param_confidence['pos_aileron_aerodynamic_center_wrt_4'] = [1e0, 1e0, 1e0]
param_confidence['pos_elevator_aerodynamic_center_wrt_4'] = [1e0, 1e0, 1e0]
param_confidence['C_LA_0'] = 1e0      # The coefficients tend to be larger
param_confidence['C_LA_max'] = 1e0    # in magnitude than the other parameters
param_confidence['C_DA_0'] = 1e0      # so we allow for larger deviation to 
param_confidence['C_DA_max'] = 1e0    # bring them on par
param_confidence['C_LE_0'] = 1e0
param_confidence['C_LE_max'] = 1e0
param_confidence['C_DE_0'] = 1e0
param_confidence['C_DE_max'] = 1e0
param_confidence['elevator_deflector_gain'] = 1e-1
param_confidence['elevator_deflector_tau'] = 1e3
param_confidence['mu_phi'] = 1e0
param_confidence['mu_theta'] = 1e0

assert set(param) == set(param_confidence), "Confidence-dictionary has to have the same entries as param!"
param_confidence_vec = np.vstack(np.concatenate([np.array(val).flatten() for val in param_confidence.values()]))
S = cs.DM(param_confidence_vec)

# Create an identificator
time_start = time.time()
print("Time start = " + str(time_start))
ident = CarouselIdentificator(model, N, dt, verbose=True, do_compile=False, expand=True, fit_imu=fit_imu)
time_end = time.time()
print("Time end = " + str(time_end))
print("Duration = "+ str(time_end - time_start) + "s")

# Initialize the identificator with the data
print("x0 = " + str(x0))
ident.init(x0 = x0, U = Us, Y = Ys, Q = Q, Q0 = Q0, R = R, S = S)
print(ident.ocp)

# Identify parameters
p_sol, x_sol, z_sol, result, stats = ident.call()
print("p_sol = " + str(p_sol))
print("x_sol = " + str(x_sol))
print("z_sol = " + str(z_sol))


covmat = ident.cov_fun(result['w'],result['p'])
print("Covariance matrix: " + str(covmat))
print("Diagonals: " + str(cs.diag(covmat)))

identified_p_vars = cs.diag(covmat)[ident.ocp.num_decision_vars - ident.NP:]

if False:
  print("\n\EIGENVALUE SHIT\n\n")
  w = result['w']
  lam_g = result['lam_g']
  eig = ident.ocp.eval_expanded_eigvecs(w,lam_g,ident.parameters)
  for elem in eig:
    print("===========================================================================")
    print("The eigenvalue shows the amount of curvature in the direction of the eigenvector.")
    print("Each eigenvalue-eigenvector pair is one effective degree of freedom in the NLP.")
    print("\neigvec_exp = Z * eigvec(Z.T * hess(lag) * Z), with Z = nullspace(jac(g))\n")
    print("Eigenvalue: ", elem[0])
    print("Eigenvector (expanded into decision space): \n")
    for key in elem[1].keys():
      print(key, " = ", elem[1][key])
    print("\n\n")


print("Identified parameters:")
idx = 0
identified_param = {}
identified_param_vars = {}
for k in range(len(model.dae.params)):
    key = str(model.dae.params[k])
    shape = model.dae.params[k].shape
    len = np.prod(shape)
    identified_param[key] = p_sol[idx:idx+len].reshape(shape).full().squeeze().tolist()
    identified_param_vars[key] = identified_p_vars[idx:idx+len].reshape(shape).full().squeeze().tolist()
    idx += len

for key in param.keys():
    #print(key + ": \n\tOLD:" + str(param[key]) + "\n\tNEW:" + str(identified_param[key]))
    info  = key + ": \n\tOLD:" + str(param[key]) 
    info +=       "  \n\tNEW:" + str(identified_param[key])
    info +=       " (variance = " + str(identified_param_vars[key]) + ")"
    print(info)
    pass

# Compute 'estimated' measurements
y_sol = cs.horzcat(*[ ident.model.out_aug_fun(x_sol[:,k], z_sol[:,k], Us[:,k], p_sol) for k in range(N) ])

# Store states and measurements in a dataframe
x_sol = pd.DataFrame(
  data=x_sol.T.full(), 
  index=data['timestamp'][:N+1], 
  columns=['roll','pitch','roll_rate','pitch_rate','elevator_deflection']
)
y_cols = ['roll','pitch','acc_0','acc_1','acc_2','gyr_0','gyr_1','gyr_2'] if fit_imu else ['roll','pitch']
y_sol = pd.DataFrame(
  data=y_sol.T.full(), 
  index=data['timestamp'][:N], 
  columns=y_cols
)

if stats['return_status'] == "Solve_Succeeded":
  # Write identified parameters to json file
  print("Writing identified parameters to", param_file_out)
  with open(param_file_out, 'w') as outfile:
    json.dump(identified_param, outfile, indent=4)

  # Write variances of parameters to other json file
  print("Writing identified variances to", param_vars_file_out)
  with open(param_vars_file_out, 'w') as outfile:
    json.dump(identified_param_vars, outfile, indent=4)

  # Write estimated state-trajectory to json file
  filename_estimated_states = dataset_path + ".ESTIMATED_STATES.csv"
  print("Written estimated states into file " + filename_estimated_states)
  x_sol.to_csv(filename_estimated_states)

  # Write estimated measurement-trajectory to json file
  filename_estimated_measurements = dataset_path + ".ESTIMATED_MEASUREMENTS.csv"
  print("Written estimated states into file " + filename_estimated_measurements)
  y_sol.to_csv(filename_estimated_measurements)

else:
  print("Identifikator returned " + str(stats['return_status'] + ". Did not write parameters to files."))
  quit(0)