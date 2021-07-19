from thesis_code.models.carousel_whitebox import CarouselWhiteBoxModel
from thesis_code.carousel_simulator import Carousel_Simulator
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import casadi as cs
from matplotlib.backends.backend_pdf import PdfPages

validate_original = False

""" ================================================================================================================ """

# Load original parameters
original_param = {}
in_original_param_filename = "params_identified_dynamic_imu.json"
print("Loading identified parameter set " + in_original_param_filename)
with open(in_original_param_filename, 'r') as file:
    original_param = json.load(file)

# Load identified parameters
identified_param = {}
in_identified_param_filename = "./params_identified_dynamic_imu_new.json"
#in_identified_param_filename = "./params_identified_dynamic_imu.json"
print("Loading identified parameter set " + in_identified_param_filename)
with open(in_identified_param_filename, 'r') as file:
    identified_param = json.load(file)

""" ================================================================================================================ """

# Load all validation sets (preprocessed trajectories)
#dataset_names = ['STEADY_STATE', 'RECT', 'SINE']
dataset_names = ['SINE']
datasets = {}

for name in dataset_names:
  # Set path
  path = "./data_phys/CAROUSEL_ZCMLOG_2019_09_13_"+name+"_CONTROL.zcmlog.PREPROCESSED.csv"
  # Load dataset
  print("Loading validation dataset: " + path)
  datasets[name] = pd.read_csv(path, header=0, parse_dates=[0])
  # Check dataset content
  if len(datasets[name]) == 0:
      print(bcolors.FAIL + "FAIL! N = 0. Does dataset " + name + " contain samples?" + bcolors.ENDC)
      quit(0)

""" ================================================================================================================ """

# Create models using the original and identified parameters
kwargs = {'with_angle_output':False, 'with_imu_output':True}

original_model = CarouselWhiteBoxModel(original_param, *kwargs.values())
identified_model = CarouselWhiteBoxModel(identified_param, *kwargs.values())
constants = CarouselWhiteBoxModel.getConstants()

# Simulation data and convenience functions
dt = 0.05
#dt = 1. / 2.
get_roll = lambda yr:  constants['roll_sensor_offset'] - yr
get_pitch = lambda yp: constants['pitch_sensor_offset'] - yp

# This is the data we will compare later
Us_columns = ['TRIM_TAB_ANGLE']
Xs_columns = ['VE1_SCAPULA_ELEVATION_0', 'VE1_SCAPULA_ROTATION_0']
Ys_columns = ['VE2_MPU_ACC_0', 'VE2_MPU_ACC_1', 'VE2_MPU_ACC_2', 
              'VE2_MPU_GYRO_0', 'VE2_MPU_GYRO_1', 'VE2_MPU_GYRO_2']

# Do validation for each of the datasets
result_real = {}
result_original = {}
result_identified = {}
N_margin = 3000
for name in dataset_names:
  # Fetch dataaset
  data = datasets[name]

  # Set simulation horizon
  N = min([N_margin+1500,len(data)])
  print("N = " +str(N))

  # Fetch controls
  controls = data['VE2_KIN4_SET_0'].head(N)

  # Set initial state (derivatives computed via central differences)
  roll_0 = get_roll(data['VE1_SCAPULA_ELEVATION_0'][0])
  pitch_0 = get_pitch(data['VE1_SCAPULA_ROTATION_0'][0])
  roll_1 = get_roll(data['VE1_SCAPULA_ELEVATION_0'][1])
  pitch_1 = get_pitch(data['VE1_SCAPULA_ROTATION_0'][1])
  roll_2 = get_roll(data['VE1_SCAPULA_ELEVATION_0'][2])
  pitch_2 = get_pitch(data['VE1_SCAPULA_ROTATION_0'][2])
  x0 = cs.DM([
    roll_1, pitch_1,
    (roll_2-roll_0)/(2*dt), (pitch_2-pitch_0)/(2*dt),
    controls[1]]
  )

  # Set up the simulator
  original_simulator = Carousel_Simulator(model = original_model, x0 = x0)
  identified_simulator = Carousel_Simulator(model = identified_model, x0 = x0)

  # Simulate using the excitation signal from the validation dataset
  Xs_original_sim = [x0.full()]
  Ys_original_sim = []
  Xs_identified_sim = [x0.full()]
  Ys_identified_sim = []
  print("Simulating " + name + "..")
  original_simulator_crashed = False
  identified_simulator_crashed = False
  max_sq_val = 1000
  for k in range(1,N):
    if validate_original:
      if not original_simulator_crashed:
        xf_k, zf_k, y0_k = original_simulator.simstep(controls[k], dt)
        Xs_original_sim += [xf_k.full()]
        Ys_original_sim += [y0_k.full()]
        if cs.mtimes(xf_k.T, xf_k) > max_sq_val:
          original_simulator_crashed = True
          print("Original model crashed!")
   
    if not identified_simulator_crashed:
      xf_k, zf_k, y0_k = identified_simulator.simstep(controls[k], dt)
      Xs_identified_sim += [xf_k.full()]
      Ys_identified_sim += [y0_k.full()]
      if cs.mtimes(xf_k.T, xf_k) > max_sq_val:
        identified_simulator_crashed = True
        print("Identified model crashed!")

    if original_simulator_crashed and identified_simulator_crashed:
      print("Crashed at " + str(k))
      break

  print("Simulation done.")
  
  # Fetch real data
  print("Fetching real data..")
  Xs_real = data[Xs_columns].head(len(data[Xs_columns]))
  Ys_real = data[Ys_columns].head(len(data[Ys_columns]))
  Xs_real['VE1_SCAPULA_ELEVATION_0'] = Xs_real['VE1_SCAPULA_ELEVATION_0'].apply(get_roll)
  Xs_real['VE1_SCAPULA_ROTATION_0'] = Xs_real['VE1_SCAPULA_ROTATION_0'].apply(get_pitch)
  # Store data
  result_real[name] = {
    'Us': controls,
    'Xs': Xs_real,
    'Ys': Ys_real
  }

  """
  if validate_original:
    print("Computing MSE for original parameters..")
    # Convert to numpy arrays
    Us_original_sim = np.array(Xs_original_sim).squeeze()[:,4]
    Xs_original_sim = np.array(Xs_original_sim).squeeze()[:,:2]
    Ys_original_sim = np.array(Ys_original_sim).squeeze()
    # Put simulated states and outputs in dataframes  
    Us_original_sim = pd.DataFrame(Us_original_sim.squeeze(), columns=Us_columns)
    Xs_original_sim = pd.DataFrame(Xs_original_sim.squeeze(), columns=Xs_columns)
    Ys_original_sim = pd.DataFrame(Ys_original_sim.squeeze(), columns=Ys_columns)
    # Compute errors
    size = len(Xs_original_sim)
    x_err_original = Xs_real[:size] - Xs_original_sim
    y_err_original = Ys_real[:size] - Ys_original_sim
    # Compute MSE (original parameters)
    err_sum_original = x_err_original.sum(axis='columns') + y_err_original.sum(axis='columns')
    mse_original = (err_sum_original ** 2).mean(axis='index')
    # Store data
    result_original[name] = {
      'Us': Us_original_sim, # actually tab angles
      'Xs': Xs_original_sim,
      'Ys': Ys_original_sim,
      'x_abs_err': x_err_original.abs(),
      'y_abs_err': y_err_original.abs(),
      'mse': mse_original
    }
  """

  print("Computing MSE for identified parameters..")
  # Convert to numpy arrays
  Us_identified_sim = np.array(Xs_identified_sim)[:,4]
  Xs_identified_sim = np.array(Xs_identified_sim)[:,:2]
  Ys_identified_sim = np.array(Ys_identified_sim)
  
  # Put simulated states and outputs in dataframes
  Us_identified_sim = pd.DataFrame(Us_identified_sim.squeeze(), columns=Us_columns)
  Xs_identified_sim = pd.DataFrame(Xs_identified_sim.squeeze(), columns=Xs_columns)
  Ys_identified_sim = pd.DataFrame(Ys_identified_sim.squeeze(), columns=Ys_columns)
  # Compute errors
  size = len(Xs_identified_sim)
  print("Xs_identified_sim size = " + str(size))
  x_err_identified = Xs_real[:size] - Xs_identified_sim
  y_err_identified = Ys_real[:size] - Ys_identified_sim
  # Compute MSE (identified parameters)
  err_sum_identified = x_err_identified.sum(axis='columns') + y_err_identified.sum(axis='columns')
  mse_identified = (err_sum_identified ** 2).mean(axis='index')
  # Store data
  result_identified[name] = {
    'Us': Us_identified_sim, # actually tab angles
    'Xs': Xs_identified_sim,
    'Ys': Ys_identified_sim,
    'x_abs_err': x_err_identified.abs(),
    'y_abs_err': y_err_identified.abs(),
    'mse': mse_identified
  }

print("All done.")


def toDeg(rad):
    return rad * 360 / (2*np.pi)


# Plot states
if True:
  print("Plotting results..")
  
  # Create one figure for each dataset
  n_cases = len(dataset_names)
  N_valid = 1e10
  N_valid = min([N_valid, max([len(Xs_original_sim), len(Xs_identified_sim)])])
  N_valid = min([int(30./dt), N_valid])

  for i in range(n_cases):

    # Fetch dataset
    key = dataset_names[i]  
    data_real = result_real[key]
    data_iden = result_identified[key]

    print("Data (REAL) size (Xs) = " + str(len(data_real['Xs'])))
    print("Data (IDEN) size (Xs) = " + str(len(data_iden['Xs'])))

    print("N_margin = " + str(N_margin))
    print("N_valid = " + str(N_valid))

    for subkey in data_real.keys():
      data_real[subkey] = data_real[subkey][N_margin:N_margin+N_valid]
      data_iden[subkey] = data_iden[subkey][N_margin:N_margin+N_valid]

    print("Data (REAL) size (Xs) after = " + str(len(data_real['Xs'])))
    print("Data (IDEN) size (Xs) after = " + str(len(data_iden['Xs'])))

    data_real['Xs'] = data_real['Xs']#.head(N_valid)
    #data_real['Xs']['ELEVATION'] = data_real['Xs']['VE1_SCAPULA_ELEVATION_0'].apply(toDeg)
    #data_real['Xs']['PITCH'] = data_real['Xs']['VE1_SCAPULA_ROTATION_0'].apply(toDeg)
    data_iden['Xs'] = data_iden['Xs']#.head(N_valid)
    #data_iden['Xs']['ELEVATION'] = data_iden['Xs']['VE1_SCAPULA_ELEVATION_0'].apply(toDeg)
    #data_iden['Xs']['PITCH'] = data_iden['Xs']['VE1_SCAPULA_ROTATION_0'].apply(toDeg)
    data_real['Ys'] = data_real['Ys']#.head(N_valid)
    data_iden['Ys'] = data_iden['Ys']#.head(N_valid)

    filename_out = "result_ident_" + key + ".pdf"
    with PdfPages("../../tex/thesis/figures/identification/" + filename_out) as pdf:

      # Create a shared x-axis
      tAxis = dt * np.arange(N_valid)

      # Create the figure
      #fig, ax = plt.subplots(1,1,figsize=(5,5))

      #mse_iden_string = 'MSE = ' + "{0:.3f}".format(data_iden['mse'])
      #plt.annotate('Dataset ' + key + '\n' + mse_orig_string + '\n' + mse_iden_string )

      # Pens
      real_args = {}#{'linewidth':3, 'linestyle':'-'}
      sim_args = {}#{'linewidth':2, 'linestyle':'--'}

      xsize = 10 # inches
      annot_xloc = 0.65
      annot_yloc = 0.05

      # Plot dataset's controls
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, data_real['Us'])
      plt.ylabel('Control $u(t)$')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()

      # Plot dataset's elevation
      real = data_real['Xs']['VE1_SCAPULA_ELEVATION_0'].apply(toDeg)
      iden = data_iden['Xs']['VE1_SCAPULA_ELEVATION_0'].apply(toDeg)
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, label="Real")
      plt.plot(tAxis, iden, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Elevation $\phi(t)$ [deg]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()
      
      # Plot dataset's rotation
      real = data_real['Xs']['VE1_SCAPULA_ROTATION_0'].apply(toDeg)
      iden = data_iden['Xs']['VE1_SCAPULA_ROTATION_0'].apply(toDeg)
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, label="Real")
      plt.plot(tAxis, iden, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Rotation $\\theta(t)$ [deg]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()


      # Plot dataset's accelerometer output
      real = data_real['Ys']['VE2_MPU_ACC_0']
      iden = data_iden['Ys']['VE2_MPU_ACC_0']
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, **real_args, label="Real")
      plt.plot(tAxis, iden, **sim_args, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Accelerometer x $a_x(t)$ [m$\cdot$s${}^{-2}$]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()

      real = data_real['Ys']['VE2_MPU_ACC_1']
      iden = data_iden['Ys']['VE2_MPU_ACC_1']
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, **real_args, label="Real")
      plt.plot(tAxis, iden, **sim_args, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Accelerometer y $a_y(t)$ [m$\cdot$s${}^{-2}$]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()

      real = data_real['Ys']['VE2_MPU_ACC_2']
      iden = data_iden['Ys']['VE2_MPU_ACC_2']
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, **real_args, label="Real")
      plt.plot(tAxis, iden, **sim_args, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Accelerometer z $a_z(t)$ [m$\cdot$s${}^{-2}$]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()

      # Plot dataset's gyroscope output
      real = data_real['Ys']['VE2_MPU_GYRO_0']
      iden = data_iden['Ys']['VE2_MPU_GYRO_0']
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, **real_args, label="Real")
      plt.plot(tAxis, iden, **sim_args, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Gyroscope x $\omega_x(t)$ [m$\cdot$s${}^{-2}$]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()
      
      real = data_real['Ys']['VE2_MPU_GYRO_1']
      iden = data_iden['Ys']['VE2_MPU_GYRO_1']
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, **real_args, label="Real")
      plt.plot(tAxis, iden, **sim_args, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Gyroscope y $\omega_y(t)$ [m$\cdot$s${}^{-2}$]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()
      
      real = data_real['Ys']['VE2_MPU_GYRO_2']
      iden = data_iden['Ys']['VE2_MPU_GYRO_2']
      err = (real - iden)
      mse = (err ** 2).mean()
      fig, ax = plt.subplots(1,1,figsize=(xsize,5))
      plt.plot(tAxis, real, **real_args, label="Real")
      plt.plot(tAxis, iden, **sim_args, label="Simulated")
      plt.annotate("MSE = " + "{0:.3e}".format(mse), xy=(annot_xloc, annot_yloc), xycoords="axes fraction", fontsize=16)
      plt.ylabel('Gyroscope z $\omega_z(t)$ [m$\cdot$s${}^{-2}$]')
      plt.xlabel('Time $t$ [s]')
      pdf.savefig()

      plt.close()
      exit(0)


# Plot measurements
if False:
  print("Plotting results..")
  
  # Create a plot. One column for each case
  n_cases = len(dataset_names)
  fig, ax = plt.subplots(5,n_cases,sharex='all', sharey='row')

  plt.sca(ax[0,0])
  plt.ylabel('Controls')
  plt.sca(ax[1,0])
  plt.ylabel('Acceleration')
  plt.sca(ax[2,0])
  plt.ylabel('Acceleration SQE')
  plt.sca(ax[3,0])
  plt.ylabel('Gyroscope')
  plt.sca(ax[4,0])
  plt.ylabel('Gyroscope SQE')

  # Fill each column
  for i in range(n_cases):
    key = dataset_names[i]
    data_real = result_real[key]
    #data_orig = result_original[key]
    data_iden = result_identified[key]

    print("i="+str(i))
    xacc_real = data_real['Ys']['VE2_MPU_ACC_0'].mean()
    yacc_real = data_real['Ys']['VE2_MPU_ACC_1'].mean()
    zacc_real = data_real['Ys']['VE2_MPU_ACC_2'].mean()

    xacc_iden = data_iden['Ys']['VE2_MPU_ACC_0'].mean()
    yacc_iden = data_iden['Ys']['VE2_MPU_ACC_1'].mean()
    zacc_iden = data_iden['Ys']['VE2_MPU_ACC_2'].mean()

    xacc_diff = xacc_real - xacc_iden
    yacc_diff = yacc_real - yacc_iden
    zacc_diff = zacc_real - zacc_iden

    xacc_iden_proz_err = xacc_diff / xacc_iden
    yacc_iden_proz_err = yacc_diff / yacc_iden
    zacc_iden_proz_err = zacc_diff / zacc_iden
    
    xacc_real_proz_err = xacc_diff / xacc_real
    yacc_real_proz_err = yacc_diff / yacc_real
    zacc_real_proz_err = zacc_diff / zacc_real

    print("x_iden: " + str(xacc_iden_proz_err))
    print("y_iden: " + str(yacc_iden_proz_err))
    print("z_iden: " + str(zacc_iden_proz_err))

    print("x_real: " + str(xacc_real_proz_err))
    print("y_real: " + str(yacc_real_proz_err))
    print("z_real: " + str(zacc_real_proz_err))

    # Plot dataset's controls
    plt.sca(ax[0, i])
    #mse_orig_string = 'MSE (orig) = ' + "{0:.3f}".format(data_orig['mse'])
    mse_iden_string = 'MSE (iden) = ' + "{0:.3f}".format(data_iden['mse'])
    #plt.title('Dataset ' + key + '\n' + mse_orig_string + '\n' + mse_iden_string )
    plt.plot(data_iden['Us'].head(N_valid), label='Simulated (identified parameters)')
    plt.plot(data_real['Us'].head(N_valid), label='Setpoint')

    # Plot dataset's acceleration
    plt.sca(ax[1, i])
    plt.plot(data_iden['Ys']['VE2_MPU_ACC_0'], label="X: Simulated (identified parameters)")
    plt.plot(data_iden['Ys']['VE2_MPU_ACC_1'], label="Y: Simulated (identified parameters)")
    plt.plot(data_iden['Ys']['VE2_MPU_ACC_2'], label="Z: Simulated (identified parameters)")
    plt.plot(data_real['Ys']['VE2_MPU_ACC_0'], label="X: Real")
    plt.plot(data_real['Ys']['VE2_MPU_ACC_1'], label="Y: Real")
    plt.plot(data_real['Ys']['VE2_MPU_ACC_2'], label="Z: Real")

    # Plot dataset's acceleration error metric
    plt.sca(ax[2, i])
    plt.plot(data_iden['y_abs_err']['VE2_MPU_ACC_0'], label="X: Simulated (identified parameters)")
    plt.plot(data_iden['y_abs_err']['VE2_MPU_ACC_1'], label="Y: Simulated (identified parameters)")
    plt.plot(data_iden['y_abs_err']['VE2_MPU_ACC_2'], label="Z: Simulated (identified parameters)")
    plt.yscale('log')

    # Plot dataset's pitch
    plt.sca(ax[3, i])
    plt.plot(data_iden['Ys']['VE2_MPU_GYRO_0'], label="X: Simulated (identified parameters)")
    plt.plot(data_iden['Ys']['VE2_MPU_GYRO_1'], label="Y: Simulated (identified parameters)")
    plt.plot(data_iden['Ys']['VE2_MPU_GYRO_2'], label="Z: Simulated (identified parameters)")
    plt.plot(data_real['Ys']['VE2_MPU_GYRO_0'], label="X: Real")
    plt.plot(data_real['Ys']['VE2_MPU_GYRO_1'], label="Y: Real")
    plt.plot(data_real['Ys']['VE2_MPU_GYRO_2'], label="Z: Real")


    # Plot dataset's pitch MSE
    plt.sca(ax[4, i])
    plt.plot(data_iden['y_abs_err']['VE2_MPU_GYRO_0'], label="X: Simulated (identified parameters)")
    plt.plot(data_iden['y_abs_err']['VE2_MPU_GYRO_1'], label="Y: Simulated (identified parameters)")
    plt.plot(data_iden['y_abs_err']['VE2_MPU_GYRO_2'], label="Z: Simulated (identified parameters)")
    plt.yscale('log')
  
  plt.sca(ax[0,0])
  plt.legend().draggable()
  plt.sca(ax[1,0])
  plt.legend().draggable()

plt.show()
quit(0)
