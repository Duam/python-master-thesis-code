from thesis_code.model import CarouselModel
from thesis_code.simulator import CarouselSimulator
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import casadi as cs
from matplotlib.backends.backend_pdf import PdfPages

validate_original = False

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

# Load all validation sets (preprocessed trajectories)
#dataset_names = ['STEADY_STATE', 'RECT', 'SINE']
dataset_names = ['SINE']
datasets = {}

for name in dataset_names:
    path = "./data_phys/CAROUSEL_ZCMLOG_2019_09_13_"+name+"_CONTROL.zcmlog.PREPROCESSED.csv"
    print("Loading validation dataset: " + path)
    datasets[name] = pd.read_csv(path, header=0, parse_dates=[0])
    # Check dataset content
    if len(datasets[name]) == 0:
        print(bcolors.FAIL + "FAIL! N = 0. Does dataset " + name + " contain samples?" + bcolors.ENDC)
        quit(0)

# Create models using the original and identified parameters
kwargs = {'with_angle_output':False, 'with_imu_output':True}

original_model = CarouselModel(original_param, *kwargs.values())
identified_model = CarouselModel(identified_param, *kwargs.values())
constants = CarouselModel.getConstants()

# Simulation data and convenience functions
dt = 0.05
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
    data = datasets[name]

    # Set simulation horizon
    N = min([N_margin+1500,len(data)])
    print("N = " +str(N))

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
    original_simulator = CarouselSimulator(model = original_model, x0 = x0)
    identified_simulator = CarouselSimulator(model = identified_model, x0 = x0)

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
                xf_k, zf_k, y0_k = original_simulator.simulate_timestep(controls[k], dt)
                Xs_original_sim += [xf_k.full()]
                Ys_original_sim += [y0_k.full()]
                if cs.mtimes(xf_k.T, xf_k) > max_sq_val:
                    original_simulator_crashed = True
                    print("Original model crashed!")

        if not identified_simulator_crashed:
            xf_k, zf_k, y0_k = identified_simulator.simulate_timestep(controls[k], dt)
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

    data_real['Xs'] = data_real['Xs']
    data_iden['Xs'] = data_iden['Xs']
    data_real['Ys'] = data_real['Ys']
    data_iden['Ys'] = data_iden['Ys']

    filename_out = "result_ident_" + key + ".pdf"
    with PdfPages("../../tex/thesis/figures/identification_imu/" + filename_out) as pdf:

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

plt.show()
