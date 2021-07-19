from thesis_code.models.carousel_whitebox import CarouselWhiteBoxModel
from thesis_code.carousel_simulator import Carousel_Simulator
import matplotlib.pyplot as plt
import datetime, argparse, pprint, json
import casadi as cs
import numpy as np
import pandas as pd

# Today's date as a string
today = datetime.datetime.now().strftime("%Y_%m_%d")

# Parse arguments:
parser = argparse.ArgumentParser(description='Identification pipeline step 4: Validation')
parser.add_argument(
  '-p', '--prefix',
  dest='dataset_prefix',
  default="./",
  help="Prefix of the data set (local path and everything up to the underscore after the date)"
)
parser.add_argument(
  '-v', '--virtual',
  dest='is_virtual_experiment', default='True',
  help="Flag if this is a virtual experiment"
)


# Fetch arguments
args = parser.parse_args()
is_virtual_experiment = args.is_virtual_experiment == "True"

# Set input file prefix
in_file_prefix = args.dataset_prefix

""" ================================================================================================================ """

# Set validation data input filename
in_data_filename = in_file_prefix + "PREPROCESSED.csv"
print("Loading validation dataset " + in_data_filename)

# Load validation data
data = pd.read_csv(in_data_filename, header=0, parse_dates=[0])
print("Done. Loaded " + str(len(data)) + " data points.")

# Load original parameters
original_param = {}
in_original_param_filename = in_file_prefix + "INITIAL_PARAMS.json"
print("Loading identified parameter set " + in_original_param_filename)
with open(in_original_param_filename, 'r') as file:
    original_param = json.load(file)

# Load identified parameters
identified_param = {}
in_identified_param_filename = in_file_prefix + "IDENTIFIED_PARAMS.json"
print("Loading identified parameter set " + in_identified_param_filename)
with open(in_identified_param_filename, 'r') as file:
    identified_param = json.load(file)

print("Done. Parameters:")
pprint.pprint(identified_param)

""" ================================================================================================================ """

# Create models using the original and identified parameters
original_model = CarouselWhiteBoxModel(original_param)
identified_model = CarouselWhiteBoxModel(identified_param)

# We simulate for N steps
N = min([4000, len(data)])
#N = len(data)
#N = 20

# Timestep todo: read out from dataset?
dt = 0.05

# Set initial state
get_roll = lambda yr:  original_param['roll_sensor_offset'] - yr
get_pitch = lambda yp: original_param['pitch_sensor_offset'] - yp
yaw_0 = data['yaw'][0]
roll_0 = get_roll(data['roll'][0])
pitch_0 = get_pitch(data['pitch'][0])
roll_1 = get_roll(data['roll'][1])
pitch_1 = get_pitch(data['pitch'][1])
x0 = cs.DM([
    roll_0 + 1e-1,
    pitch_0 + 1e-2,
    yaw_0,
    (roll_1 - roll_0) / dt,
    (pitch_1 - pitch_0) / dt,
    -2,
    identified_model.u0()[0] # Identified by hand previously
])

# Set up the simulator
original_simulator = Carousel_Simulator(model = original_model, x0 = x0)
identified_simulator = Carousel_Simulator(model = identified_model, x0 = x0)

# Simulate using the excitation signal from the validation dataset
Ys_original_sim = []
Ys_identified_sim = []
print("Simulating..")
for k in range(N):
    xf_k, zf_k, y0_k = original_simulator.simstep(data['control'][k], dt)
    Ys_original_sim += [y0_k.full()]
    xf_k, zf_k, y0_k = identified_simulator.simstep(data['control'][k], dt)
    Ys_identified_sim += [y0_k.full()]

print("Simulation done.")

# Put simulated output in a dataframe

#columns=['gyr_0', 'gyr_1', 'gyr_2', 'acc_0', 'acc_1', 'acc_2', 'yaw_sin', 'yaw_cos', 'roll', 'pitch']
columns=['roll', 'pitch']
Ys_original_sim = pd.DataFrame(np.array(Ys_original_sim).squeeze(), columns=columns)
Ys_identified_sim = pd.DataFrame(np.array(Ys_identified_sim).squeeze(), columns=columns)

# Fetch relevant validation data
Ys_val = data
#Ys_val['yaw_sin'] = np.sin(Ys_val['yaw'])
#Ys_val['yaw_cos'] = np.cos(Ys_val['yaw'])
Ys_val = Ys_val[columns]

""" ================================================================================================================ """

# Trim datasets
Ys_val = Ys_val.head(N)
Ys_original_sim = Ys_original_sim.head(N)
Ys_identified_sim = Ys_identified_sim.head(N)

# Compute MSE of original parameter set
err_original = Ys_val - Ys_original_sim
original_rms_eachField_allSteps = (err_original ** 2).mean(axis='index') ** 0.5
original_rms_allFields_allSteps = (err_original ** 2).sum(axis='columns').mean(axis='index') ** 0.5
original_rms_allFields_eachStep = (err_original ** 2).mean(axis='columns') ** 0.5
original_rms_eachField_eachStep = (err_original ** 2) ** 0.5

# Compute RMS of identified parameter set
err_identified = Ys_val - Ys_identified_sim
identified_rms_eachField_allSteps = (err_identified ** 2).mean(axis='index') ** 0.5
identified_rms_allFields_allSteps = (err_identified ** 2).sum(axis='columns').mean(axis='index') ** 0.5
identified_rms_allFields_eachStep = (err_identified ** 2).mean(axis='columns') ** 0.5
identified_rms_eachField_eachStep = (err_identified ** 2) ** 0.5

print("ORIGINAL PARAMETERS:")
print("RMS value per field is\n" + str(original_rms_eachField_allSteps))
print("IDENTIFIED PARAMETERS:")
print("RMS value per field is\n" + str(identified_rms_eachField_allSteps))
print("TOTAL IMPROVEMENT:")
print("Total RMS (original) = " + str(original_rms_allFields_allSteps))
print("Total RMS (identified) = " + str(identified_rms_allFields_allSteps))

# Plot result
if True:
    print("Plotting results..")
    fig, ax = plt.subplots(4,2,sharex='all')
    plt.sca(ax[0, 0])
    plt.title('Controls')
    plt.plot(data['control'][:N])

    plt.sca(ax[1,0])
    plt.title('Roll')
    plt.plot(Ys_identified_sim['roll'], label="Simulated (identified parameters)")
    plt.plot(Ys_original_sim['roll'], '--', label="Simulated (original parameters)")
    plt.plot(Ys_val['roll'], label="Real")

    plt.sca(ax[2, 0])
    plt.title('Pitch')
    plt.plot(Ys_identified_sim['pitch'], label="Simulated (identified parameters)")
    plt.plot(Ys_original_sim['pitch'], '--', label="Simulated (original parameters)")
    plt.plot(Ys_val['pitch'], label="Real")
    plt.legend()

    """
    plt.sca(ax[3, 0])
    plt.title('Yaw')
    plt.plot(Ys_identified_sim['yaw_sin'], label="Simulated (identified parameters) (sin)")
    plt.plot(Ys_original_sim['yaw_sin'], '--', label="Simulated (original parameters) (sin)")
    plt.plot(Ys_val['yaw_sin'], label="Real (sin)")
    plt.plot(Ys_identified_sim['yaw_cos'], label="Simulated (identified parameters) (cos)")
    plt.plot(Ys_original_sim['yaw_cos'], '--', label="Simulated (original parameters) (cos)")
    plt.plot(Ys_val['yaw_cos'], label="Real (cos)")
    plt.legend()
    """

    plt.sca(ax[0, 1])
    plt.title("Total RMS per step")
    plt.plot(identified_rms_allFields_eachStep, label="identified parameters")
    plt.plot(original_rms_allFields_eachStep, '--', label="original parameters")
    plt.yscale('log')

    plt.sca(ax[1, 1])
    plt.title('Roll RMS')
    plt.plot(identified_rms_eachField_eachStep['roll'], label="identified parameters")
    plt.plot(original_rms_eachField_eachStep['roll'], '--', label="original parameters")
    plt.yscale('log')

    plt.sca(ax[2, 1])
    plt.title('Pitch RMS')
    plt.plot(identified_rms_eachField_eachStep['pitch'], label="identified parameters")
    plt.plot(original_rms_eachField_eachStep['pitch'], '--', label="original parameters")
    plt.legend()
    plt.yscale('log')

    """
    plt.sca(ax[3, 1])
    plt.title('Yaw RMS')
    plt.plot(identified_rms_eachField_eachStep['yaw_sin'], label="identified parameters (sin)")
    plt.plot(original_rms_eachField_eachStep['yaw_sin'], '--', label="original parameters (sin)")
    plt.plot(identified_rms_eachField_eachStep['yaw_cos'], label="identified parameters (cos)")
    plt.plot(original_rms_eachField_eachStep['yaw_cos'], '--', label="original parameters (cos")
    plt.legend().draggable()
    plt.yscale('log')
    """

    plt.show()
    quit(0)
