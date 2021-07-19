import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
pd.plotting.register_matplotlib_converters()
from thesis_code.models.carousel_whitebox import CarouselWhiteBoxModel
import thesis_code.utils.data_processing_tools as proc

do_plot_timeseries = False
do_plot_errorplot = True

# Fetch params
model_params = CarouselWhiteBoxModel.getDefaultParams()
model_constants = CarouselWhiteBoxModel.getConstants()
offsets = {
    'VE1_SCAPULA_ELEVATION': model_constants['roll_sensor_offset'],
    'VE1_SCAPULA_ROTATION': model_constants['pitch_sensor_offset']
}

# Open and load logfile
#filename = './data/zcmlog-2019-09-29_ss_playback_mhe_imu.0002'
#filename = './data/zcmlog-2019-09-29_rect_playback_mhe_imu.0002'
#filename = './data/zcmlog-2019-09-29_sine_playback_mhe_imu.0002'
dataname = "rect-02"
filename = "./data/2019-10-04-mhe3-mpc3/mhe3-mpc3-" + dataname + ".0000"
filename = "./data/2019-10-04-mhe3-mpc3-imu/mhe3-mpc3-imu-" + dataname + ".0000"

channels = ['VE2_KIN4_SET', 'VE1_SCAPULA_ELEVATION', 'VE1_SCAPULA_ROTATION', 'state_estimate']

# Load data
print("Loading file \"" + filename + "\". Channels: " + str(channels))
dataset = proc.load_log_to_dataframe(filename, channels)

# Print mean and variances
print("Printing statistics..")
for key in dataset.keys():
    data = dataset[key]
    mean = data.mean().values
    variance = data.var().values
    print(key + ": mean = " + str(mean) + ", variance = " + str(variance))

# Convert angle measurements
for key in ['VE1_SCAPULA_ELEVATION', 'VE1_SCAPULA_ROTATION']:
    offs = offsets[key]
    dataset[key] = offs - dataset[key]

# Resample (zero-order-hold)
dataset = proc.resample_data(dataset, "50ms")
dataset = proc.join_and_trim_data(dataset)

# Set sizes
dt = 0.05
T_end = 100
T_start = 100
N_skip = len(data) - int(T_start / dt)
N = min([int(T_end / dt), len(data)])
dataset = dataset.tail(N_skip).head(N)

# Plot state estimate vs. real data
print("Plotting state estimate vs. truth..")

def toDeg(rad):
    return rad * 360 / (2*np.pi)

# Fetch data
controls_real = dataset['VE2_KIN4_SET_0']
tab_angle_esti = dataset['state_estimate_4']
elevation_real = dataset['VE1_SCAPULA_ELEVATION_0']
elevation_esti = dataset['state_estimate_0']
rotation_real = dataset['VE1_SCAPULA_ROTATION_0']
rotation_esti = dataset['state_estimate_1']

# Compute mse ('mean' part only relevant if I do multiple trials)
elevation_err = elevation_real - elevation_esti
rotation_err = rotation_real - rotation_esti
dataset['elevation_err'] = pd.Series(elevation_err, index=dataset.index)
dataset['rotation_err'] = pd.Series(rotation_err, index=dataset.index)



if do_plot_timeseries:
    # Plot
    fig, ax = plt.subplots(3, 2, sharex='all')
    # plt.suptitle('First carousel state estimation test \n MHE (N=3), constant controls')
    plt.sca(ax[0, 0])
    plt.title('Controls')
    plt.plot(controls_real, label='Setpoint')
    plt.plot(tab_angle_esti, label='Estimated')
    plt.legend(loc='lower right').draggable()
    plt.sca(ax[1, 0])

    plt.title('Roll')
    plt.plot(elevation_esti, label='Estimated')
    plt.plot(elevation_real, label='Real')
    plt.ylabel('$\phi(t)$ [rad]')
    plt.legend(loc='lower right').draggable()

    plt.sca(ax[1, 1])
    plt.title('Roll MSE')
    plt.plot(elevation_err)
    #plt.yscale('log')

    plt.sca(ax[2, 0])
    plt.title('Pitch')
    plt.plot(rotation_esti, label='Estimated')
    plt.plot(rotation_real, label='Real')
    plt.ylabel('$\\theta(t)$ [rad]')
    plt.xlabel('Time $t$ [s]')

    plt.sca(ax[2, 1])
    plt.title('Pitch MSE')
    plt.plot(rotation_err)
    plt.xlabel('Time $t$ [s]')
    #plt.yscale('log')

if do_plot_errorplot:
    # Weniger gridlines
    # Keine borderlines

    elevation_real = elevation_real.apply(toDeg)
    elevation_esti = elevation_esti.apply(toDeg)
    rotation_real = rotation_real.apply(toDeg)
    rotation_esti = rotation_esti.apply(toDeg)
    
    # Plot
    margin = 0.15
    fig, ax = plt.subplots(1, 2)
    plt.sca(ax[0])
    plt.title("Elevation")
    plt.plot(elevation_real, elevation_esti, 'o', markersize=2, color='black')
    sb.kdeplot(elevation_real, elevation_esti, shade=True, shade_lowest=False)
    minv = min([ min(elevation_real), min(elevation_esti) ])
    maxv = max([ max(elevation_real), max(elevation_esti) ])
    dist = maxv - minv
    minv -= margin * dist
    maxv += margin * dist
    plt.plot([minv,maxv],[minv,maxv])
    plt.xlabel('Ground Truth $\phi$ [deg]')
    plt.ylabel('Estimate $\hat{\phi}$ [deg]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    plt.sca(ax[1])
    plt.title("Rotation")
    plt.plot(rotation_real, rotation_esti, 'o', markersize=2, color='black')
    sb.kdeplot(rotation_real, rotation_esti, shade=True, shade_lowest=False)
    minv = min([ min(rotation_real), min(rotation_esti) ])
    maxv = max([ max(rotation_real), max(rotation_esti) ])
    dist = maxv - minv
    minv -= margin * dist
    maxv += margin * dist
    plt.plot([minv,maxv],[minv,maxv])
    plt.xlabel('Ground Truth $\\theta$ [deg]')
    plt.ylabel('Estimate $\hat{\\theta}$ [deg]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

plt.show()

