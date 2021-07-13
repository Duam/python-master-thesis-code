import pandas as pd
import numpy as np
from thesis_code.models.carousel_whitebox import CarouselWhiteBoxModel
import thesis_code.utils.data_processing_tools as proc
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pd.plotting.register_matplotlib_converters()

# Fetch params
model_constants = CarouselWhiteBoxModel.getConstants()
offsets = {
    'VE1_SCAPULA_ELEVATION': model_constants['roll_sensor_offset'],
    'VE1_SCAPULA_ROTATION': model_constants['pitch_sensor_offset']
}

dt = 0.05
T = 200
Tper = 10

channels = ['CONTROLS_sampled',
            'ANGLES_sampled',
            'state_estimate',
            'state_reference',
            'control_reference',
            'roll_reference',
            'controller_info',
            'estimator_info']

keys = [
    'CONTROLS_sampled_0', 'control_reference_0',
    'ANGLES_sampled_0', 'state_estimate_0', 'state_reference_0', 'roll_reference_0',
    'ANGLES_sampled_1', 'state_estimate_1', 'state_reference_1',
    'state_estimate_4', 'state_reference_4',
    'controller_info_0', 'controller_info_1', 'controller_info_2', 'controller_info_3', 'controller_info_4',
    'estimator_info_0', 'estimator_info_1', 'estimator_info_2', 'estimator_info_3', 'estimator_info_4'
]

""" ================================================================================================================ """

xsize = 10 # inches

def toDeg(rad):
    return rad * 360 / (2*np.pi)

def load_and_preprocess_dataset(path, channels, freq, period_offset):
    global dt, T, Tper

    # Load dataset
    print("Loading file \"" + path + "\". Channels: " + str(channels))
    dataset = proc.load_log_to_dataframe(path, channels)
    dataset = proc.resample_data(dataset, freq)
    dataset = proc.join_and_trim_data(dataset)
    
    # Trim dataset
    N = min([int(T / dt), len(dataset)])
    dataset = dataset.head(N)

    # Put data into time bins:
    data = {}
    for key in keys:
        # Fetch dataset and ditch timestamps
        ds = dataset[key]
        ds.index = range(len(ds))

        # Compute sample and period numbers
        sample_offset = int( period_offset / float(dt) )
        samples_per_period = int( Tper / float(dt) )
        num_periods = int( len(ds) / float(samples_per_period) )

        # Split up each time series into Tper long segments
        series_dict = {}
        for i in range(num_periods-1):
            k_start = i * samples_per_period + sample_offset
            k_end = k_start + samples_per_period
            series_dict[str(i)] = ds[k_start:k_end].array

        # Create a 2d data table for the time series
        df = pd.DataFrame.from_dict(series_dict)
        data[key] = df
    
    return data


def time_series_error_plot(value1, 
                           value2, 
                           value_ref, 
                           reference, 
                           Tper=10, 
                           ylabel="Value", 
                           value_label="Real", 
                           value_ref_label="Reference", 
                           annot_loc=(0.60,0.05),
                           ylim=None):
    global xsize

    # Compute error for each timestamp
    error1 = value_ref - value1
    error2 = value_ref - value2
    # Compute mean values for each timestep
    error1_mean = error1.mean(axis=1)
    error2_mean = error2.mean(axis=1)
    # Compute standard deviation and confidence interval
    error1_stddev = np.sqrt(error1.var(axis=1))
    error1_ub = error1_mean + error1_stddev
    error1_lb = error1_mean - error1_stddev
    error2_stddev = np.sqrt(error2.var(axis=1))
    error2_ub = error2_mean + error2_stddev
    error2_lb = error2_mean - error2_stddev

    # Compute error metric
    err1_metric_value = (error1 ** 2).mean().mean()
    err2_metric_value = (error2 ** 2).mean().mean()

    cmap = plt.get_cmap('tab10')

    # Compute the time-axis and start plotting
    tAxis = np.linspace(0, Tper, len(error1_mean))
    fig, ax = plt.subplots(1, 1, figsize=(xsize,3))
    #ax2 = ax.twinx()
    #plt.plot(tAxis, reference, alpha=0.1, label='Target', color='orange')
    #plt.ylabel('Target Elevation [deg]', color='orange')
    #plt.sca(ax)
    plt.plot(tAxis, error2_mean, label='Error (IMU)', color=cmap(0)) #color='blue', linewidth=3)
    plt.fill_between(tAxis, error2_lb, error2_ub, alpha=0.5, color=cmap(0))#, color='grey')
    plt.plot(tAxis, error1_mean, label='Error (ENC)', color=cmap(1)) #color='blue', linewidth=3)
    plt.fill_between(tAxis, error1_lb, error1_ub, alpha=0.5, color=cmap(1))
    #plt.fill_between(tAxis, phi_lb, phi_ub, alpha=0.5, color='grey')
    #plt.fill_between(tAxis, phi_est_lb, phi_est_ub, alpha=0.5, color='orange')
    
    #plt.plot(tAxis, error1, alpha=0.1, color='grey')
    plt.annotate("MSE (ENC) = " + "{0:.3e}".format(err1_metric_value), xy=(annot_loc[0], annot_loc[1]), xycoords="axes fraction", fontsize=16)
    plt.annotate("MSE (IMU) = " + "{0:.3e}".format(err2_metric_value), xy=(annot_loc[0], annot_loc[1]+0.1), xycoords="axes fraction", fontsize=16)

    plt.ylabel(ylabel)
    plt.xlabel('Time $t$ [s]')
    plt.tight_layout()
    if ylim is not None:
        plt.gca().set_ylim(ylim)

    plt.legend(loc='upper left').draggable()
    #plt.grid(True)



def time_series_plot(value, 
                     value_ref, 
                     Tper=10, 
                     ylabel="Value", 
                     value_label="Real", 
                     value_ref_label="Reference", 
                     legend_loc="upper left", 
                     annot_loc=(0.65,0.05),
                     ylim=None):
    global xsize

    # Compute mean values for each timestep
    value_mean = value.mean(axis=1)
    value_ref_mean = value_ref.mean(axis=1)
    # Compute standard deviation and confidence interval
    value_stddev = np.sqrt(value.var(axis=1))
    value_ub = value_mean + value_stddev
    value_lb = value_mean - value_stddev
    # Compute error metric
    err_metric_value = ((value - value_ref)**2).mean().mean() # MSE

    # Compute the time-axis and start plotting
    tAxis = np.linspace(0, Tper, len(value_mean))
    fig, ax = plt.subplots(1, 1, figsize=(xsize,3))
    plt.plot(tAxis, value_ref_mean, color='red', label=value_ref_label, linewidth=3)
    plt.plot(tAxis, value_mean, color='blue', label=value_label)
    plt.plot(tAxis, value, alpha=0.1, color='grey')
    #print(np.tile(tAxis,len(value[1])).shape)
    #print(value.values.flatten().shape)
    #sb.kdeplot(np.tile(tAxis, len(value)), value.values.flatten(), shade=True, shade_lowest=False)
    #plt.fill_between(tAxis, phi_lb, phi_ub, alpha=0.5, color='grey')
    #plt.fill_between(tAxis, phi_est_lb, phi_est_ub, alpha=0.5, color='orange')
    #plt.plot(phi_bar_mean, label='Actual ref')
    plt.annotate("MSE = " + "{0:.3e}".format(err_metric_value), xy=(annot_loc[0], annot_loc[1]), xycoords="axes fraction", fontsize=16)
    plt.ylabel(ylabel)
    plt.xlabel('Time $t$ [s]')
    plt.tight_layout()
    
    if ylim is not None:
        plt.gca().set_ylim(ylim)

    plt.legend(loc=legend_loc).draggable()
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.grid(True)


def characteristic_curve_plot(value, value_ref, Nplot=100, xlabel="Ground Truth [deg]", ylabel="Estimate [deg]", plotlim=[0,20]):
    linelim = 100
    value_real = value.tail(Nplot)
    value_esti = value_ref.tail(Nplot)
    
    value_real_min = value_real.min().min()
    value_real_max = value_real.max().max()
    value_esti_min = value_esti.min().min()
    value_esti_max = value_esti.max().max()
    
    # Plot
    margin = 0.15
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    plt.plot(value_real, value_esti, 'o', markersize=2, color='black')
    #sb.kdeplot(value_real, value_esti, shade=True, shade_lowest=False)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot([-linelim,linelim],[-linelim,linelim])
    plt.gca().set_xlim(plotlim)
    plt.gca().set_ylim(plotlim)
    plt.grid(True)



# Choose which component+reference to analyize
components = ['tarsel', 'estimator', 'controller']
component = components[1]

references = ["const", "rect", "saw", "tria"]
reference = references[1]

# Directories
dir1 = "./data/2019-10-04-mhe3-mpc3/"
dir2 = "./data/2019-10-04-mhe3-mpc3-imu/"

# Filenames
paths1_in = []
paths2_in = []
if reference == "const":
    paths1_in = [ dir1 + "mhe3-mpc3-const.0000" ]
    paths2_in = [ dir2 + "mhe3-mpc3-imu-const.0000" ]
else:
    paths1_in = [ dir1 + "mhe3-mpc3-" + reference + "-02.0000",
                      dir1 + "mhe3-mpc3-" + reference + "-03.0000",
                      dir1 + "mhe3-mpc3-" + reference + "-04.0000",
                      dir1 + "mhe3-mpc3-" + reference + "-05.0000"]
    paths2_in = [ dir2 + "mhe3-mpc3-imu-" + reference + "-02.0000",
                      dir2 + "mhe3-mpc3-imu-" + reference + "-03.0000",
                      dir2 + "mhe3-mpc3-imu-" + reference + "-04.0000",
                      dir2 + "mhe3-mpc3-imu-" + reference + "-05.0000"]

    #paths1_in = [ paths1_in[0] ]
    #paths2_in = [ paths2_in[0] ]

# The output filename
filename_out = component + "_result_" + reference + ".pdf"


period_offset_time = 0
if reference == "const":
    period_offset_time = 0.0
    plotlim = [7,16]
elif reference == "rect":
    period_offset_time = 4.5
    plotlim = [0,20]
elif reference == "tria":
    period_offset_time = 6.0
    plotlim = [0,20]

with PdfPages("../../tex/thesis/figures/control/" + filename_out) as pdf:
    for k in range(len(paths1_in)):

        # Load real-estimator dataset and fetch data
        data = load_and_preprocess_dataset(paths1_in[k], channels, "50ms", period_offset_time)
        u = data['CONTROLS_sampled_0']
        u_tar = data['control_reference_0']
        phi = data['ANGLES_sampled_0'].apply(toDeg)
        phi_est = data['state_estimate_0'].apply(toDeg)
        phi_tar = data['state_reference_0'].apply(toDeg)
        phi_ref = data['roll_reference_0'].apply(toDeg)

        # Append plots to pdf
        if component == "tarsel":
            time_series_plot(
                phi_tar, phi_ref, 
                ylabel="Elevation $\phi$ [deg]", 
                value_label="Target", 
                value_ref_label="Reference",
                ylim=(5, 36.25))
            pdf.savefig()

        else:
            # Load cheat-estimator dataset and fetch data
            data_cheat = load_and_preprocess_dataset(paths2_in[k], channels, "50ms", period_offset_time)
        
            # Do different plots depending on the component under analysis
            if component == "estimator":
                phi_cheat_est = data_cheat['state_estimate_0'].apply(toDeg)
                time_series_error_plot(
                    phi_est, phi_cheat_est, phi, phi_tar,
                    ylabel="Elevation Estimation Error [deg]", 
                    value_label="", 
                    value_ref_label="", 
                    ylim=(-9., 9.)
                    )
                pdf.savefig()

                plotlim = (10,45)   
                characteristic_curve_plot(
                    phi, phi_est, 
                    xlabel="Elevation $\phi$ [deg]",
                    ylabel="Elevation Estimate $\hat{\phi}$ [deg]",
                    plotlim=plotlim
                )
                pdf.savefig()

            if component == "controller":
                # Plot the NMPC performance with the IMU-NMHE
                time_series_plot(
                    phi, phi_tar,
                    ylabel="Elevation $\phi$ [deg]", 
                    value_label="NMPC + NMHE",
                    value_ref_label="Target", 
                    ylim=(3, 40)
                    )
                pdf.savefig()

                time_series_plot(
                    u, u_tar,
                    ylabel="Control $u$", 
                    value_label="Real",
                    value_ref_label="Target",
                    legend_loc="lower left",
                    annot_loc=(0.2,0.05),
                    ylim=(-0.1, 1.1)
                )
                pdf.savefig()

                # Plot the NMPC performance with the cheat-NMHE
                phi_cheat = data_cheat['ANGLES_sampled_0'].apply(toDeg)
                u_cheat = data_cheat['CONTROLS_sampled_0']
                time_series_plot(
                    phi_cheat, phi_tar,
                    ylabel="Elevation $\phi$ [deg]", 
                    value_label="NMPC + NMHE",
                    value_ref_label="Target", 
                    ylim=(3, 40)
                    )
                pdf.savefig()

                time_series_plot(
                    u_cheat, u_tar,
                    ylabel="Control $u$", 
                    value_label="Real",
                    value_ref_label="Target", 
                    legend_loc="lower left",
                    annot_loc=(0.2,0.05),
                    ylim=(-0.1, 1.1)
                )
                pdf.savefig()

        plt.close()

#plt.show()
