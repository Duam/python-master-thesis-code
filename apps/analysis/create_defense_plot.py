import pandas as pd
import numpy as np
import thesis_code.utils.data_processing_tools as proc
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pd.plotting.register_matplotlib_converters()
matplotlib.rcParams.update({'font.size': 14})

xsize = 10 # inches
dt = 0.05
T = 200
Tper = 10

channels = [
  'CONTROLS_sampled',
  'ANGLES_sampled',
  'state_estimate',
  'state_reference',
  'control_reference',
  'roll_reference',
  'controller_info',
  'estimator_info'
]

keys = [
    'CONTROLS_sampled_0', 'control_reference_0',
    'ANGLES_sampled_0', 'state_estimate_0', 'state_reference_0', 'roll_reference_0',
    'ANGLES_sampled_1', 'state_estimate_1', 'state_reference_1',
    'state_estimate_4', 'state_reference_4',
    'controller_info_0', 'controller_info_1', 'controller_info_2', 'controller_info_3', 'controller_info_4',
    'estimator_info_0', 'estimator_info_1', 'estimator_info_2', 'estimator_info_3', 'estimator_info_4'
]
""" ================================================================================================================ """

def toDeg(rad):
    return rad * 360 / (2*np.pi)

def round_to_extreme(val):
    maxval = val.max()
    minval = val.min()
    avgval = (maxval + minval) / 2.0
    return maxval if val >= avgval else minval

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

def time_series_plot(value, value_tar, value_ref,
                     Tper=10, 
                     ylabel="Value", 
                     value_label="Real", value_tar_label="Target", value_ref_label="Reference", 
                     legend_loc="upper left", annot_loc=(0.65,0.05), ylim=None,
                     round=False, pdf=None):
    global xsize

    # Compute mean values for each timestep
    value_ref_mean = value_ref.mean(axis=1)
    value_tar_mean = value_tar.mean(axis=1)
    value_mean = value.mean(axis=1)

    # Round the reference values. Needed for clean rectangle signal
    if round:
        maxval = value_ref_mean.max()
        minval = value_ref_mean.min()
        avgval = (maxval + minval) / 2.0
        def round_to_extreme(val):    
            return maxval if val >= avgval else minval
        value_ref_mean = value_ref_mean.apply(round_to_extreme)

    # Compute standard deviation and confidence interval
    value_stddev = np.sqrt(value.var(axis=1))
    value_ub = value_mean + value_stddev
    value_lb = value_mean - value_stddev
    # Compute error metric
    err_metric_value = ((value - value_ref)**2).mean().mean() # MSE
    #err_metric_value = ((value_mean - value_ref_mean)**2).mean()

    # Compute the time-axis and start plotting
    cmap = plt.get_cmap('tab10')
    tAxis = np.linspace(0, Tper, len(value_mean))
    fig, ax = plt.subplots(1, 1, figsize=(xsize,5))
    plt.ylabel(ylabel)
    plt.xlabel('Time $t$ [s]')
    plt.tight_layout()    
    if ylim is not None:
        plt.gca().set_ylim(ylim)


    plt.plot(tAxis, value_ref_mean, color=cmap(1), label=value_ref_label, linewidth=1)
    plt.legend(loc=legend_loc).draggable()
    pdf.savefig()
    
    plt.plot(tAxis, value_tar_mean, color=cmap(2), label=value_tar_label, linewidth=1)
    plt.legend(loc=legend_loc).draggable()
    pdf.savefig()

    plt.plot(tAxis, value_mean, color=cmap(0), label=value_label, linewidth=3)
    plt.legend(loc=legend_loc).draggable()
    plt.fill_between(tAxis, value_lb, value_ub, alpha=0.5, color=cmap(0))
    plt.annotate("MSE = " + "{0:.3e}".format(err_metric_value), xy=(annot_loc[0], annot_loc[1]), xycoords="axes fraction", fontsize=16)
    pdf.savefig()

    
    


period_offset_time = 0
period_offset_time = 4.5

in_filenames = [ 
  "./data/2019-10-04-mhe3-mpc3-imu/mhe3-mpc3-imu-rect-02.0000",
  "./data/2019-10-04-mhe3-mpc3-imu/mhe3-mpc3-imu-rect-04.0000"
]
out_filename = "../../tex/presentations/figures/experiment_result_plot.pdf"

with PdfPages(out_filename) as pdf:
    for k in range(len(in_filenames)):

        # Load real-estimator dataset and fetch data
        data = load_and_preprocess_dataset(in_filenames[k], channels, "50ms", period_offset_time)
        phi = data['ANGLES_sampled_0']#.apply(toDeg)
        phi_tar = data['state_reference_0']#.apply(toDeg)
        phi_ref = data['roll_reference_0']#.apply(toDeg)

        # Append plots to pdf
        time_series_plot(
            phi, phi_tar, phi_ref, 
            ylabel="Elevation $\phi$ [rad]", 
            value_label="Response", 
            value_tar_label="Target",
            value_ref_label="Reference",
            #ylim=(5, 36.25),
            #ylim=(5*np.pi/180, 36.5*np.pi/180),
            ylim=(0.0, 0.75),
            round=True, 
            pdf=pdf
        )

        plt.close()

#plt.show()
