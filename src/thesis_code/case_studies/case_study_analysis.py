#!/usr/bin/python3

import numpy as np
np.set_printoptions(linewidth=np.inf)
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Colormap, Normalize
import matplotlib.cm as cmx
from matplotlib.ticker import LogFormatter
import argparse

# Parse input filename
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help=".nc file")
args = parser.parse_args()

filename = args.file

# Open dataset
data = {}
#filename = 'case_study_data_raw.nc'
try:
  data = xr.open_dataset(filename)
  print("Dataset opened. Content:")
  print(data)
except:
  print("Dataset \'" + filename + "\' could not be opened. Does it exist?")
  print("Aborting.")
  exit(1)

# Fetch some meta data
print("============================ FETCHING META INFORMATION ==============================")
NUM_STUDIES = data.sizes['study']
NUM_CASES = data.sizes['case']
NUM_RUNS = data.sizes['run']
NUM_SAMPLES_PER_PERIOD = data.sizes['trial_sample']
NUM_TRIALS_WARMUP = data.attrs['M_warmup']
NUM_TRIALS_SETTLE = data.attrs['M_settle']
NUM_TRIALS_CONVER = data.attrs['M_conv']
NUM_TRIALS_ERRORS = data.attrs['M_err']
NUM_TRIALS_TOTAL = NUM_TRIALS_WARMUP + NUM_TRIALS_SETTLE + NUM_TRIALS_CONVER + NUM_TRIALS_ERRORS
NUM_SAMPLES_INIT = (data.attrs['M_warmup'] + data.attrs['M_settle']) * NUM_SAMPLES_PER_PERIOD
NUM_SAMPLES_TOTAL = data.sizes['sample']
NUM_SAMPLES = NUM_SAMPLES_TOTAL - NUM_SAMPLES_INIT
NUM_SAMPLES_CONVERGENCE_TEST = data.attrs['M_conv'] * NUM_SAMPLES_PER_PERIOD
NUM_SAMPLES_ERROR_TEST = data.attrs['M_err'] * NUM_SAMPLES_PER_PERIOD
SAMPLE_MARKERS = [k for k in range(NUM_SAMPLES_PER_PERIOD)]
TRIAL_MARKERS = [0, 
                 NUM_TRIALS_WARMUP,
                 NUM_TRIALS_WARMUP + NUM_TRIALS_SETTLE,
                 NUM_TRIALS_WARMUP + NUM_TRIALS_SETTLE + NUM_TRIALS_CONVER,
                 NUM_TRIALS_WARMUP + NUM_TRIALS_SETTLE + NUM_TRIALS_CONVER + NUM_TRIALS_ERRORS ]


print("======================= REORDERING ==============================")
# Put samples into bins
new_sample_axis = SAMPLE_MARKERS
new_trial_axis = range(NUM_TRIALS_TOTAL)
ind = pd.MultiIndex.from_product((new_trial_axis,new_sample_axis), names=['trials','samples'])

data = data.assign(sample=ind).unstack('sample')
data = data.rename({'samples':'sample', 'trials':'trial'})

# Compute error between simulation and estimation
E = data['parameters_sim'] - data['parameters_est']
E = np.sqrt(np.square(E))

# Fetch convergence and error analysis bins
E_con = E[dict(trial=range(TRIAL_MARKERS[2],TRIAL_MARKERS[3]))].squeeze()
E_err = E[dict(trial=range(TRIAL_MARKERS[3],TRIAL_MARKERS[4]))].squeeze()

print("===================== COMPUTING CONVERGENCE RATES =======================")
# Compute convergence rate
C = E_con[dict(trial=1)].max('sample') / E_con[dict(trial=0)].max('sample')
# Truncate first element (w=0)
#C = C.where(C['run'] != 0.0, drop=True)
print('C = ', C)

print("========================= COMPUTING ERRORS ==========================")
# Compute errors
E = E_err[dict(trial=NUM_TRIALS_ERRORS-1)].max(dim='sample')
#E = E_err.mean(dim='sample').mean(dim='trial')
#E = E_err.mean(dim='sample').max(dim='trial')
# Truncate first element (w=0)
#E = E.where(E['run'] != 0.0, drop=True)
print('E=', E)

print("========================= PLOTTING ==========================")


plot_fam = False
plot_contour = False #True
plot_convergence = True

# Plot family of curves E(w,alpha) vs alpha
if plot_fam:
  # Create axis values
  alphaAxis = data.data_vars['alphas']

  # Create curve labels
  wAxis = data.data_vars['weight_W']

  # Create line colors
  jet = plt.get_cmap('jet')
  cNorm = LogNorm(vmin=wAxis[1], vmax=wAxis[-1])
  scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=jet)

  # Create and configure the figure
  fig, ax = plt.subplots(NUM_STUDIES+1, 1)
  fig.suptitle("Dataset " + filename, fontsize=16)
  
  plt.sca(ax[0])
  plt.gca().set_title(r"Error $\mathcal{E}_{i,j,k}$ vs. $\alpha_j$")
  
  # Plot results for all studies
  for i in range(NUM_STUDIES):
    # Configure subplot
    plt.sca(ax[i])
    
    plt.gca().set_ylabel(r"$\mathcal{E}_{" + str(i) + r",j,k}$")
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().grid()

    Ei = E[dict(study=i)]
    Emin = Ei.min().data
    Emax = Ei.max().data
    #plt.gca().set_ylim(Emin,Emax)

    for k in range(1,NUM_RUNS):
      yData = Ei[dict(run=k)]
      plt.plot(
        alphaAxis, yData,
        marker='x',
        color=scalarMap.to_rgba(wAxis[k]),
        label = wAxis[k].data,
        linewidth=1,
        markersize=4
      )    
    
  
  # Plot result for w=0 (same for all studies)
  plt.sca(ax[-1])
  plt.gca().set_ylabel(r"$\mathcal{E}_{" + str(i) + r",j,0}$")
  plt.gca().set_yscale('log')
  plt.gca().set_xscale('log')
  plt.gca().grid()

  yData = E[dict(study=0,run=0)]
  plt.plot(
    alphaAxis, yData,
    marker='x',
    markersize=4,
    label = wAxis[0].data
  )    

  plt.gca().legend(title="Weight W", loc='best', fontsize='small')
  plt.gca().set_xlabel(r"$\alpha_j$")

  # Plot a colorbar to show value of w
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  scalarMap._A = []
  #plt.sca(ax[-2])
  cbar = plt.colorbar(scalarMap, cax=cbar_ax, orientation='vertical')
  cbar_ax.set_title(r'$w_k$')



# Contour-plot E(w,alpha) vs. w and alpha
# Plot family of curves E(w,alpha) vs alpha
if plot_contour:

  # Choose data (alpha !=0, w != 0)  
  E_contour = E.where(E['case'] != 0, drop=True).where(E['run'] != 0, drop=True)

  # Create axis values
  alphaAxis = data.data_vars['alphas'].data[1:]
  wAxis = data.data_vars['weight_W'].data[1:]

  # Create and configure the figure
  fig, ax = plt.subplots(NUM_STUDIES, 1)
  fig.suptitle("Dataset " + filename, fontsize=16)
  
  plt.sca(ax[0])
  plt.gca().set_title(r"Contour plot of error $\mathcal{E}_{i,j,k}$ vs. $\alpha_j$ and $w_{i,k}$")

  # Plot results for all studies
  for i in range(NUM_STUDIES):
    # Configure subplot
    plt.sca(ax[i])
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylabel(r"$w_{" + str(i+1) + r",k}$")
    plt.gca().grid()

    # Fetch data and get min and max exponent
    E_i = E_contour[dict(study=i)]
    E_i_min = E_i.min().data
    E_i_max = E_i.max().data
    min_exp = min(np.floor(np.log10(np.abs(E_i_min))),0)
    max_exp = max(np.ceil(np.log10(np.abs(E_i_max))),min_exp+1)
    
    #print(min_exp)
    #print(max_exp)
    #print(E_i_min)
    #print(np.floor(np.log10(E_i_min)))
    #print(E_i_max)
    #print(np.ceil(np.log10(E_i_max)))

    levels = np.logspace(min_exp,max_exp, 20)

    # Plot contours
    zData = E_i.data.transpose()
    cs = plt.gca().contourf(
      alphaAxis,
      wAxis,
      zData,
      #levels=np.logspace(min_exp-1,max_exp+1,5*(abs(min_exp-1)+(max_exp+1))),
      #levels=levels,
      levels=np.geomspace(E_i.min(), E_i.max(),11),
      norm=LogNorm()
    )

    logformatter = LogFormatter(10, labelOnlyBase=False)
    cb = plt.colorbar(cs, ax=plt.gca(), extend='both', shrink=0.9, format=logformatter)
    #cb.ax.set_yticklabels(levels)
    #[label.set_visible(False) for label in cb.ax.yaxis.get_ticklabels()[1::2]]
    #[label.set_visible(True) for label in cb.ax.yaxis.get_ticklabels()[::2]]
    #[label.set_visible(True) for label in cb.ax.yaxis.get_ticklabels()]

  #plt.gca().legend(title="Weight W", loc='best', fontsize='small')
  plt.gca().set_xlabel(r"$\alpha_j$")


# Plot E,C vs W
if plot_convergence:
  # Truncate first element (w=0)
  C = C.where(C['run'] != 0.0, drop=True)
  E = E.where(E['run'] != 0.0, drop=True)

  # Create ticks on the x-axis
  wAxis = data.data_vars['weight_W'][1:]
  alphaAxis = data.data_vars['alphas']

  # Create line colors
  jet = plt.get_cmap('jet')
  cNorm = LogNorm(vmin=alphaAxis[1], vmax=alphaAxis[-1])
  scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=jet)


  # Create the figure
  fig, ax = plt.subplots(NUM_STUDIES,2)
  fig.suptitle("Dataset " + filename, fontsize=16)

  # Plot convergece rates
  plt.sca(ax[0,0])
  plt.title(r"Convergence rate vs. weight for $\alpha = 0$")
  plt.xlabel(r"$W_i$")
  for i in range(NUM_STUDIES):
    plt.sca(ax[i,0])
    plt.ylabel(r"$\mathcal{C}$ for $W_"+str(i+1)+r"$")
    plt.grid()
    plt.gca().set_ylim([C[dict(study=i,case=0)].min(),C[dict(study=i,case=0)].max()])
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    yData = C[dict(study=i,case=0)]
    plt.plot(
      wAxis, yData,
      marker='x',
      label=alphaAxis[0].data
    )
    
  plt.gca().set_xlabel(r"$w_k$")
  
  # Plot error
  plt.sca(ax[0,1])
  plt.title("Max. estimation error vs. weight")
  plt.xlabel(r"$W_i$")
  for i in range(NUM_STUDIES):
    plt.sca(ax[i,1])
    plt.ylabel(r"$\mathcal{E}$ for $W_"+str(i+1)+r"$")
    plt.grid()
    plt.gca().set_ylim([E[dict(study=i)].min(),E[dict(study=i)].max()])
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')

    for j in range(1,NUM_CASES):
      yData = E[dict(study=i,case=j)]
      plt.plot(
        wAxis, yData,
        color=scalarMap.to_rgba(alphaAxis[j]),
        marker='x',
        label = alphaAxis[j].data
      )
    
  plt.gca().set_xlabel(r"$w_k$")

  # Plot a colorbar to show value of alpha
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  scalarMap._A = []
  cbar = plt.colorbar(scalarMap, cax=cbar_ax, orientation='vertical')
  cbar_ax.set_title(r'$\alpha_j$')


# Show plots
plt.show()