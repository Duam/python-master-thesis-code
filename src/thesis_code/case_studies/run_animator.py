#!/usr/bin/python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Choose which study-case-run to plot using command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help=".nc file")
parser.add_argument("-i", "--study", help="Study number")
parser.add_argument("-j", "--case", help="Case number")
parser.add_argument("-lmin", "--min-sample", help="Starting sample number")
parser.add_argument("-lmax", "--max-sample", help="End sample number")
args = parser.parse_args()

filename = args.file
i = int(args.study)
j = int(args.case)
lmin = int(args.min_sample)
lmax = int(args.max_sample)


# Get the data
data = {}
#filename = 'case_study_data_logW_raw.nc'
try:
  data = xr.open_dataset(filename)
  print("Dataset opened. Content:")
  print(data)
except:
  print("Dataset \'" + filename + "\' could not be opened. Does it exist?")
  print("Aborting.")
  exit(1)


# Fetch relevant data
NUM_SAMPLES_TOTAL = data.attrs['N_per'] * (data.attrs['M_warmup'] + data.attrs['M_settle'] + data.attrs['M_conv'] + data.attrs['M_err'])
TRIAL_MARKERS = [l for l in range(NUM_SAMPLES_TOTAL) if np.mod(l,data.attrs['N_per']) == 0]

P_sim = data['parameters_sim']
P_est = data['parameters_est']

# Compute the parameter estimate error
E = np.sqrt(np.square(data['parameters_sim'] - data['parameters_est']))

# Description string for animation
def descr(k):
  s = "study " + str(i) + " case " + str(j)
  s += " run " + str(k)
  s += " from sample " + str(lmin) + " to sample " + str(lmax)
  return s

print("\nPlotting", descr)

# Plot the study-case-run
fig, ax = plt.subplots(nrows=2, ncols=1)
lAxis = np.arange(lmin,lmax)

plt.sca(ax[0])
plt.title(descr(0))
plt.ylabel("P simulated vs estimated")
plt.axhline(color='grey', linestyle='--')
[plt.axvline(l, color='g', linestyle='--') for l in TRIAL_MARKERS if l > lmin and l < lmax]
plt.plot(lAxis, P_sim[dict(case=j,sample=lAxis)].squeeze(), 'x--', color='red', label='P_sim', alpha=0.25)
pest_plot, = ax[0].plot(lAxis, P_est[dict(study=i,case=j,run=0,sample=lAxis)].squeeze(), 'x--', color='blue', label='P_est', alpha=0.5)
plt.legend(loc='best')

plt.sca(ax[1])
plt.grid()
plt.gca().set_yscale('log')
[plt.axvline(l, color='g', linestyle='--') for l in TRIAL_MARKERS if l > lmin and l < lmax]
ax[1].plot(lAxis, E[dict(study=i,case=j,run=0,sample=lAxis)].squeeze(), '--', color='red', alpha=0.25, label='w = 0')
err_plot, = ax[1].plot(lAxis, E[dict(study=i,case=j,run=0,sample=lAxis)].squeeze(), label='w = 0')
plt.xlabel('sample')
plt.ylabel('parameter estimation error')


ax[1].set_ylim([1e-10, 1e0])

def update(k):
  ax[0].set_title(descr(k))
  pest_plot.set_data(lAxis, P_est[dict(study=i,case=j,run=k,sample=lAxis)].squeeze())
  
  ax[1].set_title("w = " + str(data['weight_W'].data[k]))
  Ek = E[dict(study=i,case=j,run=k,sample=lAxis)].squeeze()

  err_plot.set_data(lAxis, Ek)
  err_plot.set_label('w = ' + str(data['weight_W'].data[k]))
  plt.legend()
  return err_plot,

ani = anim.FuncAnimation(fig, update, interval=500, frames=range(data.sizes['run']), blit=False)

plt.show()


