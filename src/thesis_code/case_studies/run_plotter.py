#!/usr/bin/python3

import xarray as xr
import numpy as np

# Choose which study-case-run to plot using command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help=".nc file")
parser.add_argument("-i", "--study", help="Study number")
parser.add_argument("-j", "--case", help="Case number")
parser.add_argument("-k", "--run", help="Run number")
parser.add_argument("-lmin", "--min-sample", help="Starting sample number")
parser.add_argument("-lmax", "--max-sample", help="End sample number")
args = parser.parse_args()

filename = args.file
i = int(args.study)
j = int(args.case)
k = int(args.run)
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
TRIAL_MARKERS = [k for k in range(NUM_SAMPLES_TOTAL) if np.mod(k,data.attrs['N_per']) == 0]

P_sim = data['parameters_sim']
P_est = data['parameters_est']

# Compute the parameter estimate error
E = np.sqrt(np.square(data['parameters_sim'] - data['parameters_est']))

# Plot the study-case-run
import matplotlib.pyplot as plt

descr = "study " + str(i) + " case " + str(j) + " run " + str(k) + " from sample " + str(lmin) + " to sample " + str(lmax)
print("\nPlotting", descr)

kAxis = np.arange(lmin,lmax)

fix, ax = plt.subplots(nrows=2, ncols=1)

plt.sca(ax[0])
plt.title(descr)
plt.ylabel("P simulated vs estimated")
plt.axhline(color='grey', linestyle='--')
[plt.axvline(k, color='g', linestyle='--') for k in TRIAL_MARKERS if k > lmin and k < lmax]
plt.plot(kAxis, P_sim[dict(case=j,sample=kAxis)].squeeze(), 'x--', color='red', label='P_sim', alpha=0.25)
plt.plot(kAxis, P_est[dict(study=i,case=j,run=k,sample=kAxis)].squeeze(), 'x--', color='blue', label='P_est', alpha=0.5)
plt.legend(loc='best')

plt.sca(ax[1])
plt.grid()
plt.gca().set_yscale('log')
[plt.axvline(k, color='g', linestyle='--') for k in TRIAL_MARKERS if k > lmin and k < lmax]
plt.plot(kAxis, E[dict(study=i,case=j,run=k,sample=kAxis)].squeeze(), label='w='+str(data['weight_W'].data[k]))
plt.plot(kAxis, E[dict(study=i,case=j,run=0,sample=kAxis)].squeeze(), '--', color='red', alpha=0.25, label='w='+str(data['weight_W'].data[0]))
plt.xlabel('sample')
plt.ylabel('parameter estimation error')
plt.legend()
plt.show()

#print(xAxis)

