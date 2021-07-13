#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

def plotRMS_samples (ts, refs: np.ndarray, vals: np.ndarray, taxis=1, **kwargs):
  assert refs.shape == vals.shape, "Shapes must be equal!"
  # Compute RMS for each vector component
  rms = np.sqrt(np.square(refs - vals))
  # Plot the rms
  plt.plot(ts, rms, kwargs)

def plotRMS_trials (Nper, refs:np.ndarray, vals:np.ndarray, kaxis=1):
  assert refs.shape == vals.shape, "Shapes must be equal!"
  # Put samples into bins
  N = refs.shape[1] # Number of samples
  M = np.ceil(float(N)/float(Nper)) # Number of trials
  Nper_last = Nper if np.remainder(N,Nper)==0 else np.remainer(N,Nper)

  ks_i = range(Nper)          # Sample indices (in trials)
  ks_last = range(Nper_last)  # Sample indices (in last trial)

  for i in range(M-1):
    kstart = i * Nper
    kend = (i+1) * Nper
    refs_i = refs[:,kstart:kend]
    vals_i = vals[:,kstart:kend]
    plotRMS_samples(ks, refs, vals)
    
  kstart = (M-1) * Nper
  refs_last = refs[:,kstart:]
  vals_last = vals[:,kstart:]
  plotRMS_samples(ks_last, refs_last, vals_last)

