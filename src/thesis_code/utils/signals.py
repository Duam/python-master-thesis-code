#!/usr/bin/python3

import numpy as np

################################## PULSES #########################################
def pulse_single(k:int, width:int=1, amp:float=1.0):
  """ Creates a single pulse signal at sample 'shift'.
  args:
    k[int] -- Sample index
    width[int] -- Pulse width in samples
    amp[float] -- Pulse amplitude
  returns:
    s[float] -- Signal value at sample k
  """
  return amp if (k >= 0 and k < width) else 0.0

def pulse_periodic(k:int, N:int, width:int=1, amp:float=1.0):
  """ Creates a N-periodic pulse signal that is 'width' samples wide and right-shifted by 'shift'.
  args:
    k[int] -- Sample index
    N[int] -- Samples per period
    width[int] -- Pulse width in samples
    amp[float] -- Pulse amplitude
  returns:
    s[float] -- Signal value at sample k
  """
  return pulse_single(np.mod(k,N), width, amp)

def pulse_FM_linear(k:int, N:int, width:int=1, amp:float=1.0):
  """ Creates a periodic pulse signal that doubles its frequency with each period. 
  The pulse is 'width' samples wide and its initial period is N samples wide. The signal is 0.0 for k < 0.
  args:
    k[int] -- Sample index
    N[int] -- Samples in initial period
    width[int] -- Pulse width in samples
    amp[float] -- Pulse amplitude
    shift[int] -- Starting sample index of the signal
  returns:
    s[float] -- Signal value at sample k
  """
  # Compute the number of periods that have passed up until sample k
  num_prev_periods = np.floor(k/float(N))
  # Halve the original period length for each previous period (minimum 1)
  N = max([np.floor(N/2**num_prev_periods),1])
  # Compute signal
  return pulse_periodic(k, N, width, amp) if k >= 0 else 0.0
  

################################## PERIODIC PRIMITIVES #########################################
def rectangle(k: int, N: int):
  """ Creates a rectangle signal.
  args:
    k[int] -- Sample index
    N[int] -- Samples per period
  returns:
    s[float] -- Signal value at sample k
  """
  return 1. if np.mod(k,N) < N/2. else 0.

def triangle(k:int, N:int):
  """ Creates a triangle signal.
  args:
    k[int] -- Sample index
    N[int] -- Samples per period
  returns:
    s[float] -- Signal value at sample k
  """
  kmodN = np.mod(k,N)
  return 2*kmodN/N if kmodN < N/2. else 2-2*kmodN/N

def sawtooth_periodic(k:int, N:int, amp:float=1.0, reversed:bool=False):
  return amp/float(N) * np.mod(-k if reversed else k,N)

def sawtooth_FM_linear(k:int, N:int, amp:float=1.0, reversed:bool=False):
  # Compute the number of periods that have passed up until sample k
  num_prev_periods = np.floor(k/float(N))
  # Halve the original period length for each previous period (minimum 1)
  N = max([np.floor(N/2**num_prev_periods),1])
  return sawtooth_periodic(k, N, amp, reversed) if k >= 0.0 else 0.0


def staircase_periodic(k:int, N:int, amp:float=1.0):
  return amp * np.floor((k+N)/float(N))

def staircase_FM_linear(k:int, N:int, amp:float=1.0):
  return staircase_periodic(k**2, N**2, amp) if k >= 0 else 0.0
  
################################## COMPOUND #########################################

def from_fourier(t: float, offs: float, amps: np.ndarray, freqs: np.ndarray, phases: np.ndarray = None):
  """ Creates a signal from amplitudes, frequencies and phases.
  args:
    t[float] -- Time
    offs[float] -- Signal offset
    amps[np.ndarray] -- Signal amplitude components
    freqs[np.ndarray] -- Signal frequency components
    phases[np.ndarray] -- Signal phase components (default None)
  returns:
    s[float] -- Signal value at time t
  """
  # Convert to numpy arrays if arguments are given as lists
  amps = np.array(amps)
  freqs = np.array(freqs)

  # Check dimensions and shapes
  assert amps.ndim == 1, "Amplitude array must be 1-dimensional!"
  assert freqs.ndim == 1, "Frequency array must be 1-dimensional!"
  assert amps.shape[0] == freqs.shape[0], "Amplitude and frequency arrays must be same length!"
  if phases != None:
    phases = np.array(phases)
    assert phases.ndim == 1, "Phase array must be 1-dimensional!"
    assert phases.shape[0] == amps.shape[0], "Amplitude and phase arrays must be same length!"
  else:
    phases = np.zeros(amps.shape)

  # Fetch number of signal components
  N = freqs.shape[0]

  # Compute signal
  s = offs
  for n in range(N):
    s += amps[n] * np.sin(2 * np.pi * freqs[n] * t + phases[n])

  return s
  

################################## COMPOUND (with internal state) #########################################

def decaying_spike_periodic(Ntotal:int, Nperiod:int, tau:float, amp:float=1.0, shift:int=0):
  """ Computes a periodic decaying spike signal trajectory. Starts at k >= shift
  args:
    Ntotal[int] -- Trajectory length in samples
    Nperiod[int] -- Periodicity of the signal
    tau[float] -- Decaying time constant (smaller -> faster decay)
    amp[float] -- Spike height
    shift[int] -- Trajectory shift
  returns:
    Periodic decaying spike trajectory of length Ntotal
  """
  # Create pulse trajectory
  us = [ pulse_periodic(k, Nperiod, amp=amp) for k in np.arange(shift,Ntotal+shift) ]

  # Create discrete PT1 element with varying gain
  def pt1_model(s,u):
    coeff = tau + 1.0
    K = coeff * (1 - s/amp) + s/amp
    return 1/coeff * (K * u - s) + s

  # Compute trajectory
  sig = np.zeros(Ntotal)
  for k in range(Ntotal-1):
    sig[k+1] = pt1_model(sig[k], us[k])
  
  return sig