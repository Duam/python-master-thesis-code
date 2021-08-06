#!/usr/bin/python3

# Tools
import time, datetime, argparse, pprint, json
from models.carousel_whitebox.carousel_whitebox import CarouselWhiteBoxModel
from utils.zcm_constants import carousel_zcm_constants as carzcm
from utils.bcolors import bcolors
import numpy as np
import pandas as pd
import scipy as sc
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Fetch params
model_constants = CarouselWhiteBoxModel.getConstants()
offsets = {
    'VE1_SCAPULA_ELEVATION': model_constants['roll_sensor_offset'],
    'VE1_SCAPULA_ROTATION': model_constants['pitch_sensor_offset']
}

# Today's date as a string
today = datetime.datetime.now().strftime("%Y_%m_%d")

# Parse arguments:
parser = argparse.ArgumentParser(description='Trajectory data plotter')
parser.add_argument(
  "-s", "--signal-type",
  dest='signal_type',
  help="Actuation signal type. May be \"SS\", \"RECT\", or \"SINE\""
)

# Fetch arguments
args = parser.parse_args()

signal_type = "UNDEFINED"
if args.signal_type == "SS":
  signal_type = "STEADY_STATE"
elif args.signal_type == "RECT":
  signal_type = "RECT"
elif args.signal_type == "SINE":
  signal_type = "SINE"

# Get file prefix
dataset_path = "./CAROUSEL_ZCMLOG_2019_09_13_" + signal_type + "_CONTROL.zcmlog.PREPROCESSED.csv"

def elevation_toState(y):
  return - y + offsets['VE1_SCAPULA_ELEVATION']

def pitch_toState(y):
  return - y + offsets['VE1_SCAPULA_ROTATION']

def toDeg(rad):
  return rad * 360 / (2*np.pi)

# Load data
print("Loading dataset " + dataset_path)
data = pd.read_csv(dataset_path, header=0, parse_dates=[0])
data['VE1_SCAPULA_ELEVATION_0'] = data['VE1_SCAPULA_ELEVATION_0'].apply(elevation_toState).apply(toDeg)
data['VE1_SCAPULA_ROTATION_0'] = data['VE1_SCAPULA_ROTATION_0'].apply(pitch_toState).apply(toDeg)

# We only plot the first N samples
dt = 0.05
T = 30
N = len(data)
#N = min([int(T/dt), len(data)])
tAxis = (data['timestamp'] - data['timestamp'][0]).timestep.total_seconds()[:N]

# Compute some statistics
ctrl = data['VE2_KIN4_SET_0'][:N]
ctrl_fft = abs(sc.fft(ctrl))
ctrl_freqs = fftpack.fftfreq(len(ctrl), 0.05)
num_ctrl_freqs = int(len(ctrl_freqs) / 2.)

#roll = sc.pi - data['VE1_SCAPULA_ELEVATION_0'][:N]
roll = data['VE1_SCAPULA_ELEVATION_0']
roll_mean = roll.mean()
roll_variance = roll.var()
roll_fft = abs(sc.fft(roll)) ** 2 # Compute Power Spectral Density
roll_freqs = fftpack.fftfreq(len(roll_fft), 0.05)
num_roll_freqs = int(len(roll_freqs) / 2.)

#pitch = 0.9265 - data['VE1_SCAPULA_ROTATION_0'][:N]
pitch = data['VE1_SCAPULA_ROTATION_0']
pitch_mean = pitch.mean()
pitch_variance = pitch.var()
pitch_fft = abs(sc.fft(pitch))
pitch_freqs = fftpack.fftfreq(len(pitch), 0.05)
num_pitch_freqs = int(len(pitch_freqs) / 2.)


print("Roll mean: " + str(roll_mean))
print("Roll variance: " + str(roll_variance))
print("Pitch mean: " + str(pitch_mean))
print("Pitch variance: " + str(pitch_variance))

N = min([int(T/dt), len(data)])
tAxis = tAxis[:N]
ctrl = ctrl[:N]
roll = roll[:N]
pitch = pitch[:N]

filename = "dataset_" + args.signal_type + ".pdf"

with PdfPages("../../../tex/thesis/figures/identification_imu/" + filename) as pdf:
  # Plot data
  print("Plotting dataset..")
  fig, ax = plt.subplots(1,1,figsize=(5,5))
  plt.plot(tAxis, ctrl)
  plt.ylabel('Control $u(t)$')
  plt.xlabel('Time $t$ [s]')
  pdf.savefig()

  fig, ax = plt.subplots(1,1,figsize=(5,5))
  plt.plot(roll_freqs[:num_ctrl_freqs], 10*sc.log10(ctrl_fft)[:num_ctrl_freqs])
  plt.ylabel('PSD$\{u(t)\}$ [dB]')
  plt.xlabel('Frequency [Hz]')
  pdf.savefig()

  fig, ax = plt.subplots(1,1,figsize=(5,5))
  plt.plot(tAxis, roll)
  plt.ylabel('Elevation $\phi(t)$ [deg]')
  plt.xlabel('Time $t$ [s]')
  pdf.savefig()

  fig, ax = plt.subplots(1,1,figsize=(5,5))
  plt.plot(roll_freqs[:num_roll_freqs], 10*sc.log10(roll_fft)[:num_roll_freqs])
  plt.ylabel('PSD$\{\phi(t)\}$ [dB]')
  plt.xlabel('Frequency [Hz]')
  pdf.savefig()

  fig, ax = plt.subplots(1,1,figsize=(5,5))
  plt.plot(tAxis, pitch)
  plt.ylabel('Pitch $\\theta(t)$ [deg]')
  plt.xlabel('Time $t$ [s]')
  pdf.savefig()

  fig, ax = plt.subplots(1,1,figsize=(5,5))
  plt.plot(pitch_freqs[:num_pitch_freqs], 10*sc.log10(pitch_fft)[:num_pitch_freqs])
  plt.ylabel('PSD$\{\\theta(t)\}$ [dB]')
  plt.xlabel('Frequency [Hz]')
  pdf.savefig()

  plt.close()

quit(0)
