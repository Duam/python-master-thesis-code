#!/usr/bin/python3

# Tools
from copy import deepcopy
import sys, getopt, signal
from utils.signals import rectangle
from utils.bcolors import bcolors
import time, datetime
import matplotlib.pyplot as plt
import numpy as np
import casadi as cas
import argparse
import json, csv
from carousel_identification.zcm_constants import carousel_zcm_constants

# Today's date as a string
today = datetime.datetime.now().strftime("%Y_%m_%d")

# Parse arguments:
parser = argparse.ArgumentParser(description='ZCM_collect_idle_data')
parser.add_argument('-v', '--virtual_experiment', dest='is_virtual', default='true', help="Flag if this is a virtual_experiment experiment")
parser.add_argument('-d', '--date', dest='date', default=today, help="Date of the dataset formatted as YEAR_MONTH_DAY")
args = parser.parse_args()

# Set file prefix
is_virtual = args.is_virtual
file_prefix = carousel_zcm_constants.file_prefix_virtual if is_virtual else carousel_zcm_constants.file_prefix_physical
file_prefix += "_" + args.date
print("Loading dataset " + file_prefix)

""" ================================================================================================================ """

# Load data
roll = []
pitch = []
yaw = []
acc = []
gyr = []
roll_filename = file_prefix + "_ROLL.csv"
pitch_filename = file_prefix + "_PITCH.csv"
yaw_filename = file_prefix + "_YAW.csv"
acc_filename = file_prefix + "_LINEAR_ACCELERATION.csv"
gyr_filename = file_prefix + "_ANGULAR_VELOCITY.csv"

with open(roll_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      roll += [ float(row[1]) ]

with open(pitch_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      pitch += [ float(row[1]) ]

with open(yaw_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      yaw += [ float(row[1]) ]

with open(acc_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      acc += [ [float(row[1]),float(row[2]),float(row[3])] ]

with open(gyr_filename) as file:
  reader = csv.reader(file, delimiter=',')
  first_line = True
  for row in reader:
    if first_line: first_line = False
    else:
      gyr += [ [float(row[1]),float(row[2]),float(row[3])] ]

""" ================================================================================================================ """

# Compute sample mean of measurements
roll_mean = np.mean(roll)
pitch_mean = np.mean(pitch)
yaw_mean = np.mean(yaw)
acc_mean = np.mean(acc, axis=0)
gyr_mean = np.mean(gyr, axis=0)

# Compute sample variance of scalar measurements
roll_var = np.var(roll, ddof=1)
pitch_var = np.var(pitch, ddof=1)
yaw_var = np.var(yaw, ddof=1)
acc_var = np.var(acc, axis=0, ddof=1)
gyr_var = np.var(gyr, axis=0, ddof=1)

print("RESULTS:")
print("Roll: mean = " + str(roll_mean) + ", variance = " + str(roll_var))
print("Pitch: mean = " + str(pitch_mean) + ", variance = " + str(pitch_var))
print("Yaw: mean = " + str(yaw_mean) + ", variance = " + str(yaw_var))
print("Acc: mean = " + str(acc_mean) + ", variance = " + str(acc_var))
print("Gyr: mean = " + str(gyr_mean) + ", variance = " + str(gyr_var))
