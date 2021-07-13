#!/usr/bin/python3

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

NUM_STUDIES = 2
NUM_CASES = 2
NUM_RUNS = 2
NUM_CONTROL_SAMPLES = 3
NUM_CONTROLS = 1
NUM_STATE_SAMPLES = NUM_CONTROL_SAMPLES+1
NUM_STATES = 3

STUDIES = range(NUM_STUDIES)
CASES = range(NUM_CASES)
RUNS = range(NUM_RUNS)
CONTROL_SAMPLES = range(NUM_CONTROL_SAMPLES)
CONTROLS = range(NUM_CONTROLS)
STATE_SAMPLES = range(NUM_STATE_SAMPLES)
STATES = range(NUM_STATES)

u_np = np.zeros((NUM_CONTROL_SAMPLES,NUM_CONTROLS))
x_np = np.zeros((NUM_STUDIES,NUM_CASES,NUM_RUNS,NUM_STATE_SAMPLES,NUM_STATES))

u = xr.DataArray(
  u_np,
  coords = [
    ('sample', CONTROL_SAMPLES), 
    ('control', CONTROLS)
  ]
)
u.name = "Control"

#print(u)

x = xr.DataArray(
  x_np,
  coords = [
    ('study', STUDIES),
    ('case', CASES),
    ('run', RUNS),
    ('sample', STATE_SAMPLES),
    ('state', STATES)
  ]
)
x.name = "State estimate"

xsel = x.isel(study=0,case=0,run=0,state=0)
usel = u.isel(control=0)

#print(xsel)
#print(usel)

y = range(10)
dsy = xr.Dataset(
  data_vars = {
    'y': (['sample'], y)
  },
  coords = {
    'sample': range(10)
  }
)

#print(dsy)

ds = xr.Dataset(
  data_vars = {
    'x': (['study','case', 'run', 'sample', 'state'], x_np),
    'u': (['sample','control'], np.concatenate([u_np, np.array(np.nan).reshape(1,1)],axis=0))
  },
  coords = {
    'study': STUDIES,
    'case': CASES,
    'run': RUNS,
    'sample': STATE_SAMPLES,
    'state': STATES,
    'control': CONTROLS
  }
)
print(ds)

ds.to_netcdf('test.nc')

ds_disc = xr.open_dataset('test.nc')
print(ds_disc)

print("==============================")

print(ds_disc['u'])

import casadi as cas
x = cas.SX.sym('x',1,1)
u = cas.SX.sym('u',1,1)
xnext = 0.6 * x + u
F = cas.Function('F', [x,u], [xnext], ['x','u'], ['xnext'])

print(xnext)
