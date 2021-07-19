#!/usr/bin/python3

import xarray as xr
import numpy as np
import pandas as pd

N = 8

k_ax = [k for k in range(N)]

x_dat = 2 * np.array(k_ax)
print(x_dat)

x = xr.Dataset(
  data_vars = {
    'x': (['k'], x_dat)
  },
  coords = {
    'k': k_ax
  }
)

k = [k for k in range(int(N/2))]
i = [k for k in range(2)]

ind = pd.MultiIndex.from_product((i,k),names=['i','new_k'])

x = x.assign(k=ind).unstack('k')
x = x.rename({'new_k':'k'})

print(x)
print(x[dict(k=1,i=1)])
