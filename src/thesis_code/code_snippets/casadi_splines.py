#!/usr/bin/python3

"""
Test CasADi splines. Test case: Lift/Drag-coefficients vs alpha
"""

import casadi as cas
import numpy as np
import matplotlib.pyplot as plt

# Lift-coefficient at AoA = 0
cl_0 = 1.0

# Lift-coefficients at positive and negative stall angle
delta_cl = 2.0
cl_max = cl_0 + delta_cl
cl_min = cl_0 - delta_cl

# Drag-coefficients
cl_over_cd = 100
cd_min = cl_0 / cl_over_cd
cd_st = cl_max / cl_over_cd
cd_max = 1.0

# Stall angle
AoA_stall = 25
AoA_stall *= 2*np.pi / 360

# Create spline 
cl_vals = np.array([
  [-np.pi-AoA_stall, -np.pi, -np.pi+AoA_stall,
         -AoA_stall,      0,        AoA_stall,
    np.pi-AoA_stall,  np.pi,  np.pi+AoA_stall]
])
#cl_vals = np.array([
#  [0, AoA_stall, np.pi / 2.0, np.pi-AoA_stall, np.pi, np.pi+AoA_stall, 3*np.pi/2, 2*np.pi-AoA_stall, 2*np.pi, 2*np.pi+AoA_stall],
#  [cl_0, cl_max, 0, cl_min, cl_0, cl_max, 0, cl_min, cl_0, cl_max]
#])

cd_vals = np.array([
  [0, AoA_stall, np.pi / 2.0, np.pi-AoA_stall, np.pi, np.pi+AoA_stall, 3*np.pi/2, 2*np.pi-AoA_stall, 2*np.pi, 2*np.pi+AoA_stall],
  [cd_min, cd_st, cd_max, cd_st, cd_min, cd_st, cd_max, cd_st, cd_min, cd_st]
])

cl = cas.interpolant('cl', 'bspline', [cl_vals[0,:]], cl_vals[1,:])
cd = cas.interpolant('cl', 'bspline', [cd_vals[0,:]], cd_vals[1,:])

alphaAxis = np.linspace(0,2*np.pi,40)



CL = cl(alphaAxis)
CD = cd(alphaAxis)

plt.figure()
plt.plot(alphaAxis, CL)
plt.plot(alphaAxis, CD)

plt.figure()
plt.plot(CL[:10], CD[:10])
plt.show()


alpha = cas.MX.sym('alpha')
cl_der = cas.Function('spl_der', [alpha], [cas.jacobian(spl(alpha),alpha)])
print(spl_der(AoA_stall))