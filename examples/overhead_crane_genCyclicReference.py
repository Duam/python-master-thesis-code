import casadi as cas
import numpy as np
from thesis_code.integrators.rk4step import rk4step
from thesis_code.models.overhead_crane import OverheadCrane

# Create a model
plant = OverheadCrane()

# Create system model in casadi
x = cas.MX.sym('x', 6, 1)
u = cas.MX.sym('u', 2, 1)
ode = cas.Function('ode', [x,u], [plant.ode(x,u)], ['x','u'], ['xdot'])
out = cas.Function('out', [x,u], [plant.output(x,u)], ['x','u'], ['y'])

def computeReference(ref, N, T):
  from target_selectors.RefGenNLP import RefGenNLP
  from scipy import interpolate

  # Simulation parameters
  dt = T/N

  # Create integrator
  nn = 10
  h = dt / float(nn)
  Xk = x
  for k in range(nn):
    Xk = rk4step(ode, Xk, u, h)

  F = cas.Function('F', [x,u], [Xk], ['x','u'], ['xnext'])

  ##################################
  ### Create an output reference ###

  # Create the spline
  ts_node = np.linspace(0, T, ref.shape[1])
  x_spline = interpolate.CubicSpline(ts_node, ref[0,:], bc_type='periodic')
  y_spline = interpolate.CubicSpline(ts_node, ref[1,:], bc_type='periodic')

  # Evaluate the spline
  tAxis = np.arange(0, T, dt)
  yref_x = x_spline(tAxis)
  yref_y = y_spline(tAxis)
  yref = np.array([yref_x, yref_y])

  # Set up reference generator
  gen = RefGenNLP(F, out, N, 6, 2, 2)

  # Add equality constraints
  x0_dec = gen.nlp.get_decision_var('X')[:,0]
  xN_dec = gen.nlp.get_decision_var('X')[:,N]
  gen.add_equality('cycle_x', xN_dec - x0_dec )

  # Initialize the reference generator
  Q = 1e0 * cas.DM.eye(2)
  R = 1e0 * cas.DM.eye(2)
  gen.init(Q, R)

  # Run 
  xref, uref, stats, result = gen.run(yref, plant.x0)
  return yref, xref, uref, stats, result


if __name__ == '__main__':
  '''  Test the reference generation  '''
  from models.overhead_crane.overhead_crane_viz import OverheadCrane_Visualizer
  import matplotlib.pyplot as plt

  print("This script tests the reference generator for the overhead-crane model")


  # Parameters
  T = 200
  N = 50
  dt = T/float(N)

  # Create integrator
  nn = 10
  h = dt / float(nn)
  Xk = x
  for k in range(nn):
    Xk = rk4step(ode, Xk, u, h)

  F = cas.Function('F', [x,u], [Xk], ['x','u'], ['xnext'])

  # Some reference points
  ref = np.array([
    [0.0, -1.0],
    [0.5, -2.0],
    [-0.5, -2.0],
    [0.0, -1.0]
  ]).T

  # Compute the reference 
  yref, xref, uref, stats, result = computeReference(ref, N, T)

  # Print out some stuff
  print('u0 :', uref[:,0])
  print('u1 :', uref[:,1])
  print('xref shape: ', xref.shape)
  print('uref shape: ', uref.shape)

  # Print out init & cycle constraint multipliers
  #print('lam_g (init):' + str(result['lam_g']['init']))
  #print('lam_g (cycle):' + str(result['lam_g']['cycle_x']))
  print('lam_g (shoot N-2):' + str(result['lam_g']['shoot_'+str(N-2)]))
  print('lam_g (shoot N-1):' + str(result['lam_g']['shoot_'+str(N-1)]))

  # Retrieve shooting multipliers
  lam_shooting = np.zeros((6,N))
  for k in range(N):
    lam_shooting[:,k] = result['lam_g']['shoot_'+str(k)].full().flatten()


  # Plot shooting multipliers
  nAxis = range(N)
  multiplier_fig = plt.figure('Shooting and states')
  plt.subplot(6,2,1)
  plt.plot(nAxis, lam_shooting[0,:])
  plt.title('Shooting multipliers')
  plt.ylabel('lam_0')
  plt.subplot(6,2,3)
  plt.plot(nAxis, lam_shooting[1,:])
  plt.ylabel('lam_1')
  plt.subplot(6,2,5)
  plt.plot(nAxis, lam_shooting[2,:])
  plt.ylabel('lam_2')
  plt.subplot(6,2,7)
  plt.plot(nAxis, lam_shooting[3,:])
  plt.ylabel('lam_3')
  plt.subplot(6,2,9)
  plt.plot(nAxis, lam_shooting[4,:])
  plt.ylabel('lam_4')
  plt.subplot(6,2,11)
  plt.plot(nAxis, lam_shooting[5,:])
  plt.ylabel('lam_5')
  plt.xlabel('Stage')


  # Simulate
  #uref = np.zeros((2,N))
  xsim = np.zeros((6,N+1))
  xsim[:,0] = xref[:,0].flatten()
  for k in range(N):
    xsim[:,k+1] = F(xsim[:,k], uref[:,k]).full().flatten()
    #print(uref[:,k])
    if xsim[2,k] == 0.0:
      raise ValueError('Zero cable length found at iteration', k)
    for i in range(6):
      if np.isnan(xsim[i,k]) or np.isinf(xsim[i,k]):
        #print('xsim[:,k-1] =', xsim[:,k-1])
        #print('xsim[:,k] =', xsim[:,k])
        #print('xsim[:,k+1] =', xsim[:,k+1])
        #print('uref[:,k] =', uref[:,k])
        raise ValueError('NaN/inf encountered in iteration', k, ', field', i)

  nAxis = range(N+1)
  plt.subplot(6,2,2)
  plt.plot(nAxis, xref[0,:], '-')
  plt.plot(nAxis, xsim[0,:])
  plt.title('States')
  plt.ylabel('x0')
  plt.subplot(6,2,4)
  plt.plot(nAxis, xref[1,:], '-')
  plt.plot(nAxis, xsim[1,:])
  plt.ylabel('x1')
  plt.subplot(6,2,6)
  plt.plot(nAxis, xref[2,:], '-')
  plt.plot(nAxis, xsim[2,:])
  plt.ylabel('x2')
  plt.subplot(6,2,8)
  plt.plot(nAxis, xref[3,:], '-')
  plt.plot(nAxis, xsim[3,:])
  plt.ylabel('x3')
  plt.subplot(6,2,10)
  plt.plot(nAxis, xref[4,:], '-')
  plt.plot(nAxis, xsim[4,:])
  plt.ylabel('x4')
  plt.subplot(6,2,12)
  plt.plot(nAxis, xref[5,:], '-')
  plt.plot(nAxis, xsim[5,:])
  plt.ylabel('x5')
  plt.xlabel('Stage')

  nAxis = range(N)
  ctrl_fig = plt.figure()
  plt.subplot(211)
  plt.plot(nAxis, uref[0,:])
  plt.title('Controls')
  plt.ylabel('u0')
  plt.subplot(212)
  plt.plot(nAxis, uref[1,:])
  plt.ylabel('u1')
  plt.show()


  #print(uref)
  #print(xref)

  #print(xsim.shape)
  #print(xref[:N-1].shape)
  #print(uref.shape)
  #print(stats)
  #print(uref)
  #print(xsim)

  #for k in range(10):
  #  for i in range(6):
      #print(xref[i,k], '=?=', xsim[i,k])

  # Create a visualizer and run it
  dt_viz = 0.01
  viz = OverheadCrane_Visualizer(xsim[:,:N-1], yref.T, dt_viz)
  #viz = OverheadCrane_Visualizer(xref[:,:N-1], yref.T, dt_viz)
  anim = viz.createAnimation()
  #anim.save("overhead_crane.mp4", fps=1/dt_viz)
  plt.show()


  #import matplotlib.pyplot as plt
  #tAxis = np.linspace(0,T+dt,N)
  #fig = plt.figure()
  #plt.plot(tAxis, xsim[0,:])
  #plt.plot(tAxis, xsim[1,:])
  #plt.plot(tAxis, xsim[2,:])
  #plt.plot(tAxis, xsim[3,:])
  #plt.plot(tAxis, xsim[4,:])
  #plt.plot(tAxis, xsim[5,:])

