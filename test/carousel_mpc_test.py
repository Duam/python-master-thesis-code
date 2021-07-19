import numpy as np
import casadi as cas
from thesis_code.carousel_model import CarouselWhiteBoxModel
import thesis_code.models.carousel_whitebox_viz as viz
from thesis_code.carousel_simulator import Carousel_Simulator
from thesis_code.components.carousel_mpc import Carousel_MPC
from thesis_code.components.carousel_tarsel import Carousel_TargetSelector
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)
cas.GlobalOptions.setMaxNumDir(64)

# Create default model
param = CarouselWhiteBoxModel.getDefaultParams()
model = CarouselWhiteBoxModel(param)
NX = model.NX() - 2
NZ = model.NZ()
NY = model.NY()
NU = model.NU() - 1
NP = model.NP()
x0,z0,u0 = model.get_steady_state()

# Time per revolution
T_per = 2*np.pi / abs(param['carousel_speed'])

# Simulation meta-parameters
M_sim = 20
T_sim = M_sim * T_per
dt_sim = 1. / 50.
#dt_sim = 1. / 10.
N_sim = int(np.ceil(T_sim / dt_sim))
N_sim = 50
#dt_sim = T_sim / float(N_sim)

# Target selector parameters
#N_ts = int(N_sim / float(M_sim))
N_ts = 20
dt_ts = T_per / float(N_ts)
#ref_fun = lambda psi: x0[0]
ref_fun = lambda psi: x0[0] + 15 * (2*np.pi/360) * np.sin(psi)

# Controller parameters
N_ctr = 1
jit = False
expand = True

""" ================================================================================================================ """
# Create a target selector and create a reference trajectory
target_selector = Carousel_TargetSelector(model = model, N = N_ts, reference = ref_fun)
target_selector.init(x0 = x0, z0 = z0, W = cas.DM.eye(1))
Xref_orig, Zref_orig, Uref_orig = target_selector.call()

# Create a controller
controller = Carousel_MPC(
  model = model,
  N = N_ctr,
  dt = dt_sim,
  verbose = True,
  do_compile = jit,
  expand = expand
)

Q_mpc = np.eye((NX))
R_mpc = np.eye((NU))
S_mpc = np.eye((NX))

Q_mpc = 1e-3 * np.ones((NX,1))
Q_mpc[0] = 1e-1
Q_mpc[1] = 1e-1
Q_mpc[2] = 1e0
Q_mpc[3] = 1e0
R_mpc = 1e-3 * np.ones((NU,1))
S_mpc = Q_mpc
S_mpc[0] = 1e1
S_mpc[1] = 1e1
S_mpc[2] = 1e-1
S_mpc[3] = 1e-1


# We overwrite x,z,u in order to initialize the MPC directly on the trajectory
ref_init = target_selector.get_new_reference(0.0, dt_sim, N_ctr)
x0 = ref_init[0][:,0]
z0 = ref_init[1][:,0]
Uref_init = ref_init[2][:,:-1]

print("x0 =", x0)
print("z0 =", z0)
print("Uref =", Uref_init)

x0,z0,u0 = model.get_steady_state() 
Uref_init = cas.horzcat(*[u0 for k in range(Uref_init.shape[1])])

print("OVERWRITE!")
print("x0 =", x0)
print("z0 =", z0)
print("Uref =", Uref_init)

x0 = cas.vertcat(x0[0], x0[1], x0[3], x0[4], x0[6])
Uref_init = Uref_init[0,:]
controller.init(
  x0 = x0, #z0 = z0,
  Q = Q_mpc, R = R_mpc, S = S_mpc,
  Uref = Uref_init
)

""" ======================================= """


print(controller.ocp)
w0 = controller.initial_guess
p0 = controller.parameters

#g_eval = controller.ocp.eval_g(w0,p0)

#print("COLL = ", controller.G(w0['X'][:,0], w0['Vx'], w0['Vz'], cas.vertcat(p0['P'],w0['U']))[1])

w0_eval = controller.ocp.struct_w(w0)
print("w0 =")
for key in w0_eval.keys():
  print(key, " = ", w0_eval[key])
"""
f_eval = controller.ocp.f_fun(w0,p0)
print("f(w0,p0) evaluated =", f_eval)

g_eval = controller.ocp.eval_g(w0,p0)
print("g(w0,p0) evaluated =")
for key in g_eval.keys():
  print(key, " = ", g_eval[key])
"""
"""
h_eval = controller.ocp.eval_h(w0,p0)
print("h(w0,p0) evaluated =")
for key in h_eval.keys():
  print(key, " = ", h_eval[key])
"""

#print("cost =", f_eval)
#print("inf_pr =", max([ np.sqrt(g_eval.cat[k]**2) for k in range(g_eval.cat.shape[0]) ]))

""" ======================================= """
#quit(0)


print("=======================================")
print("Doing one sample solve(), which should converge in one step..")
result = controller.ocp.solve(w0,p0) # Should converge immediately!
print(result[1]['iter_count'])
"""
print("\n\nResult\n\n")
w_k = result[0]['w']
lam_g_k = result[0]['lam_g']
eig = controller.ocp.eval_expanded_eigvecs(w_k,lam_g_k,p0)
for elem in eig:
  print("===========================================================================")
  print("The eigenvalue shows the amount of curvature in the direction of the eigenvector.")
  print("Each eigenvalue-eigenvector pair is one effective degree of freedom in the NLP.")
  print("\neigvec_exp = Z * eigvec(Z.T * hess(lag) * Z), with Z = nullspace(jac(g))\n")
  print("Eigenvalue: ", elem[0])
  print("Eigenvector (expanded into decision space): \n")
  for key in elem[1].keys():
    print(key, " = ", elem[1][key])
  print("\n\n")

"""
w1 = result[0]['w']
X1 = w1['X']
#Z1 = w1['Z']
Vx1 = w1['Vx']
Vz1 = w1['Vz']
U1 = w1['U']

print("dX =", X1-w0['X'])
#print("dZ =", Z1-w0['Z'])
print("dVx =", Vx1-w0['Vx'])
print("dVx =", Vz1-w0['Vz'])
print("dU =", U1-w0['U'])

#assert result[1]['iter_count'] == 0, "Test solve() should converge in zero steps!"
#print(result)
#quit(0)


plt_x_prior = False
if plt_x_prior == True:
  viz.plotStates_withRef(
    Xs = w0['X'], Xs_ref = Xref_init,#target_selector.get_new_reference(x0[2], dt_sim, N_ctr)[0],
    Us = w0['U'], Us_ref = Uref_init,#target_selector.get_new_reference(x0[2], dt_sim, N_ctr)[2],
    dt = dt_sim, model = model
  )
  plt.show()
  quit(0)



""" ================================================================================================================ """

# Some sample calls
x0_sim = cas.vertcat(x0[0], x0[1], 0.0, x0[2], x0[3], -2., x0[4])
Xs_sim = [ cas.DM(x0_sim) ]
Zs_sim = [ cas.DM(z0) ]
Us_sim = [ ]
Xs_ref = [ cas.DM(x0) ]
Zs_ref = [ cas.DM(z0) ]
Us_ref = [ ]

# MPC solve information
F = [ ]
G = [ ]
H = [ ]
LAM_G = [ ]
LAM_H = [ ]

# Create simulator
simulator = Carousel_Simulator(
  model = model, x0 = Xs_sim[0], z0 = Zs_sim[0],
  process_noise_mean = 0e0 * np.ones((NX+2)),
  process_noise_covar = 0e0 * np.eye((NX+2)),
  measurement_noise_mean = 0e0 * np.ones((NY)),
  measurement_noise_covar = 0e0 * np.eye((NY)),
  jit = jit
)

solve_time = []
iter_count = []
print("Starting simulation. Initial conditions:")
print("x0 =", Xs_sim[0])
print("z0 =", Zs_sim[0])
for k in range(N_sim):
  print(" =========================", k+1, " of", N_sim, " ===========================")
  # Fetch data
  x0_k = Xs_sim[-1]
  z0_k = Zs_sim[-1]

  print("Current state:")
  print("x0 =", x0_k)
  print("z0 =", z0_k)

  # Get new reference, starting from current yaw angle
  Xref, Zref, Uref = target_selector.get_new_reference(x0_k[2], dt_sim, N_ctr)

  # Fetch MPC input
  x0_k_in = cas.vertcat(x0_k[0], x0_k[1], x0_k[3], x0_k[4], x0_k[5])
  Uref_in = cas.horzcat(Uref[0,:-1]).T
  Xref_in = cas.horzcat(Xref[0,:], Xref[1,:], Xref[3,:], Xref[4,:], Xref[6,:]).T

  print("MPC input:")
  print("x0 =", x0_k_in)
  print("Xref =", Xref_in)
  print("Uref =", Uref_in)

  # Print out g
  """
  g_eval = controller.ocp.eval_g(controller.initial_guess,p0)
  print("g(wk,p0) evaluated =")
  for key in g_eval.keys():
    print(key, " = ", g_eval[key])
  """

  # Compute new control
  u0_k, ctrl_result, ctrl_stats, w0, p0 = controller.call(x0_k_in, Xref_in, Uref_in)

  # Simulate one step
  print("u(",k,") = ", u0_k[0], " (uref =", Uref[0,0], ")")
  xf_k, zf_k, y0_k = simulator.simstep(u0_k[0], dt_sim)

  print("New state:")
  print(xf_k)

  # Store states
  Xs_sim += [ xf_k ]
  Zs_sim += [ zf_k ]
  Us_sim += [ u0_k ]
  Xs_ref += [ Xref[:,1] ]
  Zs_ref += [ Zref[:,1] ]
  Us_ref += [ Uref[:,0] ]

  # Store solve() info
  F += [ ctrl_result['f'] ]
  G += [ ctrl_result['g'] ]
  H += [ ctrl_result['h'] ]
  LAM_G += [ ctrl_result['lam_g'] ]
  LAM_H += [ ctrl_result['lam_h'] ]

  #print(ctrl_stats['t_proc_hess_nlp'])
  solve_time += [ ctrl_stats['t_proc_'+controller.ocp.name] ]
  iter_count += [ ctrl_stats['iter_count']]
  #print("obj (0) =", ctrl_stats['iterations']['obj'][0], " -> ", ctrl_stats['iterations']['obj'][-1])
  #print("inf_pr (0) =", ctrl_stats['iterations']['inf_pr'][0], " -> ", ctrl_stats['iterations']['inf_pr'][-1])
  #print("f0 =", )
  print("MPC solve time: ", solve_time[-1]*1e3, "ms")
  print("MPC iter count: ", iter_count[-1])
  if ctrl_stats['return_status'] == 'Infeasible_Problem_Detected':
    print("ABORT. INFEASIBLE PROBLEM DETECTED.")
    break

  #if k == 1: break

avg_solve_time = sum(solve_time) / float(N_sim)
avg_iter_count = sum(iter_count) / float(N_sim)
print("============================================================================================")
#print("Total time in MPC solve():", total_call_time)
print("Average MPC solve() time:", avg_solve_time)
print("Average iter count: ", avg_iter_count)
"""
print("LAM_G:")
for k in range(len(LAM_G)):
  print("FOR STEP k=",k)
  lam = LAM_G[k]
  for key in lam.keys():
    print(key, " =", lam[key])

print("LAM_H:")
for lam in LAM_H:
  print(lam.cat)
"""
xs     = cas.horzcat(*Xs_sim)
xs_ref = cas.horzcat(*Xs_ref)
us     = cas.horzcat(*Us_sim)
us_ref = cas.horzcat(*Us_ref)

viz.plotStates_withRef(
  Xs = xs, Xs_ref = xs_ref,
  Us = us, Us_ref = us_ref,
  dt = dt_sim, model = model
)

if True:
  print("TODO: Plot f,g,h,lam_g,lam_h for more information!")
  pass

plt.show()
