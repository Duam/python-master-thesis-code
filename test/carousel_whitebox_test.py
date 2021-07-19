import casadi as cas
import numpy as np
from thesis_code.utils.bcolors import bcolors
import pprint
import matplotlib.pyplot as plt
from thesis_code.carousel_model import CarouselWhiteBoxModel
import thesis_code.utils.signals as signals

np.set_printoptions(linewidth=np.inf)


# Parameterize
params = CarouselWhiteBoxModel.getDefaultParams()

# Create a carousel model
model = CarouselWhiteBoxModel(params)

# Print parameters
pprint.pprint(model.params)

# The semi-explicit DAE
dae = model.dae

# All symbolics
x_sym = cas.vertcat(*dae.x)
z_sym = cas.vertcat(*dae.z)
u_sym = cas.vertcat(*dae.u)
p_sym = cas.vertcat(*dae.p)
ode_sym = cas.vertcat(*dae.ode)
alg_sym = cas.vertcat(*dae.alg)
quad_sym = cas.vertcat(*dae.quad)

""" -------------------------------------------------- Test DAE ---------------------------------------------------- """
# Fetch initial state and control
u0 = model.u0()
x0 = model.x0()
#x0[0] = 0.0

# Set simulation parameters
T_settle = 2.*np.pi
T_test = 4.*np.pi
T = T_settle + T_test
N = 200
dt = T/float(N)

# Create a control trajectory
N_settle = int(T_settle / T * N)
N_per = 20
Usim = np.repeat(u0, N, axis=1)
Usim[0,N_settle:] = [ 0.5+0.1745*signals.rectangle(k, N_per) for k in range(N-N_settle) ]

# Prepare simulation containers
Zsim = np.zeros((model.NZ(),N+1))
Zsim[:,0] = 0
Xsim      = np.zeros((model.NX(),N+1))
Xsim[:,0] = x0.full().flatten()
Xdotsim = np.zeros((model.NX(),N+1))

p = np.concatenate([np.array(val).flatten() for val in params.values()])


# Create integrator
dae_dict = {'x': x_sym, 'ode': ode_sym, 'alg': alg_sym, 'z': z_sym, 'p': cas.vertcat(p_sym, u_sym)}
int_opts = {'number_of_finite_elements': 1, 'output_t0': True, 'tf':dt}
integrator = cas.integrator('xnext', 'collocation', dae_dict, int_opts)

# Simulate!
for k in range(N):
  print("Simulate step", k+1, " of", N)

  # Fetch controls, state
  u = Usim[:,k]
  x = Xsim[:,k]
  z = Zsim[:,k]

  # Compute current state derivative and algebraic stuff
  ode,alg = model.f(x,z,u,p)
  xdot = ode.full().flatten()
  Xdotsim[:,k] = xdot

  # Fetch generalized coordinates
  q = x[:3]
  qdot = x[4:]
  qddot = xdot[4:]

  # Simulate one step forward
  #step = model.simstep(x,u,dt,z0=z)
  step = integrator(x0=x,p=cas.vertcat(p,u))
  xnext = step['xf'][:,1]
  znext = step['zf'][:,1]
  Xsim[:,k+1] = xnext.full().flatten()
  Zsim[:,k+1] = znext.full().flatten()
  
  """
  print(bcolors.FAIL, "##########################################################################", bcolors.ENDC)
  print(bcolors.WARNING, "MISC ============================================================", bcolors.ENDC)
  print("t =", (k)*dt)
  print('x =', x)
  print("z =", z)
  print("ode = ", ode)
  print("alg = ", alg)
  
  print(bcolors.OKBLUE, "POS + VEL ===========================================", bcolors.ENDC)
  print("COM pos =", model.pos_COM(x), ", COM vel =", model.vel_COM(x))
  print("cs4 pos =", model.pos_cs4(x), ", cs4 vel =", model.vel_cs4(x))
  print("cs3 pos =", model.pos_cs3(x), ", cs3 vel =", model.vel_cs3(x))
  print("ACA pos =", model.pos_ACA(x), ", ACA vel =", model.vel_ACA(x))
  print("ACE pos =", model.pos_ACE(x), ", ACE vel =", model.vel_ACE(x))
  
  print("omega =", model.rotateView(x,4,1,model.omega(x)))
  print("omega wrt 4 =", model.omega(x))
  
  print("===1 ", cas.mtimes([model.I_COM, model.omega(x)]))
  print("===2 ", model.params['m'] * cross_product(model.pos_COM(x)-model.pos_cs3(x),model.vel_COM(x)))
  
  AileronData = model.computeAileronAerodynamicsData(x)
  ElevatorData = model.computeElevatorAerodynamicsData(x)
  print(bcolors.OKGREEN, "FORCES =================================", bcolors.ENDC)
  print("ElevatorData =", ElevatorData)
  print("AileronData =", AileronData)
  print('AoA_Aileron =', AileronData['AoA'])
  print("Aileron Speed =", np.linalg.norm(model.vel_ACA(x)))
  print("F_AileronLift =", AileronData['F_Lift'])
  print("F_AileronDrag =", AileronData['F_Drag'])
  print('AoA_Elevator =', ElevatorData['AoA'])
  print("Elevator Speed =", np.linalg.norm(model.vel_ACE(x)))
  print("F_ElevatorLift =", model.rotateView(x,1,3,ElevatorData['F_Lift']))
  print("F_ElevatorDrag =", model.rotateView(x,1,3,ElevatorData['F_Drag']))
  print("M_Gravity (wrt 1)=", model.M_gravity(x))
  print("M_Gravity (wrt 3)=", model.rotateView(x,1,3,model.M_gravity(x)))
  print("M_Gravity (wrt 4)=", model.rotateView(x,1,4,model.M_gravity(x)))
  print("M_Elevator =", model.rotateView(x,1,3,model.M_Elevator(x)))
  print("M_Aileron =", model.rotateView(x,1,3,model.M_Aileron(x)))
  
  #print("M_CarouselMotor =", model.rotateView(x,1,3,model.M_CarouselMotor(x,Zsim[:,k])))


  print("==")
  print("L1 (wrt 1) =", model.rotateView(x,4,1,cas.mtimes([model.I_COM,model.omega(x)])))
  print("L2 (wrt 1) =", model.params['m'] * cross_product(model.pos_COM(x)-model.pos_cs3(x),model.vel_COM(x)))
  print("L (wrt 1) = ", model.L(x))
  print("dL (wrt 1) =", model.dL(x,z))
  print("M int (wrt 1) =", model.M_int(x,z))
  #print("M_ext (wrt 1) =", model.M_ext(x,u))
  print("==")
  print("L (wrt 3) = ",    model.rotateView(x,1,3,model.L(x)))
  print("dL (wrt 3) =",    model.rotateView(x,1,3,model.dL(x,z)))
  print("M int (wrt 3) =", model.rotateView(x,1,3,model.M_int(x,z)))
  #print("M ext (wrt 3) =", model.rotateView(x,1,3,model.M_ext(x,u)))
  
  #if k==2: quit(0)
  """

print(bcolors.FAIL, "##########################################################################", bcolors.ENDC)
print(bcolors.WARNING, "MISC ============================================================", bcolors.ENDC)
print("t =", T)
print('x =', Xsim[:,-1])
print("z =", Zsim[:,-1])
print("ode = ", model.f(Xsim[:,-1],Zsim[:,-1],u,p)[0])
print("alg = ", model.f(Xsim[:,-1],Zsim[:,-1],u,p)[1])


import models.carousel_whitebox.carousel_whitebox_viz as cplot
cplot.plotStates(model,Xs_sim=Xsim,Us_sim=Usim,dt=dt)
#cplot.plotAerodynamics(Xsim,Usim,dt,model)
#cplot.plotMoments(Xsim,Zsim,Usim,dt,model)
#cplot.plotFlightTrajectory(Xsim,model)

plt.show()
