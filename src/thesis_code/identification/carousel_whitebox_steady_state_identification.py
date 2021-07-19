#!/usr/bin/python3

"""
  Author: Paul Daum
  Date: 2019/06/24
  Brief: This script finds a steady state for the carousel whitebox model a
         nd identifies parameters given a steady state
"""

""" =========================================       PREPARATION       ============================================= """

# For the system definition
import casadi as cas
from casadi import vertcat, symvar, mtimes
# For the model
from thesis_code.carousel_model import CarouselWhiteBoxModel
# For math
import numpy as np
np.set_printoptions(linewidth=np.inf)
# For plotting
# For colorful text
# For deepcopy
# For nice dictionary prints
import pprint
# For signal generation
# For everything else
from thesis_code.utils.ParametricNLP import ParametricNLP


""" =========================================       MODEL SETUP       ============================================= """

# Get defaul model parameters
print("Creating model parameters..")
model_params = CarouselWhiteBoxModel.getDefaultParams()

# Create a carousel model
print("Creating model..")
model = CarouselWhiteBoxModel(model_params)

# Print parameters
print(" ---------------------- Model Parameters ---------------------- ")
pprint.pprint(model.params)
print(" -------------------------------------------------------------- ")

# The semi-explicit DAE
dae = model.dae

# All symbolics
x_sym = cas.vertcat(*dae.x)
z_sym = cas.vertcat(*dae.z)
u_sym = cas.vertcat(*dae.u)
p_sym = cas.vertcat(*dae.p)

# Function objects
ode_fun = cas.Function('ode', [x_sym,z_sym,u_sym,p_sym], [cas.vertcat(*dae.ode)])
alg_fun = cas.Function('alg', [x_sym,z_sym,u_sym,p_sym], [cas.vertcat(*dae.alg)])

""" =====================================       ROOTFINDER SETUP       ============================================= """
print(" -------------------------------------------------------------- ")
print("Doing steady state search for given parameters..")

# Create the rootfinder
ss_finder = ParametricNLP('ss_finder')
ss_finder.add_decision_var('x', (model.NX(),1))
ss_finder.add_decision_var('z', (model.NZ(),1))
ss_finder.add_parameter('u', (model.NU(),1))
ss_finder.add_parameter('p', (model.NP(),1))
ss_finder.bake_variables()
ss_x_sym = ss_finder.get_decision_var('x')
ss_z_sym = ss_finder.get_decision_var('z')
ss_u_sym = ss_finder.get_parameter('u')
ss_p_sym = ss_finder.get_parameter('p')

# Set up rootfinding problem
xdot = vertcat(0,0,-2,0,0,0,0)
residual_ode_sym = ode_fun(ss_x_sym,ss_z_sym,ss_u_sym,ss_p_sym) - xdot
residual_alg_sym = alg_fun(ss_x_sym,ss_z_sym,ss_u_sym,ss_p_sym)
ss_finder.set_cost(mtimes([residual_ode_sym.T,residual_ode_sym]) + mtimes([residual_alg_sym.T, residual_alg_sym]))
ss_finder.add_equality('psi', ss_x_sym[2])
ss_finder.add_equality('dots', ss_x_sym[3:6] - xdot[:3])
ss_finder.add_equality('ddots', ss_z_sym[:2] - xdot[3:5])
ss_finder.init(opts={'ipopt.print_level':0,'print_time': 0,'ipopt.sb': 'yes'})

# Create guess and parameters
ss_finder_guess = ss_finder.struct_w(0)
ss_finder_guess['x'] = cas.DM( model.x0() )
ss_finder_guess['z'] = cas.DM.zeros(3,1)
ss_finder_params = ss_finder.struct_p(0)
ss_finder_params['u'] = cas.DM( model.u0() )
ss_finder_params['p'] = cas.DM( model.p0() )

# Solve the rootfinding problem
ss_finder_solution, ss_finder_stats, ss_finder_solution_orig, lb, ub = ss_finder.solve(ss_finder_guess, ss_finder_params)
x_ss_sol = ss_finder_solution['w']['x']
z_ss_sol = ss_finder_solution['w']['z']

print("xsol =", x_ss_sol)
print("zsol =", z_ss_sol)

print("Sanity check..")
print("ode =", ode_fun(x_ss_sol,z_ss_sol,model.u0(),model.p0()))
print("alg =", alg_fun(x_ss_sol,z_ss_sol,model.u0(),model.p0()))

""" =================================       SENSITIVITY ANALYSIS       ====================================== """
print(" -------------------------------------------------------------- ")
print("Doing sensitivity analysis..")

# Compute the sensitivity of the NLP solution (jacobian of x wrt p)
sensitivity_finder = ss_finder.solver.factory('sens', ss_finder.solver.name_in(), ['jac:x:p'])
sensitivity = sensitivity_finder(
  x0 = ss_finder_solution_orig['x'],
  lam_x0 = ss_finder_solution_orig['lam_x'],
  lam_g0 = ss_finder_solution_orig['lam_g'],
  lbx = -cas.inf,
  ubx = cas.inf,
  lbg = lb,
  ubg = ub,
  p = ss_finder_params
)
print("Jacobian of x wrt p:")
sens = sensitivity['jac_x_p']
#print(sens)

print("STATES AND ALGEBRAIC VARIABLES ORDERING")
for k in range(x_sym.rows()):
  print(k, ':', x_sym[k])

for k in range(z_sym.rows()):
  print(x_sym.rows()+k, ':', z_sym[k])

print("CONTROLS ORDERING")
for k in range(u_sym.rows()):
  print(k, ':', u_sym[k])

print("PARAMETERS ORDERING")
for k in range(p_sym.rows()):
  print(k, ':', p_sym[k])

# Plot sensitivity matrix
#from seaborn import heatmap, color_palette
#ax = heatmap(sens, linewidth=0.5, cmap=color_palette("RdBu_r",61), center=0)
#plt.show()


# TODO: Plot moments and forces at steady state

#quit(0)
""" =================================       STEADY STATE IDENTIFICATION       ====================================== """
print(" -------------------------------------------------------------- ")
print("Doing steady state identification..")

# Select parameters that will not be subject to identification
# Don't select vector subcomponents. Always the whole vector/matrix
idx_fix_parameters = []
idx_fix_parameters += [0,1,2,3,4] # Carousel speed and time constant, gravity, air density, mass
idx_fix_parameters += [5,6,7,8,9,10,11,12,13] # Inertia tensor
idx_fix_parameters += [14,15,16] # Center of mass
idx_fix_parameters += [17,18,19] # Wing aerodynamic center
idx_fix_parameters += [20,21,22] # Elevator aerodynamic center
idx_fix_parameters += [23,24,25] # IMU position
idx_fix_parameters += [26,27,28] # 4 wrt 3
idx_fix_parameters += [29,30,31] # 3 wrt 2
idx_fix_parameters += [32,33] # Aileron and wing surface areas
idx_var_parameters = [k for k in range(p_sym.rows()) if k not in idx_fix_parameters]

# Split up parameter vector into fixed and variable parameters (using MX-hack1)
p_fix_sym = vertcat(*symvar(vertcat(*[p_sym[i] for i in idx_fix_parameters])))
p_var_sym = vertcat(*symvar(vertcat(*[p_sym[i] for i in idx_var_parameters])))

#print(p_fix_sym)
#print(p_var_sym)

# Augment ode and alg to be able to exclude fixed parameters
ode_aug_fun = cas.Function('ode', [x_sym,z_sym,u_sym,p_var_sym,p_fix_sym], [ode_fun(x_sym,z_sym,u_sym,p_sym)])
alg_aug_fun = cas.Function('alg', [x_sym,z_sym,u_sym,p_var_sym,p_fix_sym], [alg_fun(x_sym,z_sym,u_sym,p_sym)])

# Create the NLP
ident = ParametricNLP('ss_identificator')
ident.add_decision_var('pvar', (model.NP()-len(idx_fix_parameters),1))
ident.add_decision_var('z', (model.NZ(),1))
ident.add_parameter('x', (model.NX(),1))
ident.add_parameter('u', (model.NU(),1))
ident.add_parameter('pvar_prior', (model.NP()-len(idx_fix_parameters),1))
ident.add_parameter('pfix', (len(idx_fix_parameters),1))
ident.bake_variables()
id_pvar_sym = ident.get_decision_var('pvar')
id_x_sym = ident.get_parameter('x')
id_z_sym = ident.get_decision_var('z')
id_u_sym = ident.get_parameter('u')
id_pvar_prior_sym = ident.get_parameter('pvar_prior')
id_pfix_sym = ident.get_parameter('pfix')

# Set cost: Only regularization
xdot = vertcat(0,0,-2,0,0,0,0)
id_ode = ode_aug_fun(id_x_sym,id_z_sym,id_u_sym,id_pvar_sym,id_pfix_sym)
id_alg = alg_aug_fun(id_x_sym,id_z_sym,id_u_sym,id_pvar_sym,id_pfix_sym)
residual_pvar_sym = id_pvar_sym - id_pvar_prior_sym
residual_ode_sym = id_ode - xdot
ident.set_cost(
  cas.mtimes([residual_pvar_sym.T, residual_pvar_sym]) +
  cas.mtimes([residual_ode_sym.T, residual_ode_sym])
  #cas.mtimes([id_alg.T, id_alg])
)

# Set constraints and initialize
ident.add_equality('psi', id_x_sym[2])
ident.add_equality('dots', id_x_sym[3:6] - xdot[:3])
ident.add_equality('ddots', id_z_sym[:2] - xdot[3:5])
ident.add_equality('alg', id_alg)
ident.init(opts={'ipopt.print_level':0,'print_time': 0,'ipopt.sb': 'yes'})

# Set initial guess
id_initial_guess = ident.struct_w(0)
id_initial_guess['pvar'] = cas.DM( cas.vertcat(*[model.p0()[i] for i in idx_var_parameters]) )
id_initial_guess['z'] = vertcat(0,0,-1.4496)
#print([initial_guess['p'][k] for k in range(46)])

# Set nlp parameters
x_param = cas.DM(np.array([ np.pi-2.792526803190927, 5.316541413767342-np.pi, 0., 0., 0., -2., 0. ]))
#x_param = cas.DM(np.array([0.978885, -0.135006,  0., 0., 0., -2, 0.]))
z_param = cas.DM(np.array([0., 0., -1.4496])) # Last element should be free variable
u_param = cas.DM(np.array([ 0., -2. ]))
pvar_prior_param = cas.DM( cas.vertcat(*[model.p0()[i] for i in idx_var_parameters]) )
pfix_param = cas.DM( cas.vertcat(*[model.p0()[i] for i in idx_fix_parameters]) )

id_params = ident.struct_p(0)
id_params['x'] = x_param
#id_params['z'] = z_param
id_params['u'] = u_param
id_params['pvar_prior'] = pvar_prior_param
id_params['pfix'] = pfix_param

# Solve the nlp
id_result, id_stats, dummy,dummy,dummy = ident.solve(id_initial_guess, id_params)

z_sol = id_result['w']['z']

# Analyze the nlp
# TODO: Plot legend
#ident.spy(1)
#plt.show()

# Fetch solution, reconstruct full parameter vector # <<<<<<<<<<<<<<<<<<<<<<< pfull IS WRONGLY ASSEMBLED
pvar_sol = id_result['w']['pvar']
pfull = cas.DM.zeros(model.NP(),1)
pfix_cnt = 0
pide_cnt = 0
for k in range(model.NP()):
  if k in  idx_fix_parameters:
    pfull[k] = pfix_param[pfix_cnt]
    pfix_cnt = pfix_cnt + 1
  else:
    pfull[k] = pvar_sol[pide_cnt]
    pide_cnt = pide_cnt + 1


#x_param = cas.vertcat(x_param[0], x_param[1], 0, x_param[2:])
#ode = cas.Function('ode', [xsym,zsym,usym,pidentsym,pfixedsym], [odesym])


print("================ IDENT RESULT ==============")
cnt = 0
for k in range(p_sym.rows()):
  if k not in idx_fix_parameters:
    print(p_sym[k], ":", pvar_prior_param[cnt], "-->", pvar_sol[cnt])
    cnt = cnt + 1

print("Sanity check..")
print("x =", x_param)
print("z =", z_sol)
print("ode =", ode_aug_fun(x_param,z_sol,u_param,pvar_sol,pfix_param))
print("alg =", alg_aug_fun(x_param,z_sol,u_param,pvar_sol,pfix_param))

#print("ODE SHOULD =", ode_aug_fun(x_param,z_param,u_param,pvar_prior_param,pfix_param))
#print("ALG SHOULD =", alg_aug_fun(x_param,z_param,u_param,pvar_prior_param,pfix_param))
