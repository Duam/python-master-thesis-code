import casadi as cas
from thesis_code.utils.ParametricNLP import ParametricNLP

# Create a test optimization problem
nlp = ParametricNLP(name='test_problem', verbose=True)

# Add decision variables and parameters to the problem
nlp.add_decision_var('x', (3,1))
nlp.add_parameter('lbx', (3,1))
nlp.add_parameter('ubx', (3,1))
nlp.bake_variables()

# Fetch symbolics
x_sym = nlp.get_decision_var('x')
lbx_sym = nlp.get_parameter('lbx')
ubx_sym = nlp.get_parameter('ubx')

print(nlp.struct_w['x'])

# Create a cost function
nlp.set_cost(cas.mtimes([x_sym.T, x_sym]))

# Create an inequality constraint
nlp.add_inequality('x_iq_lbx', x_sym - lbx_sym)
nlp.add_inequality('x_iq_ubx', ubx_sym - x_sym)

# Set the parameters and initial guess
lbx_scalar = 0.5
ubx_scalar = 1.0
params = nlp.struct_p(0)
params['lbx'] = lbx_scalar * cas.DM.ones((3,1))
params['ubx'] = ubx_scalar * cas.DM.ones((3,1))
winit = nlp.struct_w(0)
winit['x'] = cas.DM.zeros((3,1))

# Some options 
opts = {}
opts['ipopt.print_info_string'] = 'yes'
opts['ipopt.print_level'] = 3
opts['ipopt.max_iter'] = 1000

# Solve the problem..
# via ipopt
nlp.init(nlpsolver='ipopt')
res_ipopt, stats_ipopt, dum,dum,dum = nlp.solve(winit=winit, param=params)

# via sqpmethod
nlp.init(nlpsolver='sqpmethod')
res_sqp, stats_sqp, dum,dum,dum = nlp.solve(winit=winit, param=params)

# via qpoases
nlp.init(is_qp = True, nlpsolver='qpoases')
res_qp, stats_qp, dum,dum,dum = nlp.solve(winit=winit, param=params)

# Solve the problem
"""
print('x:', res['w']['x'])
print('lambda ubx:', res['lam_h']['x_iq_ubx'])
print('lambda lbx:', res['lam_h']['x_iq_lbx'])
print('Number of iterations:', stats['iter_count'])
"""