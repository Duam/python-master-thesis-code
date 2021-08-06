import numpy as np
import casadi as cas
from thesis_code.model import CarouselModel
import thesis_code.models.carousel_whitebox_viz as viz
from thesis_code.simulator import CarouselSimulator
from thesis_code.components.carousel_mhe import Carousel_MHE
from thesis_code.components.carousel_tarsel import Carousel_TargetSelector
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=np.inf)

# Load identified parameters
import json
param = {}
path = "../apps/identification_imu/params_identified_dynamic_imu.json"
print("Loading identified parameter set " + path)
with open(path, 'r') as file:
    param = json.load(file)

# Create model
constants = CarouselModel.getConstants()
model = CarouselModel(param)
x0_est,z0_est,u0_est = model.get_steady_state()
NX = model.NX()
NU = model.NU()
NY = model.NY()

print(model.NP())

# Simulation meta-parameters
T_per = 2*np.pi / abs(constants['carousel_speed'])

M_sim = 2
T_sim = M_sim * T_per
dt_sim = 1. / 50.
N_sim = int(np.ceil(T_sim / dt_sim))
#N_sim = 100
#dt_sim = T_sim / N_sim
jit = False
verbose = True

# Create a target selector and some reference controls
N_ts = 20
ref_fun = lambda psi: x0_est[0] + 15 * (2*np.pi/360) * np.sin(psi)
target_selector = Carousel_TargetSelector(model = model, T=2*np.pi, N = N_ts, reference = ref_fun)
target_selector.init(x0 = x0_est, z0 = z0_est, W = cas.DM.eye(1))
target_selector.call()
reference = target_selector.get_new_reference(0.0, dt_sim, N_sim)
x0_sim = reference[0][:,0]
z0_sim = reference[1][:,0]
Us_sim = reference[2][:,:-1]

# Create an estimator
# TODO: Try out what if Q,R,S are inverse of simulation Q,R
N_mhe = 3
estimator = Carousel_MHE(model = model, N = N_mhe, dt = dt_sim, verbose = verbose, do_compile = jit, expand=True)
Q_mhe = 1e2 * np.ones((NX,1))
R_mhe = 1e-1 * np.ones((NY,1))
S_mhe = 1e3 * np.eye(NX)
print(Q_mhe)
estimator.init(x0_est = x0_est, z0_est = z0_est, Q = Q_mhe, R = R_mhe, S0 = S_mhe, Q_ekf=1/Q_mhe, R_ekf=1/R_mhe)
print(x0_est)

print(estimator.ocp)
w0 = estimator.initial_guess
p0 = estimator.parameters

print("Doing one sample solve(), which should converge in one step..")
result = estimator.ocp.solve(estimator.initial_guess,estimator.parameters) # Should converge immediately!
#assert result[1]['iter_count'] == 0
print("Done.")

# Create evaluation containers
Us_sim = [ Us_sim[:,k] for k in range(Us_sim.shape[1]) ]
Xs_sim = [ x0_sim ]
Zs_sim = [ z0_sim ]
Ys_sim = [ ]
Xs_est = [ cas.vertcat(x0_est[:2], 0.0, x0_est[2:4], -2., x0_est[4]) ]
Zs_est = [ z0_est ]

# Create simulator
Q_sim = 1e-8 * np.eye((NX+2))
R_sim = 1e-3 * np.eye((NY))
simulator = CarouselSimulator(
    model = model, x0 = Xs_sim[0], z0 = Zs_sim[0],
    process_noise_mean = 0e0 * np.ones((NX+2)),
    process_noise_covar = Q_sim,
    measurement_noise_mean = 0e0 * np.ones((NY)),
    measurement_noise_covar = R_sim,
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
    u0_k = Us_sim[k]

    # Simulate one step
    xf_k, zf_k, y0_k = simulator.simulate_timestep(u0_k[0], dt_sim)
    Xs_sim += [ xf_k ]
    Zs_sim += [ zf_k ]
    Ys_sim += [ y0_k ]

    print("x_sim(k) =", Xs_sim[-2])
    print("u_sim(k) =", Us_sim[-1])
    print("y_sim(k) =", Ys_sim[-1])

    w0 = estimator.initial_guess

    # Estimate one step
    xf_k_est, zf_k_est, est_result, est_stats, duration = estimator.call(u0_k[0], y0_k, verbose=False)
    Xs_est += [ cas.vertcat(xf_k_est[:2], -2*(k+1)*dt_sim, xf_k_est[2:4], -2., xf_k_est[4]) ]
    Zs_est += [ zf_k_est ]

    print("x_est(k) =", Xs_est[-2])

    print(est_stats['t_proc_'+str(estimator.ocp.name)])
    solve_time += [ est_stats['t_proc_'+str(estimator.ocp.name)] ]
    iter_count += [ est_stats['iter_count']]
    print("obj (0) =", est_stats['iterations']['obj'][0], " -> ", est_stats['iterations']['obj'][-1])
    print("inf_pr (0) =", est_stats['iterations']['inf_pr'][0], " -> ", est_stats['iterations']['inf_pr'][-1])
    print("MHE solve time: ", solve_time[-1]*1e3, "ms")
    #print("MHE iter count: ", iter_count[-1])
    print(est_stats['iterations']['regularization_size'])
    print(est_stats['return_status'])
    if est_stats['return_status'] != 'Solve_Succeeded':
        print("ABORT. Return status was not Solve_succeeded.")
        break


avg_solve_time = sum(solve_time) / float(N_sim)
avg_iter_count = sum(iter_count) / float(N_sim)
print("============================================================================================")
print("Average MHE solve() time:", avg_solve_time)
print("Average iter count: ", avg_iter_count)

xs_est = cas.horzcat(*Xs_est)
xs_sim = cas.horzcat(*Xs_sim)
us_sim = cas.horzcat(*Us_sim)

viz.plotStates_withRef(
    Xs = xs_est, Xs_ref = xs_sim,
    Us = us_sim, Us_ref = us_sim,
    dt = dt_sim, model = model
)

plt.show()