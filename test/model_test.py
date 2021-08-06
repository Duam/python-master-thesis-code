import pytest
import casadi as cas
import numpy as np
from thesis_code.utils.bcolors import bcolors
import pprint
import matplotlib.pyplot as plt
from thesis_code.model import CarouselModel
import thesis_code.utils.signals as signals
np.set_printoptions(linewidth=np.inf)

@pytest.fixture
def simulation_params():
    settling_time = 2.*np.pi
    test_duration = 4.*np.pi
    num_samples = 200
    return {
        'settling_time': settling_time,
        'test_duration': test_duration,
        'total_duration': settling_time + test_duration,
        'num_samples': num_samples,
        'timestep': (settling_time + test_duration) / float(num_samples)
    }

@pytest.fixture
def params():
    return CarouselModel.getDefaultParams()

@pytest.fixture
def model(params):
    return CarouselModel(params)

@pytest.fixture
def integrator(model, simulation_params):
    dae = model.dae
    x_sym = cas.vertcat(*dae.x)
    z_sym = cas.vertcat(*dae.z)
    u_sym = cas.vertcat(*dae.u)
    p_sym = cas.vertcat(*dae.params)
    ode_sym = cas.vertcat(*dae.ode)
    alg_sym = cas.vertcat(*dae.alg)
    quad_sym = cas.vertcat(*dae.quad)
    dae_dict = {'x': x_sym, 'ode': ode_sym, 'alg': alg_sym, 'z': z_sym, 'p': cas.vertcat(p_sym, u_sym)}
    int_opts = {'number_of_finite_elements': 1, 'output_t0': True, 'tf': simulation_params['timestep']}
    return cas.integrator('xnext', 'collocation', dae_dict, int_opts)

def test_simulate(simulation_params, params, model, integrator):
    p = np.concatenate([np.array(val).flatten() for val in params.values()])
    num_samples = simulation_params['num_samples']
    N_settle = int(simulation_params['settling_time'] /\
                   simulation_params['total_duration'] *\
                   num_samples)
    N_per = 20
    Usim = np.repeat(model.u0(), num_samples, axis=1)
    Usim[0, N_settle:] = [0.5 + 0.1745 * signals.rectangle(k, N_per) for k in range(num_samples - N_settle)]
    state = model.x0()
    for k in range(num_samples):
        control = Usim[:, k]
        step = integrator(x0=state, p=cas.vertcat(p, control))
        state = step['xf'][:, 1].full().flatten()
