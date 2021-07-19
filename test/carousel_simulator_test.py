import numpy as np
from thesis_code.carousel_model import CarouselWhiteBoxModel
from thesis_code.carousel_simulator import Carousel_Simulator
from thesis_code.models.carousel_whitebox_viz import *

np.set_printoptions(linewidth=np.inf)

# Create default model
model = CarouselWhiteBoxModel()

# Set initial state and simulation properties
x0, z0, u0 = model.get_steady_state()
v_mu =  0e0 * np.ones(model.NY())
R    = 1e-3 * np.eye(model.NY())
w_mu =  0e0 * np.ones(model.NX())
Q    = 1e-6 * np.eye(model.NX())

# Create a simulator
sim = Carousel_Simulator(
  model,
  x0,
  z0=z0,
  measurement_noise_covar = R,
  process_noise_covar = Q
)

# Simulate a few steps
for k in range(10):
  x,z,y = sim.simstep(0.5, 0.1)
  print(y)
