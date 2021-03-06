import signal
import time
import numpy as np
from zerocm import ZCM
from thesis_code.zcm_message_definitions.timestamped_vector_float import timestamped_vector_float
from thesis_code.zcm_message_definitions.timestamped_vector_double import timestamped_vector_double
from thesis_code.utils.zcm_constants import carousel_zcm_constants
from thesis_code.model import CarouselModel
from thesis_code.simulator import CarouselSimulator

# Create a simulation model
param_sim = CarouselModel.getDefaultParams()
model_sim = CarouselModel(param_sim)

# Set initial state and noise properties
x_ss_sim, z_ss_sim, u_ss_sim = model_sim.get_steady_state()
v_mean_sim  =  0e0 * np.ones(model_sim.NY())
v_covar_sim =  0e-3 * np.eye(model_sim.NY())
w_mean_sim  =  0e0 * np.ones(model_sim.NX())
w_covar_sim =  0e0 * np.eye(model_sim.NX())

# Create a simulator
u = u_ss_sim
simulator = CarouselSimulator(
    model_sim, x0 = x_ss_sim, z0 = z_ss_sim,
    process_noise_mean = w_mean_sim,
    process_noise_covar = w_covar_sim,
    measurement_noise_mean = v_mean_sim,
    measurement_noise_covar = v_covar_sim,
    jit=True
)

print("x0 = " + str(x_ss_sim))
print("z0 = " + str(z_ss_sim))
print("u0 = " + str(u_ss_sim))


def control_cb(channel, message) -> None:
    """Callback that executes when a new control message arrives
    :param channel: The zcm channel to listen to
    :param message: The message arriving
    :returns: None
    """
    global u
    print("Received " + str(message.values) + " on channel " + channel)
    u = message.values


zcm = ZCM(carousel_zcm_constants.url)
if not zcm.good():
    raise Exception("ZCM not good")

# Subscribe to the channels
zcm.subscribe(carousel_zcm_constants.control_elevator_channel, timestamped_vector_float, control_cb)


def signal_handler(sig, frame) -> None:
    """Create SIGINT handler for quick stop
    :param sig: The signal that triggers the handler
    :param frame: ?
    :returns: None
    """
    print('Ctrl+C: Exiting.')
    zcm.stop()
    quit(0)


# Register signal
signal.signal(signal.SIGINT, signal_handler)

# Prepare a few message containers
roll_msg = timestamped_vector_float()
roll_msg.len = 1
pitch_msg = timestamped_vector_float()
pitch_msg.len = 1
yaw_msg = timestamped_vector_double()
yaw_msg.len = 1
yaw_sin_msg = timestamped_vector_double()
yaw_sin_msg.len = 1
yaw_cos_msg = timestamped_vector_double()
yaw_cos_msg.len = 1
acc_msg = timestamped_vector_float()
acc_msg.len = 3
gyr_msg = timestamped_vector_float()
gyr_msg.len = 3
state_msg = timestamped_vector_double()
state_msg.len = 7

# Start zcm
zcm.start()

# Get startup time
startTime = time.time()
# Initialize iterations
k = 0
dt = 1/50.
# Simulation main loop
lastTime = startTime
print("Start.")
while True:
    # Simulate one step
    xf_k, zf_k, y0_k = simulator.simulate_timestep(u[0], dt)
    yaw = xf_k[2].full()

    # Extract measurements
    y0_k = y0_k.full()
    y_yaw = np.mod(yaw, 93*2*np.pi)
    y_roll = y0_k[0]
    y_pitch = y0_k[1]

    nowTime = time.time()
    timestamp = int(nowTime*1e9)

    # Publish true roll
    roll_msg.ts = timestamp
    roll_msg.values = y_roll
    zcm.publish(carousel_zcm_constants.out_roll_channel, roll_msg)

    # Publish true pitch
    pitch_msg.ts = timestamp
    pitch_msg.values = y_pitch
    zcm.publish(carousel_zcm_constants.out_pitch_channel, pitch_msg)

    # Publish true yaw
    yaw_msg.ts = timestamp
    yaw_msg.values = y_yaw
    zcm.publish(carousel_zcm_constants.out_yaw_channel, yaw_msg)

    # Publish internal state
    state_msg.ts = timestamp
    state_msg.values = xf_k.full()
    zcm.publish("CAR_SIM_STATE", state_msg)

    # calculate sleep time
    k = k + 1
    sleepTime = (startTime - nowTime) + k * dt

    # Make sure not to sleep negative time
    if sleepTime < 0:
        sleepTime = 0

    sleepTime_msg = timestamped_vector_double()
    sleepTime_msg.ts = timestamp
    sleepTime_msg.len = 1
    sleepTime_msg.values = [simulator.integrate_step.stats()['t_proc_xnext']]
    zcm.publish("T_PROC_XNEXT", sleepTime_msg)

    # sleep
    time.sleep(sleepTime)
