import signal, time, os, multiprocessing
from thesis_code.utils.bcolors import bcolors
from pynput.keyboard import Key, Listener
import numpy as np
import casadi as cs
from zerocm import ZCM
from thesis_code.zcm_message_definitions.timestamped_vector_float import timestamped_vector_float
from thesis_code.zcm_message_definitions.timestamped_vector_double import timestamped_vector_double
from thesis_code.utils.zcm_constants import carousel_zcm_constants as carzcm
import thesis_code.utils.signals as signals
from thesis_code.model import CarouselModel
from thesis_code.mhe import Carousel_MHE
from thesis_code.mpc import Carousel_MPC
from thesis_code.target_selector import Carousel_TargetSelector

# Set OMP number of threads
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

# Initialize ZCM
zcm = ZCM(carzcm.url)
if not zcm.good():
    raise Exception("ZCM not good")


def send_neutral_controls():
    """Sends out steady state controls for all aerodynamic surfaces.
    """
    print("Sending neutral controls to Half-Wing..")
    timestamp = int(time.time()*1e9)
    # Aileron
    u_ss_kin1_msg = timestamped_vector_float()
    u_ss_kin1_msg.len = 1
    u_ss_kin1_msg.ts = timestamp
    u_ss_kin1_msg.values = [0.4]
    zcm.publish(carzcm.control_aileron_channel, u_ss_kin1_msg)
    # Flap
    u_ss_kin2_msg = timestamped_vector_float()
    u_ss_kin2_msg.len = 1
    u_ss_kin2_msg.ts = timestamp
    u_ss_kin2_msg.values = [0.475]
    zcm.publish(carzcm.control_flap_channel, u_ss_kin2_msg)
    # Rudder
    u_ss_kin3_msg = timestamped_vector_float()
    u_ss_kin3_msg.len = 1
    u_ss_kin3_msg.ts = timestamp
    u_ss_kin3_msg.values = [0.5]
    zcm.publish(carzcm.control_rudder_channel, u_ss_kin3_msg)
    # Elevator
    u_ss_kin4_msg = timestamped_vector_float()
    u_ss_kin4_msg.len = 1
    u_ss_kin4_msg.ts = timestamp
    u_ss_kin4_msg.values = [0.5]
    zcm.publish(carzcm.control_elevator_channel, u_ss_kin4_msg)


# Create keypress handler for state transition
request_next_state = False
def on_press(key):
    global request_next_state
    if key == Key.esc:
        request_next_state = True
        print("{0} pressed".format(key))

listener = Listener(on_press=on_press, on_release=None)
listener.start()


def signal_handler(sig, frame):
    print('Ctrl+C: Exiting.')
    send_neutral_controls()
    zcm.stop()
    quit(0)


# Register Ctrl+C signal
signal.signal(signal.SIGINT, signal_handler)

# Sample frequency is 20 Hz
dt = 1. / 20.
dt_ctr = 1. / 20.

# Estimation and control horizons
N_ts = 40
N_est = 3
N_ctr = 3

# Flags
verbose = False
jit = True
expand = True
verbose_estimator = False
verbose_controller = False
verbose_target_selector = False
enable_estimator = True
enable_controller = True
enable_target_selector = True
use_direct_angle_measurement = True

# Load identified parameters
import json
param = {}
path = "../carousel_identification/params_identified_dynamic_imu.json"
print("Loading identified parameter set " + path)
with open(path, 'r') as file:
    param = json.load(file)

# Create the estimation and control model
if use_direct_angle_measurement:
    model = CarouselModel(param, with_angle_output=True, with_imu_output=False)
else:
    model = CarouselModel(param, with_angle_output=False, with_imu_output=True)

# Fetch sizes and steady state
num_states = model.NX()
num_alg_vars = model.NZ()
num_controls = model.NU()
num_measurements = model.NY()
num_params = model.NP()
x_ss_est, z_ss_est, u_ss_est = model.get_steady_state()

print(f"# States: {num_states}")
print(f"# Algebraic variables: {num_alg_vars}")
print(f"# Controls: {num_controls}")
print(f"# Measurements: {num_measurements}")
print(f"# Parameters: {num_params}")

# Create a target-selector
if enable_target_selector:
    # Set period
    T_ts = 10

    # Choose a signal type    
    signal = lambda t, T: 0.5
    signal = signals.rectangle
    amplitude = 0.20
    offset = 0.0
    
    # Construct signal and create target selector.
    roll_ref = lambda t: x_ss_est[0] + amplitude/2. * (signal(t, T_ts) - 0.5) - offset
    #roll_ref = lambda t: amplitude * (signal(t,T_ts) - 0.5) - offset
    target_selector = Carousel_TargetSelector(
        model,
        T_ts,
        N_ts,
        roll_ref,
        verbose=verbose_target_selector
    )

# Create an estimator
if enable_estimator:
    estimator = Carousel_MHE(
        model,
        N_est,
        dt,
        verbose=verbose_estimator,
        do_compile=jit,
        expand=expand
    )
    print(estimator.ocp)
    Q_mhe = np.array([1e2, 1e2, 1e0, 1e0, 1e1]).reshape((num_states, 1))
    S0_mhe = 1e-2 * Q_mhe
    if not use_direct_angle_measurement:
        R_mhe = np.array([1e-1, 1e-1, 1e-1, 1e0, 1e0, 1e0]).reshape((num_measurements, 1))
    if use_direct_angle_measurement:
        R_mhe = 1e1 * np.ones((num_measurements, 1))

# Create a controller
if enable_controller:
    controller = Carousel_MPC(model, N_ctr, dt_ctr, verbose=verbose_controller, do_compile=jit, expand=expand)
    print(controller.ocp)
    Q_mpc = 1e3 * np.ones((num_states, 1))
    Q_mpc[0] = 1e3
    Q_mpc[2] = 1e4
    Q_mpc[1] = 1e0
    Q_mpc[3] = 1e2
    Q_mpc[4] = 1e3
    R_mpc = 1e3 * np.ones((num_controls, 1))
    S_mpc = 1e-2* Q_mpc #1e4 * np.ones((NX,1))
    S_mpc[0] = 1e1 * Q_mpc[0]
    S_mpc[1] = 1e2 * Q_mpc[1]
    #S_mpc[3] = 1e2 * Q_mpc[3]
    #S_mpc[4] = 1e2 * Q_mpc[4]
    #S_mpc[2] = 1e-1
    #S_mpc[3] = 1e1
    #S_mpc[3] = 1e3
    #S_mpc[4] = 1e2
    #S_mpc[0] = 12e3
    #S_mpc[2] = 1e3



#quit(0)

""" ================================================================================================================ """
def send_control_configuration():
    print("Sending configuration to ZCM..")
    msg = timestamped_vector_double()
    msg.ts = int(time.time()*1e9)
    
    if enable_estimator:
        msg.len = num_states
        msg.values = Q_mhe
        zcm.publish("Q_mhe", msg)
        msg.values = S0_mhe
        zcm.publish("S0_mhe", msg)
        msg.len = num_measurements
        msg.values = R_mhe
        zcm.publish("R_mhe", msg)
    if enable_controller:
        msg.len = num_states
        msg.values = Q_mpc
        zcm.publish("Q_mpc", msg)
        msg.values = S_mpc
        zcm.publish("S_mpc", msg)
        msg.len = num_controls
        msg.values = R_mpc
        zcm.publish("R_mpc", msg)
    if enable_target_selector:
        msg.len = N_ts
        msg.values = [ roll_ref(t) for t in np.linspace(0, T_ts, N_ts) ]
        zcm.publish("roll_ref", msg)

""" ================================================================================================================ """


# Define measurement buffer and callback
current_measurement = [0.0 for k in range(num_measurements)]
def onNewMeasurement(channel, message):
    global current_measurement
    current_measurement = message.values

# Define yaw buffer and callback
current_yaw = [0.0]
def onNewYaw(channel, message):
    global current_yaw
    current_yaw = message.values[0]

# Control buffer callback (ONLY FOR TESTING)
current_control = [0.5]
def onNewControl(channel, message):
    global current_control
    current_control = message.values[0]

""" ================================================================================================================ """


# Subscribe to the measurements
if use_direct_angle_measurement:
    zcm.subscribe("ANGLES_meas_sampled", timestamped_vector_float, onNewMeasurement)
else:
    zcm.subscribe(carzcm.measurements_sampled_channel, timestamped_vector_float, onNewMeasurement)
zcm.subscribe(carzcm.out_yaw_channel, timestamped_vector_double, onNewYaw)
zcm.subscribe(carzcm.control_elevator_channel, timestamped_vector_float, onNewControl)

print("Listening for measurements on channel " + carzcm.measurements_sampled_channel)
print("Listening for yaw values on channel " + carzcm.out_yaw_channel)

""" ================================================================================================================ """
# Send neutral controls to system
zcm.start()
send_neutral_controls()
send_control_configuration()

# Data containers
Xs_est = [x_ss_est]
Zs_est = [z_ss_est]
Xs_ref = [x_ss_est]
Zs_ref = [z_ss_est]
Us_ref = [cs.DM(0.5)]
Us = [0.5]
Ys = []
est_call = False
ctrl_call = False

""" Compute a reference trajectory """
print(bcolors.WARNING + "Initializing target selector.." + bcolors.ENDC)
#x_ss_ts = cs.vertcat(x_ss_est[:2], 0.0, x_ss_est[2:4], -2, x_ss_est[4] )

print(bcolors.WARNING + "Computing reference.." + bcolors.ENDC)
#target_selector.call()
""" ============================== """

# Main loop state machine
op_states = ['idle', 'mhe_init', 'pure_estimate', 'tarsel_init', 'ctrl_init', 'nominal', 'halt']
current_op_state = 0
if not enable_estimator:
    current_op_state = 3

status_string = bcolors.OKGREEN 
status_string += " ====================================== \n"
status_string += "Switching operational state to " + op_states[current_op_state] 
status_string += "\n ======================================"
status_string += bcolors.ENDC
print(status_string)
startTime = time.time()
lastTime = startTime
time_checkpoint = startTime
trigger_shutdown = False
transition_switch = False
k = 0
ref = x_ss_est[0]
cnt = 0
while True:

    # =========== Fetch time =============
    nowTime = time.time()

    # ======= Receive measurement ========
    y0_k = cs.DM(current_measurement)
    Ys += [y0_k]
    #print(y0_k)

    # ============ Estimator =============

    stopwatch_start = time.time()
    if enable_estimator:
        if current_op_state < 1:
            # Idle: Do nothing
            pass

        elif current_op_state == 1:
            # Transition 1: Initialize MHE
            estimator.init(x0_est=x_ss_est, z0_est=z_ss_est, Q=Q_mhe, R=R_mhe, S0=S0_mhe)
            send_control_configuration()
            request_next_state = True

        elif current_op_state > 1:
            # MHE operation: Call mhe, store results
            print(bcolors.WARNING + "Estimating state .." + bcolors.ENDC)
            xf_k_est, zf_k_est, est_result, est_stats, est_duration = estimator.call(Us[-1], Ys[-1])
            est_call = True
            Xs_est += [xf_k_est]
            Zs_est += [zf_k_est]

            """
            cnt += 1
            if cnt > 500:
                quit(0)
            """

        else:
            print(bcolors.WARNING + "Invalid operational mode: " + str(current_op_state) + "." + bcolors.ENDC)
            trigger_shutdown = True

    # Fetch current state
    x0_k_est = Xs_est[-1]
    z0_k_est = Zs_est[-1]

    # Compute estimator call duration
    stopwatch_end = time.time()
    estimator_call_duration = stopwatch_end - stopwatch_start

    # ======== Target Selector ===========

    stopwatch_start = time.time()
    if enable_target_selector:
        if current_op_state < 3:
            # Idle, transition1, startup: Do nothing
            pass

        elif current_op_state == 3:
            # Transition 2: Compute a reference around the current roll angle, initialize controller
            time_checkpoint = nowTime

            # Select a new reference
            print(bcolors.WARNING + "Initializing reference generator.." + bcolors.ENDC)
            target_selector.init(x0=x_ss_est, z0=z_ss_est, W=1e3 * np.eye(1))
            target_selector.call()
            request_next_state = True

        elif current_op_state > 3:
            tau = nowTime - time_checkpoint

            # Select new reference
            print(bcolors.WARNING + "Selecting reference.." + bcolors.ENDC)
            ref = roll_ref(tau)
            Xref, Zref, Uref = target_selector.get_new_reference(tau, dt_ctr, N_ctr)
            Xs_ref += [Xref[:, 0]]
            Zs_ref += [Zref[:, 0]]
            Us_ref += [Uref[:, 0]]

        else:
            print(bcolors.WARNING + "Invalid operational mode: " + str(current_op_state) + "." + bcolors.ENDC)
            trigger_shutdown = True

    # Compute target selector call duration
    stopwatch_end = time.time()
    target_selector_call_duration = stopwatch_end - stopwatch_start

    # =========== Controller =============

    stopwatch_start = time.time()
    if enable_controller:
        # Send out neutral controls while preparing
        if current_op_state < 4:
            u0_k = 0.5

        elif current_op_state == 4:
            # Initialize the controller
            print(bcolors.WARNING + "Initializing controller.." + bcolors.ENDC)
            controller.init(x0=Xs_est[-1], Q=Q_mpc, R=R_mpc, S=S_mpc, Uref=Uref[:-1])
            send_control_configuration()
            request_next_state = True

        elif current_op_state > 4:
            # Compute new control
            print(bcolors.WARNING + "Computing control.." + bcolors.ENDC)
            u0_k, ctrl_result, ctrl_stats, ctrl_w0, ctrl_p0 = controller.call(x0_k_est, Xref, Uref[:, :-1])
            ctrl_call = True

        else:
            print(bcolors.WARNING + "Invalid operational mode: " + str(current_op_state) + "." + bcolors.ENDC)
            trigger_shutdown = True
    else:
        u0_k = 0.5

    # Compute controlelr call duration
    stopwatch_end = time.time()
    controller_call_duration = stopwatch_end - stopwatch_start

    # ========== Send control ============
    # Apply elevator control command
    u0_k_msg = timestamped_vector_float()
    u0_k_msg.len = 1
    u0_k_msg.ts = int(time.time()*1e9)
    u0_k_msg.values = [u0_k]
    zcm.publish(carzcm.control_elevator_channel, u0_k_msg)
    #zcm.publish("CONTROL", u0_k_msg)

    # ====== Preparation-phase (opt) =====
    Us += [u0_k]
    #Us += [current_control]

    # ========== State Machine ===========
    if request_next_state:
        request_next_state = False
        if current_op_state < len(op_states)-2:
            status_string = bcolors.OKGREEN 
            status_string += " ====================================== \n"
            status_string += "Switching operational state to " + op_states[current_op_state+1] 
            status_string += "\n ======================================"
            status_string += bcolors.ENDC
            print(status_string)
            current_op_state += 1
        else:
            print(bcolors.FAIL + "Cannot leave nominal mode." + bcolors.ENDC)
            pass

    # ============ Printout ==============
    print("u = " + str(u0_k))

    # Current time
    timestamp = int(time.time()*1e9)

    # Publish current state estimate
    x_est_msg = timestamped_vector_double()
    x_est_msg.ts = timestamp
    x_est_msg.len = Xs_est[-1].shape[0]
    x_est_msg.values = Xs_est[-1].full().squeeze().tolist()
    zcm.publish("state_estimate", x_est_msg)

    # Publish current state reference
    x_ref_msg = timestamped_vector_double()
    x_ref_msg.ts = timestamp
    x_ref_msg.len = Xs_ref[-1].shape[0]
    x_ref_msg.values = Xs_ref[-1].full().squeeze().tolist()
    zcm.publish("state_reference", x_ref_msg)

    # Publish current control reference
    u_ref_msg = timestamped_vector_double()
    u_ref_msg.ts = timestamp
    u_ref_msg.len = 1
    u_ref_msg.values = [Us_ref[-1].full()]
    zcm.publish("control_reference", u_ref_msg)

    # Publish true roll reference
    ref_msg = timestamped_vector_double()
    ref_msg.ts = timestamp
    ref_msg.values = [ref]
    ref_msg.len = len(ref_msg.values)
    zcm.publish("roll_reference", ref_msg)

    # Publish estimator timing info
    if est_call:
        if est_stats['return_status'] != 'Solve_Succeeded':
            print(bcolors.FAIL + "k = " + str(k) + ", Estimator returned \"" + est_stats['return_status'] + "\"." + bcolors.ENDC)

        name = estimator.ocp.name
        est_info = [
            estimator_call_duration,
            est_stats['iterations']['obj'][-1],
            est_stats['iter_count'],
            est_stats['t_proc_'+name],
            est_stats['t_wall_'+name]
        ]
        est_info_msg = timestamped_vector_double()
        est_info_msg.ts = timestamp
        est_info_msg.len = len(est_info)
        est_info_msg.values = est_info
        zcm.publish("estimator_info", est_info_msg)

        print("ESTIMATOR:")
        print("\tCall duration: " + str(est_info[0]))
        print("\tIterations: " + str(est_info[2]))
        print("\tt_proc: " + str(est_info[3]))
    
    
    # Publish controller timing info
    if ctrl_call:
        if ctrl_stats['return_status'] != 'Solve_Succeeded':
            print(bcolors.FAIL + "k = " + str(k) + ", Controller returned \"" + ctrl_stats['return_status'] + "\"." + bcolors.ENDC)

        name = controller.ocp.name
        info = [
            controller_call_duration,
            ctrl_stats['iterations']['obj'][-1],
            ctrl_stats['iter_count'],
            ctrl_stats['t_proc_'+name],
            ctrl_stats['t_wall_'+name]
        ]
        info_msg = timestamped_vector_double()
        info_msg.ts = timestamp
        info_msg.len = len(info)
        info_msg.values = info
        zcm.publish("controller_info", info_msg)

        print("CONTROLLER")
        print("\tCall duration: " + str(info[0]))
        print("\tIterations: " + str(info[2]))
        print("\tt_proc: " + str(info[3]))
    
    # Transmit a "1" if the deadline was missed, else transmit a "0"
    deadline_missed = (estimator_call_duration + controller_call_duration) > dt
    deadline_missed_msg = timestamped_vector_double()
    deadline_missed_msg.ts = timestamp
    deadline_missed_msg.len = 1
    deadline_missed_msg.values = [1] if deadline_missed else [0]
    zcm.publish("deadline_missed", deadline_missed_msg)

    # calculate sleep time
    k = k + 1
    nowTime = time.time()
    sleepTime = (startTime - nowTime) + k * dt

    # Make sure not to sleep negative time
    if sleepTime < 0:
        sleepTime = 0

    # sleep
    time.sleep(sleepTime)

zcm.stop()
quit(0)
