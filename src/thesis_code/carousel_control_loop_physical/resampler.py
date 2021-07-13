import time, signal
import numpy as np
from zerocm import ZCM
from thesis_code.zcm_message_definitions import timestamped_vector_double, timestamped_vector_float
from thesis_code.utils.zcm_constants import carousel_zcm_constants as carzcm

current_time = int(time.time()*1e9)
current_control = [0.0]
current_roll_meas = [3.14]
current_pitch_meas = [0.9]
current_roll = [0.0]
current_pitch = [0.0]
current_yaw = [0.0]
current_yaw_sin = [0.0]
current_yaw_cos = [1.0]
current_acc = [0.0, 0.0, 0.0]
current_gyr = [0.0, 0.0, 0.0]

# Create a moving average filter for acceleration and gyro measurements
dt_sam = 1./50.
dt_imu = 1./1000.
N_MAF = 50
k_acc_MAF = 0
acc_MAF_buffer = np.zeros((3,N_MAF))
k_gyro_MAF = 0
gyro_MAF_buffer = np.zeros((3,N_MAF))

def publishMeasurement():
    global zcm, current_time, current_control, current_roll, current_pitch, current_yaw, current_acc, current_gyr
    global currect_yaw_sin, current_yaw_cos
    global acc_MAF_buffer, gyro_MAF_buffer
    timestamp = int(time.time()*1e9)

    control_msg = timestamped_vector_double()
    control_msg.ts = timestamp
    control_msg.len = 1
    control_msg.values = current_control

    measurement1_msg = timestamped_vector_float()
    measurement1_msg.ts = timestamp
    measurement1_msg.values = np.mean(acc_MAF_buffer, axis=1).tolist() + np.mean(gyro_MAF_buffer, axis=1).tolist()
    measurement1_msg.len = len(measurement1_msg.values)

    measurement2_msg = timestamped_vector_float()
    measurement2_msg.ts = timestamp
    measurement2_msg.values = current_roll_meas + current_pitch_meas
    measurement2_msg.len = len(measurement2_msg.values)

    angles_msg = timestamped_vector_float()
    angles_msg.ts = timestamp
    angles_msg.values = current_roll + current_pitch
    angles_msg.len = len(angles_msg.values)
    
    zcm.publish(carzcm.controls_sampled_channel, control_msg)
    zcm.publish(carzcm.measurements_sampled_channel, measurement1_msg)
    zcm.publish("ANGLES_meas_sampled", measurement2_msg)
    zcm.publish("ANGLES_sampled", angles_msg)
    

def onNewControl(channel, message):
    global current_control
    current_control = [*message.values]

def onNewRoll(channel, message):
    global current_roll, current_roll_meas
    y_roll = message.values[0]
    current_roll_meas = [y_roll]
    current_roll = [np.pi - y_roll]

def onNewPitch(channel, message):
    global current_pitch, current_pitch_meas
    y_pitch = message.values[0]
    current_pitch_meas = [y_pitch]
    current_pitch = [0.9265 - y_pitch]

def onNewYaw(channel, message):
    global current_yaw, current_yaw_cos, current_yaw_sin
    current_yaw = [*message.values]
    current_yaw_sin = [*np.sin(current_yaw)]
    current_yaw_cos = [*np.cos(current_yaw)]

def onNewLinearAcceleration(channel, message):
    global current_acc, acc_MAF_buffer, k_acc_MAF
    current_acc = [*message.values]
    acc_MAF_buffer[:,k_acc_MAF] = np.array(current_acc)
    k_acc_MAF = np.mod(k_acc_MAF + 1, N_MAF)

def onNewAngularVelocity(channel, message):
    global current_gyr, gyro_MAF_buffer, k_gyro_MAF
    current_gyr = [*message.values]
    gyro_MAF_buffer[:,k_gyro_MAF] = np.array(current_gyr)
    k_gyro_MAF = np.mod(k_gyro_MAF + 1, N_MAF)

if __name__ == '__main__':
    # Create ZCM
    zcm = ZCM()
    if not zcm.good():
        raise RuntimeError('ZCM not good')

    # Create SIGINT handler for quick stop
    def signal_handler(sig, frame):
        print('Ctrl+C: Exiting.')
        zcm.stop()
        quit(0)

    # Register signal
    signal.signal(signal.SIGINT, signal_handler)

    # Subscribe to channels
    zcm.subscribe(carzcm.control_elevator_channel, timestamped_vector_float, onNewControl)
    zcm.subscribe(carzcm.out_roll_channel, timestamped_vector_float, onNewRoll)
    zcm.subscribe(carzcm.out_pitch_channel, timestamped_vector_float, onNewPitch)
    zcm.subscribe(carzcm.out_yaw_channel, timestamped_vector_double, onNewYaw)
    zcm.subscribe(carzcm.out_acc_channel, timestamped_vector_float, onNewLinearAcceleration)
    zcm.subscribe(carzcm.out_gyr_channel, timestamped_vector_float, onNewAngularVelocity)

    period = 1. / 20.

    # Get startup time
    startTime = time.time()
    # Initialize iterations
    k = 0
    # Simulation main loop
    lastTime = startTime
    print("Start.")
    zcm.start()
    while True:

        # Publish measurements
        publishMeasurement()

        # calculate sleep time
        k = k + 1
        nowTime = time.time()
        sleepTime = (startTime - nowTime) + k * period

        # Make sure not to sleep negative time
        if sleepTime < 0:
            sleepTime = 0

        # sleep
        time.sleep(sleepTime)
