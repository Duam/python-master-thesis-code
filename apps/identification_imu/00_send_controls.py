import signal, time, argparse
from thesis_code.utils.signals import rectangle
import numpy as np
from zerocm import ZCM
from thesis_code.zcm_message_definitions.timestamped_vector_float import timestamped_vector_float
from thesis_code.utils.zcm_constants import carousel_zcm_constants as carzcm

# Parse arguments:
parser = argparse.ArgumentParser(description='Identification pipeline step 1: Data collection')
parser.add_argument(
  "-t", "--time",
  dest='capture_time', default='1',
  help="The total sending time"
)
parser.add_argument(
  "-s", "--signal-type",
  dest='signal_type', default='SS',
  help="Actuation signal type. May be \"SS\", \"RECT\", or \"SINE\""
)

# Fetch arguments
args = parser.parse_args()
totalTime = float(args.capture_time)
signal_type = args.signal_type

# Initialize ZCM
zcm = ZCM(carzcm.url)
if not zcm.good():
  raise Exception("ZCM not good")

def signal_handler(sig, frame):
    print('Ctrl+C: Exiting.')
    zcm.stop()
    quit(0)

signal.signal(signal.SIGINT, signal_handler)

dt = 1/50.
N_approx = int(totalTime/dt)
M = 10

# Capture for T seconds
print("Sendinc controls for " + str(totalTime) + " seconds ..")
zcm.start()
startTime = time.time()
lastTime = startTime
k = 0
print("Start.")
while True:

    # Create control
    nowTime = time.time()
    timestamp = int(nowTime * 1e9)

    steadyTime = 1.0
    steadySamples = int(steadyTime / dt)

    if nowTime - startTime < steadyTime:
        # Send steady state control for some time
        u = 0.5

    else:
        if signal_type == "SS":
            u = 0.5
        elif signal_type == "RECT":
            # A rectangle signal that starts fast and goes slower with time
            current_sample = N_approx - k - steadySamples
            num_samples_until_slowdown = int(N_approx / M)
            rectangle(current_sample, 2 ** int(current_sample / float(num_samples_until_slowdown)))
        elif signal_type == "SINE":
            u = 0.5 + 0.5 * np.sin(k * dt - steadyTime)
        else:
            print("Invalid signal type.")
            u = 0.5

    # Publish control
    control_msg = timestamped_vector_float()
    control_msg.ts = timestamp
    control_msg.len = 1
    control_msg.values = [u]
    zcm.publish(carzcm.control_elevator_channel, control_msg)

    # calculate sleep time
    k = k + 1
    sleepTime = (startTime - nowTime) + k * dt

    # Make sure not to sleep negative time
    if sleepTime < 0:
        sleepTime = 0

    # sleep
    time.sleep(sleepTime)

    # Break out of loop when we're done
    if nowTime > startTime + totalTime:
        break

zcm.stop()
print("Stop.")