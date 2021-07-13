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

""" ================================================================================================================ """

# Fetch arguments
args = parser.parse_args()
totalTime = float(args.capture_time)
signal_type = args.signal_type

# Initialize ZCM
zcm = ZCM(carzcm.url)
if(not zcm.good()):
  raise Exception("ZCM not good")

""" ================================================================================================================ """

# Create SIGINT handler for quick stop
def signal_handler(sig, frame):
    print('Ctrl+C: Exiting.')
    zcm.stop()
    quit(0)


# Register signal
signal.signal(signal.SIGINT, signal_handler)

""" ================================================================================================================ """

dt = 1/50.
N_approx = int(totalTime/dt)
M = 10

def rectangle_excitation(k,N):
  """rectangle_excitation A rectangle signal that starts fast and goes slower with time
  Args:
    k[int] -- The current sample number
    N[int] -- The number of samples after which the rectangle switches to a slower mode
  Returns:
    A rectangle signal
  """
  return rectangle(k, 2**int(k/float(N)))

""" ================================================================================================================ """

# Capture for T seconds
print("Sendinc controls for " + str(totalTime) + " seconds ..")
# Start zcm
zcm.start()

# Get startup time
startTime = time.time()
lastTime = startTime
# Initialize iterations
k = 0
# Simulation main loop
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
            u = rectangle_excitation(N_approx - k - steadySamples, int(N_approx / M))
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