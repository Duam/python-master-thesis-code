import signal, time, datetime, argparse, csv
from thesis_code.utils.bcolors import bcolors
from zerocm import ZCM
from thesis_code.zcm_message_definitions.timestamped_vector_float import timestamped_vector_float
from thesis_code.zcm_message_definitions.timestamped_vector_double import timestamped_vector_double
from thesis_code.utils.zcm_constants import carousel_zcm_constants as carzcm
from thesis_code.utils.zcm_constants import Dataset
import os

# Today's date as a string
today = datetime.datetime.now().strftime("%Y_%m_%d")

# Parse arguments:
parser = argparse.ArgumentParser(description='Identification pipeline step 1: Data collection')
parser.add_argument(
  "-t", "--time",
  dest='capture_time', default='1',
  help="The total capture time"
)
parser.add_argument(
  "-s", "--signal-type",
  dest='signal_type',
  help="Actuation signal type. May be \"SS\", \"RECT\", or \"SINE\""
)
parser.add_argument(
  '-v', '--virtual',
  dest='is_virtual_experiment', default='True',
  help="Flag if this is a virtual experiment. If yes, the output file"
        " gets a \"VIRTUAL\" prefix, otherwise \"PHYSICAL\"."
)

args = parser.parse_args()

if not args.signal_type:
  print("Choose signal type (argument -s)")
  quit(0)

# Fetch arguments
totalTime = float(args.capture_time)
signal_type = args.signal_type
is_virtual_experiment = args.is_virtual_experiment == "True"

# Get file prefix
data_prefix  = ""
data_prefix += carzcm.prefix_virtual if is_virtual_experiment else carzcm.prefix_physical
data_prefix += today + "_"
data_prefix += "ACT_" + signal_type

out_path = ""
out_path += carzcm.path_data_sim if is_virtual_experiment else carzcm.path_data_phys
out_path += data_prefix + "/"

out_file_prefix  = out_path + data_prefix + "_"

print("Output file prefix: " + out_file_prefix)

# Create output folder
os.mkdir(out_path)
print("Created folder " + out_path)

""" ================================================================================================================ """
# Define datasets
ctrl = Dataset(out_file_prefix+"CONTROL.csv", name="control")
roll = Dataset(out_file_prefix+"ROLL.csv", name="roll")
pitch = Dataset(out_file_prefix+"PITCH.csv", name="pitch")
yaw = Dataset(out_file_prefix+"YAW.csv", name="yaw")
acc = Dataset(out_file_prefix+"LINEAR_ACCELERATION.csv", name="acc")
gyr = Dataset(out_file_prefix+"ANGULAR_VELOCITY.csv", name="gyr")
sim_state = Dataset(out_file_prefix+"SIMULATOR_STATE.csv", name="sim_state")

""" ================================================================================================================ """

use_receive_timestamp = True

# Define the ZCM subscription callbacks
def onNewControl(channel, message):
  global ctrl
  timestamp = int(time.time()*1e9) if use_receive_timestamp else int(message.ts)
  ctrl.data += [( timestamp, [float(message.values[0]) ] )]

def onNewRoll(channel, message):
  global roll
  timestamp = int(time.time()*1e9) if use_receive_timestamp else int(message.ts)
  roll.data += [( timestamp, [float(message.values[0]) ] )]

def onNewPitch(channel, message):
  global pitch
  timestamp = int(time.time() * 1e9) if use_receive_timestamp else int(message.ts)
  pitch.data += [( timestamp, [float(message.values[0])]  )]
  
def onNewYaw(channel, message):
  global yaw
  timestamp = int(time.time() * 1e9) if use_receive_timestamp else int(message.ts)
  yaw.data += [( timestamp, [float(message.values[0])] )]
  
def onNewLinearAcceleration(channel, message):
  global acc
  timestamp = int(time.time() * 1e9) if use_receive_timestamp else int(message.ts)
  acc.data += [( timestamp, [float(elem) for elem in message.values[0:3]] )]

def onNewAngularVelocity(channel, message):
  global gyr
  timestamp = int(time.time() * 1e9) if use_receive_timestamp else int(message.ts)
  gyr.data += [( timestamp, [float(elem) for elem in message.values[0:3]] )]

def onNewSimulatorState(channel, message):
  global sim_state
  timestamp = int(time.time() * 1e9) if use_receive_timestamp else int(message.ts)
  sim_state.data += [( timestamp, [float(elem) for elem in message.values[0:7]])]

# Initialize ZCM 
zcm = ZCM(carzcm.url)
if(not zcm.good()):
  raise Exception("ZCM not good")

# Subscribe to the channels
zcm.subscribe(carzcm.control_elevator_channel, timestamped_vector_float, onNewControl)
zcm.subscribe(carzcm.out_roll_channel, timestamped_vector_float, onNewRoll)
zcm.subscribe(carzcm.out_pitch_channel, timestamped_vector_float, onNewPitch)
zcm.subscribe(carzcm.out_yaw_channel, timestamped_vector_double, onNewYaw)
zcm.subscribe(carzcm.out_acc_channel, timestamped_vector_float, onNewLinearAcceleration)
zcm.subscribe(carzcm.out_gyr_channel, timestamped_vector_float, onNewAngularVelocity)
if is_virtual_experiment:
  zcm.subscribe("CAR_SIM_STATE", timestamped_vector_double, onNewSimulatorState)

""" ================================================================================================================ """

# Create SIGINT handler for quick stop
def signal_handler(sig, frame):
  print('Ctrl+C: Exiting.')
  zcm.stop()
  quit(0)
  
# Register signal
signal.signal(signal.SIGINT, signal_handler)

""" ================================================================================================================ """

# Capture for T seconds
print ("Capturing for " + str(totalTime) + " seconds ..")
# Start zcm
zcm.start()
# Sleep
time.sleep(totalTime)
# Wake up
zcm.stop()
print("Stop.")

# Check data
do_abort = False
for dataset in ctrl,roll,pitch,yaw,acc,gyr:
  # Get dataset
  filename = dataset.filename
  name = dataset.name
  data = dataset.data

  if len(data) == 0:
    print(bcolors.FAIL + "Dataset \"" + name + "\" is empty!" + bcolors.ENDC)
    do_abort = True

if do_abort:
  print("One or more datasets were empty. Aborting.")
  quit(0)

""" ================================================================================================================ """

# Store inputs in csv file
print("Storing data as csv..")

for dataset in ctrl,roll,pitch,yaw,acc,gyr:
  # Get dataset
  filename = dataset.filename
  name = dataset.name
  data = dataset.data

  # Open csv file
  with open(filename, 'w') as outfile:

    # Determine column names
    firstRow = [ 'timestamp' ]
    if len(data[0][1]) > 1:
      for k in range(len(data[0][1])):
        firstRow += [ name+"_"+str(k) ]
    else:
      firstRow += [ name ]

    # Write first row (metadata)
    writer = csv.writer(outfile)
    writer.writerow(firstRow)

    # Write actual data
    for k in range(len(data)):
      timestamp = data[k][0]
      values = data[k][1]
      writer.writerow([timestamp,*values])

  print(name + " written to " + filename)

if is_virtual_experiment:
  for dataset in [sim_state]:
    # Get dataset
    filename = dataset.filename
    name = dataset.name
    data = dataset.data

    # Open csv file
    with open(filename, 'w') as outfile:

      # Determine column names
      firstRow = [ 'timestamp' ]
      if len(data[0][1]) > 1:
        for k in range(len(data[0][1])):
          firstRow += [ name+"_"+str(k) ]
      else:
        firstRow += [ name ]

      # Write first row (metadata)
      writer = csv.writer(outfile)
      writer.writerow(firstRow)

      # Write actual data
      for k in range(len(data)):
        timestamp = data[k][0]
        values = data[k][1]
        writer.writerow([timestamp,*values])

    print(name + " written to " + filename)
