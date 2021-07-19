import signal, time, datetime, argparse, csv
from thesis_code.utils.bcolors import bcolors
from zerocm import ZCM
from thesis_code.zcm_message_definitions.timestamped_vector_float import timestamped_vector_float
from thesis_code.zcm_message_definitions.timestamped_vector_double import timestamped_vector_double
from apps.identification_imu.carousel_identification import carousel_zcm_constants as carzcm
from apps.identification_imu.carousel_identification import Dataset

# Today's date as a string
today = datetime.datetime.now().strftime("%Y_%m_%d")

# Parse arguments:
parser = argparse.ArgumentParser(description='Data collection script for physical_experiment experiments')
parser.add_argument(
    "-t", "--time",
    dest='capture_time', default='1',
    help="The total capture time"
)
parser.add_argument(
    '-v', '--virtual_experiment',
    dest='is_virtual_experiment', default='True',
    help="Flag if this is a virtual_experiment experiment. If yes, the output file"
         " gets a \"VIRTUAL\" prefix, otherwise \"PHYSICAL\"."
)
args = parser.parse_args()

# Fetch arguments
totalTime = float(args.capture_time)
is_virtual_experiment = args.is_virtual_experiment == "True"
# Get file prefix
file_prefix = carzcm.get_file_prefix(
    is_virtual=is_virtual_experiment,
    is_live_experiment=True,
    is_identification_data=False,
    pipeline_step=1
)
file_prefix += today + "_"
print("File prefix: " + file_prefix)

""" ================================================================================================================ """
# Define datasets
ctrl = Dataset(file_prefix + "CONTROL.csv", name="control")
roll = Dataset(file_prefix + "ROLL.csv", name="roll")
pitch = Dataset(file_prefix + "PITCH.csv", name="pitch")
yaw = Dataset(file_prefix + "YAW.csv", name="yaw")
acc = Dataset(file_prefix + "LINEAR_ACCELERATION.csv", name="acc")
gyr = Dataset(file_prefix + "ANGULAR_VELOCITY.csv", name="gyr")

""" ================================================================================================================ """


# Define the ZCM subscription callbacks
def onNewControl(channel, message):
    global ctrl
    ctrl.data += [(int(message.ts), [float(message.values[0])])]


def onNewRoll(channel, message):
    global roll
    roll.data += [(int(message.ts), [float(message.values[0])])]


def onNewPitch(channel, message):
    global pitch
    pitch.data += [(int(message.ts), [float(message.values[0])])]


def onNewYaw(channel, message):
    global yaw
    yaw.data += [(int(message.ts), [float(message.values[0])])]


def onNewLinearAcceleration(channel, message):
    global acc
    acc.data += [(int(message.ts), [float(elem) for elem in message.values[0:3]])]


def onNewAngularVelocity(channel, message):
    global gyr
    gyr.data += [(int(message.ts), [float(elem) for elem in message.values[0:3]])]


# Initialize ZCM
zcm = ZCM(carzcm.url)
if (not zcm.good()):
    raise Exception("ZCM not good")

# Subscribe to the channels
zcm.subscribe(carzcm.control_channel, timestamped_vector_double, onNewControl)
zcm.subscribe(carzcm.out_roll_channel, timestamped_vector_float, onNewRoll)
zcm.subscribe(carzcm.out_pitch_channel, timestamped_vector_float, onNewPitch)
zcm.subscribe(carzcm.out_yaw_channel, timestamped_vector_float, onNewYaw)
zcm.subscribe(carzcm.out_acc_channel, timestamped_vector_float, onNewLinearAcceleration)
zcm.subscribe(carzcm.out_gyro_channel, timestamped_vector_float, onNewAngularVelocity)

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
print("Capturing for " + str(totalTime) + " seconds ..")
# Start zcm, let this thread sleep a while, then stop
zcm.start()
time.sleep(totalTime)
zcm.stop()
print("Stop.")

# Check data
do_abort = False
for dataset in ctrl, roll, pitch, yaw, acc, gyr:
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

for dataset in ctrl, roll, pitch, yaw, acc, gyr:
    # Get dataset
    filename = dataset.filename
    name = dataset.name
    data = dataset.data

    # Open csv file
    with open(filename, 'w') as outfile:

        # Determine column names
        firstRow = ['timestamp']
        if len(data[0][1]) > 1:
            for k in range(len(data[0][1])):
                firstRow += [name + "_" + str(k)]
        else:
            firstRow += [name]

        # Write first row (metadata)
        writer = csv.writer(outfile)
        writer.writerow(firstRow)

        # Write actual data
        for k in range(len(data)):
            timestamp = data[k][0]
            values = data[k][1]
            writer.writerow([timestamp, *values])

    print(name + " written to " + filename)
