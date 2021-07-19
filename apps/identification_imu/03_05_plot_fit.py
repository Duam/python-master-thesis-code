import matplotlib.pyplot as plt
import argparse
import pandas as pd

# Parse arguments:
parser = argparse.ArgumentParser(description='Identification pipeline step 3.5: Plotting')
parser.add_argument(
  '-p', '--path',
  dest='path',
  default="./DATA.zcmlog",
  help="Local path of the preprocessed datafile (csv file)"
)

# Set path
args = parser.parse_args()
path = args.path
print("Loading dataset " + path)

""" ================================================================================================================ """

# Set validation data input filename
y_est_filename = path + ".PREPROCESSED.csv.ESTIMATED_MEASUREMENTS.csv"
y_filename = path + ".PREPROCESSED.csv"

# Load recorded data
print("Loading dataset of recorded measurements " + y_filename)
Ys = pd.read_csv(y_filename, header=0, parse_dates=[0])
print("Done. Loaded " + str(len(Ys)) + " data points.")

# Load estimated data
print("Loading dataset of estimated measurements " + y_est_filename)
Ys_est = pd.read_csv(y_est_filename, header=0, parse_dates=[0])
print("Done. Loaded " + str(len(Ys_est)) + " data points.")

""" ================================================================================================================ """

# We plot N samples
N = min([ len(Ys), len(Ys_est) ])

print("Recorded measurements info: " + str(Ys.info()))
print("Estimated measurements info: " + str(Ys_est.info()))

# Plot result

print("Plotting results..")
fig, ax = plt.subplots(3,1,sharex='all')
plt.sca(ax[0])
plt.title('Controls')
#plt.plot(Ys_est['state_estimate_4'])
plt.plot()

plt.sca(ax[1])
plt.title('Roll')
plt.plot(Ys_est['roll'][:N], label="Estimated")
plt.plot(Ys['VE1_SCAPULA_ELEVATION_0'][:N], label="Real")
plt.legend()

plt.sca(ax[2])
plt.title('Pitch')
plt.plot(Ys_est['pitch'][:N], label="Estimated")
plt.plot(Ys['VE1_SCAPULA_ROTATION_0'][:N], label="Real")
plt.legend()

plt.show()
quit(0)
