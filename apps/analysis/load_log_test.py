from zerocm import LogFile
import matplotlib.pyplot as plt
import pandas as pd
pd.plotting.register_matplotlib_converters()
from thesis_code.model import CarouselModel

# Fetch params
model_params = CarouselModel.getDefaultParams()
model_constants = CarouselModel.getConstants()
offsets = {
  'VE1_SCAPULA_ELEVATION': model_constants['roll_sensor_offset'],
  'VE1_SCAPULA_ROTATION': model_constants['pitch_sensor_offset']
}

#cases = ['idle', 'mhe', 'mpc', 'mhe_mpc']
#case = 'idle'
case = 'mhe'

# Open and load logfile
if case == 'idle':
  filename = 'data/zcmlog-2019-09-18-idle-data-for-variance-analysis.0001'
  filename = 'data/zcmlog-2019-09-20-idle-data-for-variance-analysis-MAF50.0000'
  channels = ['VE2_KIN4_SET', 
              'VE1_SCAPULA_ELEVATION', 'VE1_SCAPULA_ROTATION', 'VE1_MPU_ACC', 'VE1_MPU_GYRO',
              'VE2_MPU_ACC', 'VE2_MPU_GYRO', 'MEASUREMENTS_sampled']
elif case == 'mhe':
  #filename = './data/zcmlog-2019-09-18-mhe-test-with-measured-angles.0000'
  filename = 'data/zcmlog-2019-09-19-mhe-test-with-gyro.0000'
  filename = 'data/zcmlog-2019-09-19-mhe-test-with-gyro.0003'
  filename = 'data/zcmlog-2019-09-19-mhe-test-with-accelerometer.0000'
  filename = 'data/zcmlog-2019-09-19-mhe-test-with-acc-and-gyro.0004'
  #filename = './data/zcmlog-2019-09-23-mhe-test-with-acc-and-gyro-jit_R1e-3.0017'
  filename = 'data/zcmlog-2019-09-29_ss_playback_mhe_imu.0002'
  #filename = './data/zcmlog-2019-09-29_rect_playback_mhe_imu.0002'
  filename = 'data/zcmlog-2019-09-29_sine_playback_mhe_imu.0002'
  channels = ['VE2_KIN4_SET', 'VE1_SCAPULA_ELEVATION', 'VE1_SCAPULA_ROTATION', 'state_estimate']
else:
  filename = "NO_CASE_SPECIFIED"
  channels = []

# Load data
print("Loading file \"" + filename + "\"")
file = LogFile(filename, 'r')
quit(0)

# TODO(paul): Somehow load and decode events
arrays = load_and_decode_events(file, channels)

# Print fields
print("Data fields:")
for key in arrays.keys():
  arr = arrays[key]
  print(key + ": " + str(len(arr)) + " msgs, " + str(len(arr[0].message.values)) + " fields each")

# Put data in a dict of pandas arrays
dataset = {}
for key in arrays.keys():
  arr = arrays[key]
  N = len(arr)
  values = [[elem.event_ts] + list(elem.message.values) for elem in arr]
  n = len(values[0])
  dataset[key] = pd.DataFrame(
    index=range(N),
    columns=['timestamp',*[key+'_'+str(k) for k in range(n-1)]],
    data=values
  )
  dataset[key]['timestamp'] = pd.to_datetime(dataset[key]['timestamp'])
  
# Reorder according to timestamp
print("Ordering.. (ascending timestamp)")
for key in dataset.keys():
  dataset[key] = dataset[key].sort_values(by='timestamp').set_index('timestamp')

# Print mean and variances
print("Printing statistics..")
for key in dataset.keys():
  data = dataset[key]
  mean = data.mean().values
  variance = data.var().values
  print(key + ": mean = " + str(mean) + ", variance = " + str(variance))

# Convert angle measurements
for key in ['VE1_SCAPULA_ELEVATION', 'VE1_SCAPULA_ROTATION']:
  offs = offsets[key]
  dataset[key] = offs - dataset[key]

# Resample (zero-order-hold)
dt = "50ms"
print("Resampling.. dt="+dt)
for key in dataset.keys():
  dataset[key] = dataset[key].resample(dt,closed='left').first().ffill()

# Join dataframes such that timestamps align
print("Joining data..")
dataset = pd.DataFrame().join(dataset.values(), how='outer')
dataset.index = (dataset.index - dataset.index[0]).total_seconds()

# Trim dataframes so no NaNs are at the edges
print("Trimming data..")
dataset = dataset.dropna()

# Print out dataset
#print(dataset)

# Plot state estimate vs. real data
if case == 'mhe':
  print("Plotting state estimate vs. truth..")

  # Fetch data
  dt = 0.05
  T = 30
  N = min([int(T / dt), len(data)])
  dataset = dataset.head(N)
  controls_real = dataset['VE2_KIN4_SET_0']
  tab_angle_esti = dataset['state_estimate_4']
  roll_real = dataset['VE1_SCAPULA_ELEVATION_0']
  roll_esti = dataset['state_estimate_0']
  pitch_real = dataset['VE1_SCAPULA_ROTATION_0']
  pitch_esti = dataset['state_estimate_1']

  # Compute mse ('mean' part only relevant if I do multiple trials)
  roll_err = roll_real - roll_esti
  roll_mse = (roll_err ** 2) ** 0.5
  pitch_err = pitch_real - pitch_esti
  pitch_mse = (pitch_err ** 2) ** 0.5

  # Plot
  fig, ax = plt.subplots(3,2,sharex='all')
  #plt.suptitle('First carousel state estimation test \n MHE (N=3), constant controls')
  plt.sca(ax[0,0])
  plt.title('Controls')
  plt.plot(controls_real, label='Setpoint')
  plt.plot(tab_angle_esti, label='Estimated')
  plt.legend(loc='lower right').draggable()
  plt.sca(ax[1,0])

  plt.title('Roll')
  plt.plot(roll_esti, label='Estimated')
  plt.plot(roll_real, label='Real')
  plt.ylabel('$\phi(t)$ [rad]')
  plt.legend(loc='lower right').draggable()

  plt.sca(ax[1,1])
  plt.title('Roll MSE')
  plt.plot(roll_mse)
  plt.yscale('log')

  plt.sca(ax[2,0])
  plt.title('Pitch')
  plt.plot(pitch_esti, label='Estimated')
  plt.plot(pitch_real, label='Real')
  plt.ylabel('$\\theta(t)$ [rad]')
  plt.xlabel('Time $t$ [s]')
  #plt.legend(loc='best')

  plt.sca(ax[2,1])
  plt.title('Pitch MSE')
  plt.plot(pitch_mse)
  plt.xlabel('Time $t$ [s]')
  plt.yscale('log')

  plt.show()

