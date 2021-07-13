import thesis_code.utils.data_processing_tools as tools
import argparse

""" ================================================================================================================ """
# Parse arguments:
parser = argparse.ArgumentParser(description='Identification pipeline step 2: Preprocessing')
parser.add_argument(
  '-p', '--path',
  dest='path',
  default="./DATA.zcmlog",
  help="Local path of the datafile (zcmlog file)"
)

# Fetch arguments
args = parser.parse_args()
# Set input file prefix
path = args.path
print("Loading dataset " + path)

# Set output file prefix
out_filename  = path + ".PREPROCESSED.csv"
print("Result will be stored in dataset " + out_filename)

# Start preprocessing
dataset = tools.load_log_to_dataframe(path, 
  channels = [
    'VE2_KIN4_SET',
    'VE1_SCAPULA_ELEVATION', 
    'VE1_SCAPULA_ROTATION', 
    'VE2_MPU_ACC', 
    'VE2_MPU_GYRO', 
    'MEASUREMENTS_sampled'
  ]
)
dataset = tools.smooth_data(dataset, 20, channels = ['VE2_MPU_ACC', 'VE2_MPU_GYRO'])
dataset = tools.resample_data(dataset, "50ms")
#dataset = tools.resample_data(dataset, "500ms")
dataset = tools.join_and_trim_data(dataset)
print(dataset.info())
print("Roll measurement mean = " + str(dataset['VE1_SCAPULA_ELEVATION_0'].mean()))
print("Pitch measurement mean = " + str(dataset['VE1_SCAPULA_ROTATION_0'].mean()))
print("Acceleration mean (x) = " + str(dataset['VE2_MPU_ACC_0'].mean()))
print("Acceleration mean (y) = " + str(dataset['VE2_MPU_ACC_1'].mean()))
print("Acceleration mean (z) = " + str(dataset['VE2_MPU_ACC_2'].mean()))
print("Gyroscope mean (x) = " + str(dataset['VE2_MPU_GYRO_0'].mean()))
print("Gyroscope mean (y) = " + str(dataset['VE2_MPU_GYRO_1'].mean()))
print("Gyroscope mean (z) = " + str(dataset['VE2_MPU_GYRO_2'].mean()))

# Write to CSV
dataset.to_csv(out_filename)
print("Written processed data into file " + out_filename)
