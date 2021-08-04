from zerocm import LogFile
import pandas as pd


def load_log_to_dataframe(filename, channels):
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

    # Return all the data
    return dataset

def smooth_data(dataset, N, channels=None):
    if channels == None:
        channels = dataset.keys()
    print("Applying smoothing.. N="+str(N) + " to fields " + str(channels))
    for key in channels:
        # Fetch timestamps and values
        data = dataset[key]
        # Apply filter
        """ filtered_data = sum([ data.shift(k) for k in range(N) ]) / float(N) """
        filtered_data = data.rolling(N, win_type=None).mean()
        # Write back filtered data
        dataset[key] = filtered_data.dropna()

    # Return smoothed data
    return dataset

def resample_data(dataset, dt):
    print("Resampling.. dt="+dt)
    for key in dataset.keys():
        dataset[key] = dataset[key].resample(dt,closed='left').first().ffill()
    return dataset

def join_and_trim_data(dataset):
    print("Joining data..")
    dataset = pd.DataFrame().join(dataset.values(), how='outer')
    print("Trimming data..")
    dataset = dataset.dropna()
    return dataset
