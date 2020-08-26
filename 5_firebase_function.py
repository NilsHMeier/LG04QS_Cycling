import copy
import math
from google.cloud import storage
import pandas as pd
import scipy.signal as signal
from scipy import special
import numpy as np
import io
import gcsfs
import pickle
import sklearn
from sklearn.svm import SVC


def predict(request):
    # Get ML model
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('sensordaten-d713c.appspot.com')
    blob = bucket.blob('ml_model/pickle_svm.sav')
    pickle_in = blob.download_as_string()
    best_svm = pickle.loads(pickle_in)
    print('ML Model:')
    print(best_svm)

    # Get CSV data
    raw_data = pd.read_csv('gs://sensordaten-d713c.appspot.com/file_to_process/sensordaten.csv')
    raw_data.columns = ['time', 'x', 'y', 'z', 'abs']
    raw_data = raw_data.drop(columns=['abs'])
    print('Data Head of csv file:')
    print(raw_data.head())
    # Get SPC from CSV
    SPC = 0.31  # TODO Read SPC from CSV
    f = 100  # TODO Read frequency from CSV
    # Process Offset
    start = 10
    end = raw_data['time'].max() - 10
    offset_data = raw_data[(raw_data['time'] >= start) & (raw_data['time'] <= end)]
    offset_data['time'] = offset_data['time'] - offset_data['time'].min()
    offset_data.index = offset_data.index - offset_data.index.min()
    print(offset_data.head())
    # Process Filter & SPC
    filter_data = copy.deepcopy(offset_data)
    sos = signal.butter(10, (5, 30), btype='bandpass', fs=f, output='sos')
    for col in ['x', 'y', 'z']:
        filter_data[col] = signal.sosfilt(sos, offset_data[col])
        filter_data[col] = filter_data[col] / (1 - SPC)

    # Create datatable for feature engineering
    max_time = int(filter_data['time'].max())
    period = 10
    aggregated_data = pd.DataFrame(list(range(0, max_time, period)), columns=['time'])
    for col in ['x', 'y', 'z']:
        aggregated_data[f'{col}_mean'] = np.nan
        aggregated_data[f'{col}_max'] = np.nan
        aggregated_data[f'{col}_min'] = np.nan
        aggregated_data[f'{col}_std'] = np.nan
    for timestamp in aggregated_data['time']:
        relevant_rows = raw_data[((raw_data['time'] >= timestamp) & (raw_data['time'] < timestamp + period))]
        relevant_rows.index = relevant_rows.index - relevant_rows.index.min()
        outlier_relevant = process_outlier_detection(relevant_rows, ['x', 'y', 'z'])
        for col in ['x', 'y', 'z']:
            aggregated_data.loc[timestamp / period, str(col) + str('_mean')] = np.mean(outlier_relevant[col])
            aggregated_data.loc[timestamp / period, str(col) + str('_max')] = np.max(outlier_relevant[col])
            aggregated_data.loc[timestamp / period, str(col) + str('_min')] = np.min(outlier_relevant[col])
            aggregated_data.loc[timestamp / period, str(col) + str('_std')] = np.std(outlier_relevant[col])

    print('Aggregated Datatable:')
    print(aggregated_data)
    # Predict underground for aggregated values
    predictions = best_svm.predict(aggregated_data.drop(columns=['time']))
    export_pred = pd.DataFrame(predictions, columns=['prediction'])
    print(export_pred)


def process_outlier_detection(data, cols):
    data_table = copy.deepcopy(data)
    for col in cols:
        # Compute the mean and standard deviation.
        mean = data_table[col].mean()
        std = data_table[col].std()
        criterion = 0.005
        # Consider the deviation for the data points.
        deviation = abs(data_table[col] - mean) / std
        # Express the upper and lower bounds.
        low = -deviation / math.sqrt(2)
        high = deviation / math.sqrt(2)
        prob = []
        mask = []
        for i in data_table.index:
            prob.append(1.0 - 0.5 * (special.erf(high[i]) - special.erf(low[i])))
            mask.append(prob[i] < criterion)
        data_table[col + '_outlier'] = mask
        data_table.loc[data_table[f'{col}_outlier'], col] = np.nan
        del data_table[col + '_outlier']
        data_table = impute_interpolate(data_table, col)
    data['x_out'] = data_table['x']
    data['y_out'] = data_table['y']
    data['z_out'] = data_table['z']
    return data_table


def impute_interpolate(dataset, col):
    dataset[col] = dataset[col].interpolate()
    dataset[col] = dataset[col].fillna(method='bfill')
    return dataset
