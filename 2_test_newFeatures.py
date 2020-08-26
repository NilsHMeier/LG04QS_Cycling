import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PreProcessing.FeatureEngineering import FeatureEngineering

SHOW_PLOT = False
shown_types = {'AS': False, 'KO': False, 'RW': False, 'SC': False, 'WW': False}
SOURCE_PATH = 'Data/ProcessedData/'
frequency_features = ['mean_PSD', 'max_PSD_f', 'max_PSD', 'PSD_above_20', 'weighted_mean']
cols = ['x', 'y', 'z']
features = list(col + '_' + feature for col in cols for feature in frequency_features)
features.append('surface')
feature_datatable = pd.DataFrame(columns=features)

period = 10
low_freq = 10
high_freq = 30
names = ['NM']
color_lookup = {'AS': 'r', 'RW': 'b', 'SC': 'y', 'WW': 'g', 'KO': 'c'}

for file in os.listdir(SOURCE_PATH):
    if len(names) > 0 and file[3:5] not in names:
        continue
    data = pd.read_csv(SOURCE_PATH + file, index_col=0)
    data_table = FeatureEngineering.create_dataset(cols, period, int(data['time'].max()))
    for timestamp in data_table['time']:
        relevant_rows = data[((data['time'] >= timestamp) & (data['time'] < timestamp + period))]
        generated_features = {'surface': file[0:2]}
        for col in cols:
            values = relevant_rows[col]
            dt = 0.005 if file[3:5] == 'NB' else 0.01
            n = len(relevant_rows['time'])
            fhat = np.fft.fft(values, n)
            PSD = fhat * np.conj(fhat) / n
            freq = (1 / (dt * n)) * np.arange(n)
            L = np.arange(1, np.floor(n / 2), dtype='int')
            if SHOW_PLOT:
                plt.scatter(freq[L], PSD[L])
                plt.title(f'FFT of {file} from {timestamp} to {timestamp + period}')
                plt.show()
            # Create Dataframe from data
            freq_data = pd.DataFrame(data={'freq': freq, 'PSD': PSD, 'FHAT': fhat})
            freq_data_relevant = freq_data[((freq_data['freq'] >= low_freq) & (freq_data['freq'] <= high_freq))]
            freq_data_relevant.index = freq_data_relevant.index - np.min(freq_data_relevant.index)
            freq_data_relevant['weighted'] = freq_data_relevant['freq'] * freq_data_relevant['PSD']
            # Add features to dictionary to append to dataframe
            generated_features[col + '_mean_PSD'] = np.mean(freq_data_relevant['PSD']).real
            generated_features[col + '_max_PSD_f'] = freq_data_relevant.loc[
                np.argmax(freq_data_relevant['PSD']), 'freq']
            generated_features[col + '_max_PSD'] = np.max(freq_data_relevant['PSD']).real
            generated_features[col + '_PSD_above_20'] = \
                sum(1 if psd.real > 20 else 0 for psd in freq_data_relevant['PSD'])
            generated_features[col + '_weighted_mean'] = np.mean(freq_data_relevant['weighted']).real
        # Append dictionary to the dataframe
        feature_datatable = feature_datatable.append(generated_features, ignore_index=True)

surface_feature = {surface: feature_datatable[feature_datatable['surface'] == surface] for surface in color_lookup}

# Plot with 3 features out of 12
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_axis = 'y_weighted_mean' #'z_PSD_above_20'
y_axis = 'z_PSD_above_20' #'y_mean_PSD'
z_axis = 'z_mean_PSD' #'x_mean_PSD'
for surface in color_lookup:
    surface_data = surface_feature[surface]
    ax.scatter(surface_data[x_axis], surface_data[y_axis], surface_data[z_axis],
               color=color_lookup[surface], label=surface)
plt.legend()
plt.title('Frequency Features')
ax.set_xlabel(x_axis)
# ax.set_xlim(0, 150)
ax.set_ylabel(y_axis)
# ax.set_ylim(0, 250)
ax.set_zlabel(z_axis)
# ax.set_zlim(0, 600)
plt.show()

plt.subplot(111)
for surface in color_lookup:
    surface_data = surface_feature[surface]
    plt.scatter(surface_data[z_axis], surface_data[x_axis], color=color_lookup[surface], label=surface)
plt.legend()
plt.show()
print(type(values))
