import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PreProcessing.FeatureEngineering import FeatureEngineering
from PreProcessing.PreProcess import PreProcess

datasets = []
names = []
colors = ['red', 'green', 'blue']
gravel_data = pd.read_csv('../Data/CuttedData/SC_NM_Hinter Gut Brockwinkel.csv')
gravel_data.columns = ['time', 'x', 'y', 'z', 'abs']
datasets.append(gravel_data)
names.append('Gravel')
asphalt_data = pd.read_csv('../Data/CuttedData/AS_NM_Brockwinkel nach Reppenstedt.csv')
asphalt_data.columns = ['time', 'x', 'y', 'z', 'abs']
datasets.append(asphalt_data)
names.append('Asphalt')
cobblestone_data = pd.read_csv('../Data/CuttedData/KO_NM_Salzbrücker Straße.csv')
cobblestone_data.columns = ['time', 'x', 'y', 'z', 'abs']
datasets.append(cobblestone_data)
names.append('Cobblestone')

fig, axs = plt.subplots(3, sharex=True, sharey=False)
for i in range(len(datasets)):
    data = datasets[i]
    data = PreProcess.process_filter(data, 100, ['x', 'y', 'z'])
    values = data['y']
    dt = 0.01
    n = len(data['time'])
    # n = int(time.max())
    fhat = np.fft.fft(values, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(1, np.floor(n / 2), dtype='int')
    axs[i].scatter(freq[L], PSD[L], color=colors[i],  label=f'FFT {names[i]}')
plt.show()
cols = ['x', 'y', 'z']
period = 10
for i in range(len(datasets)):
    raw_data = datasets[i]
    raw_data = PreProcess.process_filter(raw_data, 100, cols)
    data_table = FeatureEngineering.create_dataset(cols, period, int(raw_data['time'].max()))
    for timestamp in data_table['time']:
        relevant_rows = raw_data[((raw_data['time'] >= timestamp) & (raw_data['time'] < timestamp + period))]
        values = relevant_rows['y']
        dt = 0.01
        n = len(relevant_rows['time'])
        # n = int(time.max())
        fhat = np.fft.fft(values, n)
        PSD = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)
        L = np.arange(1, np.floor(n / 2), dtype='int')
        plt.scatter(freq[L], PSD[L])
        plt.title(f'FFT of {names[i]} from {timestamp} to {timestamp+period}')
        plt.show()
