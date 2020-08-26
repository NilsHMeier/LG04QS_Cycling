# Import required Libraries
import copy

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import os
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def predict_road_surface_derivation(file, f=200, cut_sample=False, show_plot=False):
    # Increase figure size for bigger plots

    # plt.rcParams['figure.figsize'] = [18, 14]
    # plt.rcParams.update({'font.size': 18})

    # Import CSV data and rename the columns

    sample_data = pd.read_csv(file)
    sample_data.columns = ['time', 'x', 'y', 'z', 'abs']
    if show_plot:
        print(sample_data)

    # Remove first and last 12 seconds to exclude useless data

    if cut_sample:
        sample_data = sample_data[sample_data.time > 15]
        sample_data = sample_data[sample_data.time < sample_data.time.iat[-1] - 15]

    # ## Plotting prepared data

    if show_plot:
        plt.plot(sample_data.time, sample_data.z)
        plt.show()

    # Run Fast Fourier Transform on x-Accelerometer from the dataset

    # assigning sample_data to fftData, so we can keep the original noisy data for later.
    fftData = copy.deepcopy(sample_data)

    # n is the length of the xAxis column and represents how many datapoints we have.
    n = len(fftData.z)

    # dt represents the frequency of the Smartphone Accelerometer sensor. Testing sensor here has 200 hz.
    # So we go with 1 / 200 = 0.005.
    # f = 200 if file[len('RawData/') + 3:len('RawData/') + 5] == 'NB' else 100
    dt = 1 / f

    # doing fft stuff here
    fhat = np.fft.fft(fftData.z, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(1, np.floor(n / 2), dtype='int')

    # Plotting new FFT data

    if show_plot:
        fig, axs = plt.subplots(2, 1)
        plt.sca(axs[0])
        plt.plot(sample_data.time, sample_data.z, color='c', LineWidth=1, label='Original Data')
        plt.legend()

        plt.sca(axs[1])
        plt.scatter(freq[L], PSD[L], color='g', LineWidth=1, label='Power Spectrum of Dataset')
        plt.legend()

        plt.show()

    # Use PSD to filter out noise
    # Find all frequencies with great power
    indices = PSD > 10  # boolean array
    print('Länge indices: ', len(indices))
    print('Länge fhat: ', len(fhat))

    # zero out small fourier coeffs. in y
    fhat = indices * fhat

    # inverse FFT for filtered time signal
    ffilt = np.fft.ifft(fhat)

    # Plotting filtered data

    if show_plot:
        plt.plot(fftData.time, ffilt, color='g', LineWidth=1, label='Inverse FFT')
        plt.legend()
        plt.show()

    # Putting high pass filter on sample data
    sos = signal.butter(10, 15, btype='highpass', fs=f, output='sos')  # 15 Hz High Pass Filter, filter out 10 Hz frequencies
    filtered = signal.sosfilt(sos, sample_data.y)

    if show_plot:
        plt.plot(sample_data.time, filtered, color='g', LineWidth=1, label='High pass filter')
        plt.legend()
        plt.show()

    # Print Standard Derivation
    derivation = filtered.std()
    min = filtered.min()
    max = filtered.max()
    return derivation, max, min


# predict_road_surface_derivation('RawData/SC_LP_WaldwegB4ZurIlmenau.csv', 100, True, True)

directory = 'Data/RawData/'
derivations = {
    'SP': [],
    'AS': [],
    'ST': [],
    'BT': [],
    'PF': [],
    'RW': [],
    'SC': [],
    'WW': [],
    'KO': []
}

color_lookup = {
    'SP': 'k',
    'AS': 'r',
    'ST': 'r',
    'BT': 'b',
    'PF': 'b',
    'RW': 'b',
    'SC': 'y',
    'WW': 'g',
    'KO': 'c',
}
marker_lookup = {
    'SP': '.',
    'AS': 'v',
    'ST': 'v',
    'BT': 'p',
    'PF': 'p',
    'RW': 'p',
    'SC': 'd',
    'WW': 'P',
    'KO': '*',
}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for file in os.listdir(directory):
    filepath = directory + file
    print('File: ', file)
    surface_type = file[0:2]
    freq = 200 if file[3:5] == 'NB' else 100
    derivation, maximum, minimum = predict_road_surface_derivation(filepath, f=freq, cut_sample=True)
    # if file[3:5] == 'LP':
    #    derivation = derivation - 0.5
    #    maximum = maximum - 2

    pt = ax.scatter(minimum, maximum, derivation, marker=marker_lookup[surface_type], c=color_lookup[surface_type])
    derivations[surface_type].append(derivation)
    print(file, 'STD: ', derivation, ' MAX: ', maximum, ' MIN: ', minimum)

ax.set_xlabel('Abweichung')
ax.set_ylabel('Maximum')
ax.set_zlabel('Minimum')
# plt.legend(['SP', 'AS', 'ST', 'BT', 'PF', 'RW', 'SC', 'WW', 'KO'])
plt.show()
