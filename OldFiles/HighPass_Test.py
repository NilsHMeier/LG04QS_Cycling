import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

for file in ["KO_NM_Salzbrücker Straße.csv", "AS_LP_RadwegUni.csv", "RW_LP_Kirchsteig1.csv",
             "SC_NM_Hinter Gut Brockwinkel.csv"]:
    rawData = pd.read_csv(str("Data/RawData/") + str(file))
    rawData.columns = ['time', 'x', 'y', 'z', 'abs']
    # print(rawData.head())

    '''fig, axs = plt.subplots(3, sharex=True, sharey=False)
    axs[0].plot(rawData['time'], rawData['x'], label='RawData X-Acc')
    plt.legend()
    axs[1].plot(rawData['time'], rawData['y'], label='RawData Y-Acc')
    plt.legend()
    axs[2].plot(rawData['time'], rawData['z'], label='RawData Z-Acc')
    plt.legend()
    plt.show()'''

    # start = int(input("Start (s): "))
    # end = int(input("End (s): "))
    # rawData = rawData[(rawData['time'] >= start) & (rawData['time'] <= end)]
    # rawData['time'] = rawData['time'] - start

    f = 100
    # Putting high pass filter on sample data
    for cutter_freq in [2, 5, 10, 15, 20]:
        sos = signal.butter(10, cutter_freq, btype='highpass', fs=f,
                            output='sos')  # 15 Hz High Pass Filter, filter out 10 Hz frequencies
        filteredX = signal.sosfilt(sos, rawData['x'])
        filteredY = signal.sosfilt(sos, rawData['y'])
        filteredZ = signal.sosfilt(sos, rawData['z'])
        print('File:', file, ' Cutter frequency:', cutter_freq, 'Hz')
        print('X-Std: ', filteredX.std(), 'Y-Std: ', filteredY.std(), 'Z-Std: ', filteredZ.std())
        '''print('X-Max: ', filteredX.max(), ' X-Std: ', filteredX.std())
        print('Y-Max: ', filteredY.max(), ' Y-Std: ', filteredY.std())
        print('Z-Max: ', filteredZ.max(), ' Z-Std: ', filteredZ.std())'''
    '''plt.subplot(321)
    plt.plot(rawData['time'], rawData['x'], label='Raw X')
    plt.subplot(322)
    plt.plot(rawData['time'], filteredX, label='Filtered X')
    plt.subplot(323)
    plt.plot(rawData['time'], rawData['y'], label='Raw Y')
    plt.subplot(324)
    plt.plot(rawData['time'], filteredY, label='Filtered Y')
    plt.subplot(325)
    plt.plot(rawData['time'], rawData['z'], label='Raw Z')
    plt.subplot(326)
    plt.plot(rawData['time'], filteredZ, label='Filtered Z')
    plt.show()'''
