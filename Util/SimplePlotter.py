import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PreProcessing.PreProcess import PreProcess

BASE_PATH = 'Data/EvaluationData/'
'''for file in os.listdir(BASE_PATH):
    data = pd.read_csv(BASE_PATH + file)
    data.columns = ['time', 'x', 'y', 'z', 'abs']
    fig, axs = plt.subplots(3, sharex=True, sharey=False)
    axs[0].plot(data['time'], data['x'], color='r', label='X-Acceleration')
    plt.legend()
    axs[1].plot(data['time'], data['y'], color='g', label='y-Acceleration')
    plt.legend()
    axs[2].plot(data['time'], data['z'], color='b', label='Z-Acceleration')
    plt.legend()
    plt.show()'''

data = pd.read_csv('Data/RawData/KO_NM_Salzbrücker Straße.csv')
data.columns = ['time', 'x', 'y', 'z', 'abs']
fig, axs = plt.subplots(3, sharex=True, sharey=False)
axs[0].plot(data['time'], data['x'], label='X-Acceleration')
axs[1].plot(data['time'], data['y'], label='y-Acceleration')
axs[2].plot(data['time'], data['z'], label='Z-Acceleration')
plt.show()
