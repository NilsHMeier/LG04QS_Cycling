import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PreProcessing.PreProcess import PreProcess

data_suspension = pd.read_csv('Data/RawData/AS_NM_Landwehr Bardowick.csv')
data_suspension.columns = ['time', 'x', 'y', 'z', 'abs']
# data_suspension = PreProcess.process_suspension_coefficient(data_suspension, 0.5, ['x', 'y', 'z'])
data_suspension = PreProcess.process_filter(data_suspension, 100, ['x', 'y', 'z'])
data_suspension = PreProcess.process_outlier_detection(data_suspension, ['x', 'y', 'z'])
data_reference = pd.read_csv('Data/RawData/AS_LP_Landwehr Bardowick.csv')
data_reference.columns = ['time', 'x', 'y', 'z', 'abs']
data_reference = PreProcess.process_filter(data_reference, 100, ['x', 'y', 'z'])
data_reference = PreProcess.process_outlier_detection(data_reference, ['x', 'y', 'z'])

fig, axs = plt.subplots(3, 2, sharex=True, sharey=False)
axs[0][0].plot(data_reference['time'], data_reference['x'], color='r')
axs[1][0].plot(data_reference['time'], data_reference['y'], color='r')
axs[2][0].plot(data_reference['time'], data_reference['z'], color='r')
axs[0][1].plot(data_suspension['time'], data_suspension['x'], color='g')
axs[1][1].plot(data_suspension['time'], data_suspension['y'], color='g')
axs[2][1].plot(data_suspension['time'], data_suspension['z'], color='g')
plt.show()
