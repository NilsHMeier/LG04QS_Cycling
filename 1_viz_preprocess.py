import math
from scipy import special
import pandas as pd
import matplotlib.pyplot as plt
from PreProcessing.PreProcess import PreProcess
from Util.Data_Vizualizer import Vizualizer

# Use cobblestone sample to show the need of cropping the data
raw_data = pd.read_csv('Data/RawData/KO_NM_Salzbrücker Straße.csv')
raw_data.columns = ['time', 'x', 'y', 'z', 'abs']
Vizualizer.plot_line(raw_data, 'time', ['x', 'y', 'z'], 3, 1)

# Use asphalt sample to show the effect of processing a filter and outlier detection
data = pd.read_csv('Data/RawData/AS_LP_Brockwinkler Straße.csv')
data.columns = ['time', 'x', 'y', 'z', 'abs']
# Call process_offset to show rawData and set offsets
data = PreProcess.process_offset(data)
filter_data = PreProcess.process_filter(data, 100)
for col in ['x', 'y', 'z']:
    filter_data[f'raw_{col}'] = data[col]
# Show graph with raw y data and y data after filter was processed
Vizualizer.plot_line(filter_data, 'time', ['y', 'raw_y'], 2, 1)

# Show outlier detection with various criterions
# CRITERIONS = [0.01, 0.005, 0.001, 0.0005, 0.0001]
CRITERIONS = [0.05, 0.005, 0.0005]
COLS = ['x']
for col in COLS:
    print('Processing col', col)
    counter = 1
    mean = data[col].mean()
    std = data[col].std()
    n = len(data['time'])
    deviation = abs(data[col] - mean) / std
    low = -deviation / math.sqrt(2)
    high = deviation / math.sqrt(2)
    for criterion in CRITERIONS:
        prob = []
        mask = []
        outlier_count = 0
        for i in data.index:
            prob.append(1.0 - 0.5 * (special.erf(high[i]) - special.erf(low[i])))
            mask.append(prob[i] < criterion)
            outlier_count += 1 if prob[i] < criterion else 0
        plt.subplot(len(CRITERIONS), 1, counter)
        for i in range(0, len(mask)):
            plt.plot(data.loc[i, 'time'], data.loc[i, col], 'r+' if mask[i] else 'b+')
        plt.ylabel(f'Criterion={criterion}, Outliers={outlier_count}')
        plt.xlim(0, data['time'].max())
        print(f'Outlier count for {criterion} on col {col} is {outlier_count}')
        counter += 1
plt.show()
# Create plot with distribution
plt.subplot(1, 1, 1)
plt.scatter(data['x'], prob)
plt.xlabel('x-axis acceleration')
plt.ylabel('Probability')
plt.show()
