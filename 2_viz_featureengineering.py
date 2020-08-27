from MachineLearning.PrepareData import PrepareDataset
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Set colors for underground types
color_lookup = {'AS': 'r', 'RW': 'b', 'SC': 'y', 'WW': 'g', 'KO': 'c'}
# Set source path and call method to prepare the datatable
SOURCE_PATH = 'Data/AggregatedData/'
engineer = PrepareDataset(SOURCE_PATH, True, True, 'label', ['x', 'y', 'z'])
data_table = engineer.fill_datatable(['NM'])
print(data_table.head())

# Create 3D-scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
labels = {}
# Set the features to be shown on the three axis
axis = {'x': 'x_std', 'y': 'y_std', 'z': 'z_std'}
# Scattering all data points in the prepared datatable using the defined colors
for i in data_table.index:
    label = data_table.loc[i, 'label']
    pt = ax.scatter(data_table.loc[i, axis['x']], data_table.loc[i, axis['y']], data_table.loc[i, axis['z']],
                    color=color_lookup[label])
    if label not in labels:
        labels[label] = pt
# Set the axis label to the chosen feature
ax.set_xlabel(axis['x'])
ax.set_ylabel(axis['y'])
ax.set_zlabel(axis['z'])
plt.legend(labels.values(), labels.keys())
plt.show()
