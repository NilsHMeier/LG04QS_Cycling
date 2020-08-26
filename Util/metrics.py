import os as os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm


def seconds_to_string(s: float) -> str:
    s = int(s)
    hours = str(int(s / 3600))
    minutes = "0" + str(int(s / 60 % 60)) if int(s / 60 % 60) < 10 else str(int(s / 60 % 60))
    seconds = "0" + str(int(s % 60)) if int(s % 60) < 10 else str(int(s % 60))
    return hours + ":" + minutes + ":" + seconds


# Set the source path to create the metrics for
SOURCE_PATH = 'Data/RawData/'
# Create counter for general and type specific metrics
counter = 0
num_files = 0
max_duration = 0
type_count = {}
type_duration = {}

# Process all files in the given directory
for file in os.listdir(SOURCE_PATH):
    num_files += 1
    # Build the filepath
    filepath = SOURCE_PATH + file
    # Read in the data. In case of ProcessedData it has to be read in differently
    if SOURCE_PATH == 'Data/RawData/' or SOURCE_PATH == 'Data/EvaluationData/':
        rawData = pd.read_csv(filepath, sep=",")
        rawData.columns = ['time', 'x', 'y', 'z', 'abs']
    elif SOURCE_PATH == 'Data/ProcessedData/':
        rawData = pd.read_csv(filepath, sep=",", index_col=0)
    # Get the duration of the sample by selecting the max of time col
    last_row = rawData['time'].max()
    # Add duration to general & specific time counter and increment the type counter.
    counter += last_row
    max_duration = max(max_duration, last_row)
    if file[0:2] not in type_count:
        type_count[file[0:2]] = 1
    else:
        type_count[file[0:2]] += 1
    if file[0:2] not in type_duration:
        type_duration[file[0:2]] = last_row
    else:
        type_duration[file[0:2]] += last_row

# Print the metrics of all files
print("Total files:", num_files)
print("Total time:", seconds_to_string(counter))
print("Longest sample:", max_duration, "Average duration:", counter / num_files)
print("Different types:", type_count)
type_numeric = list(type_duration[k] for k in type_duration)
for k in type_duration:
    type_duration[k] = seconds_to_string(type_duration[k])
print("Type duration:", type_duration)
# Create bar chart with the number of samples for each type
plt.subplot(121)
for key in type_count.keys():
    plt.bar(key, type_count[key])
plt.title('Verteilung der Anzahl an Messungen')
# Create pie chart to show the temporal distribution of the different types
plt.subplot(122)
plt.pie(type_numeric, labels=type_duration.keys(), autopct='%1.1f%%')
plt.title('Zeitliche Verteilung der Untergründe')
plt.show()
