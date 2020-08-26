import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PreProcessing.FeatureEngineering import FeatureEngineering
from MachineLearning.PrepareData import PrepareDataset
from MachineLearning.LearningAlgorithms import MachineLearning

PERIODS = np.arange(1, 21, 1)
BASE_PATH = 'Data/ProcessedData/'
SCORES = []
LENGTHS = []
selected_features = [col + feature for col in ['x', 'y', 'z'] for feature in ['_mean', '_max', '_min', '_std']]

for period in PERIODS:
    print(f'Running learning for period {period}')
    learning_data = PrepareDataset.create_datatable('label')
    # print(learning_data)
    for file in os.listdir(BASE_PATH):
        data = pd.read_csv(BASE_PATH + file, index_col=0)
        aggregated_data = FeatureEngineering.aggregate_data(data, ['x', 'y', 'z'], period)
        aggregated_data['label'] = file[0:2]
        learning_data = learning_data.append(aggregated_data, ignore_index=True)
    learning_data = learning_data.drop(columns='time')
    # print(learning_data.head())

    X_train, X_test, Y_train, Y_test = PrepareDataset.split_dataset(learning_data, selected_features, 'label', 0.3)
    train_data = {'x': X_train, 'y': Y_train}
    test_data = {'x': X_test, 'y': Y_test}
    score, svm = MachineLearning.support_vector_machine_with_kernel(train_data, test_data, 100, 1)
    SCORES.append(score)
    LENGTHS.append(len(learning_data))

plt.subplot(211)
plt.plot(PERIODS, LENGTHS)
plt.title('Number of instances in learning data')
plt.xlim(1, 20)
plt.subplot(212)
plt.plot(PERIODS, SCORES)
plt.title('SVM score for different periods')
plt.xlim(1, 20)
plt.show()
