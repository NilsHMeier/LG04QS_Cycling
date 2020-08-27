import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PreProcessing.FeatureEngineering import FeatureEngineering
from MachineLearning.PrepareData import PrepareDataset
from MachineLearning.LearningAlgorithms import MachineLearning

PERIODS = np.arange(1, 21, 1)
BASE_PATH = 'Data/ProcessedData/'
SVM_SCORES = []
KNN_SCORES = []
LENGTHS = []
selected_features = [col + feature for col in ['x', 'y', 'z'] for feature in ['_mean', '_max', '_min', '_std']]
data_preparing = PrepareDataset('', True, False, 'label', ['x', 'y', 'z'])

for period in PERIODS:
    print(f'Running learning for period {period}')
    learning_data = data_preparing.create_datatable()
    engineer = FeatureEngineering('', '', period)
    # print(learning_data)
    for file in os.listdir(BASE_PATH):
        data = pd.read_csv(BASE_PATH + file, index_col=0)
        aggregated_data = engineer.aggregate_data(data, ['x', 'y', 'z'], file[3:5])
        aggregated_data = aggregated_data.drop(columns=[col for col in aggregated_data.columns
                                                        if col not in selected_features])
        aggregated_data['label'] = file[0:2]
        learning_data = learning_data.append(aggregated_data, ignore_index=True)
    # print(learning_data.head())

    X_train, X_test, Y_train, Y_test = data_preparing.split_dataset_selected(learning_data, selected_features, 0.3)
    train_data = {'x': X_train, 'y': Y_train}
    test_data = {'x': X_test, 'y': Y_test}
    score, svm = MachineLearning.support_vector_machine_with_kernel(train_data, test_data, 100, 1)
    SVM_SCORES.append(score)
    score, knn = MachineLearning.k_nearest_neighbours(train_data, test_data, 1)
    KNN_SCORES.append(score)
    LENGTHS.append(len(learning_data))

plt.subplot(211)
plt.plot(PERIODS, LENGTHS)
plt.title('Number of instances in learning data')
plt.xlim(1, 20)
plt.subplot(212)
plt.plot(PERIODS, SVM_SCORES, label='SVM')
plt.plot(PERIODS, KNN_SCORES, label='KNN')
plt.legend()
plt.title('Scores for different periods')
plt.xlim(1, 20)
plt.show()
