import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from threading import Thread
from PreProcessing.FeatureEngineering import FeatureEngineering
from MachineLearning.PrepareData import PrepareDataset
from MachineLearning.LearningAlgorithms import MachineLearning

PERIODS = np.arange(1, 21, 1)
BASE_PATH = '../Data/ProcessedData/'
KNN_SCORES = [0 for x in PERIODS]
DT_SCORES = [0 for x in PERIODS]
SVM_SCORES = [0 for x in PERIODS]
LENGTHS = [0 for x in PERIODS]
selected_features = [col + feature for col in ['x', 'y', 'z'] for feature in ['_mean', '_max', '_min', '_std']]


def calculate_score(period):
    print(f'Running learning for period {period}')
    learning_data = PrepareDataset.create_datatable('label')
    # print(learning_data)
    for file in os.listdir(BASE_PATH):
        data = pd.read_csv(BASE_PATH + file, index_col=0)
        aggregated_data = FeatureEngineering.aggregate_data(data, ['x', 'y', 'z'], period)
        aggregated_data['label'] = file[0:2]
        learning_data = learning_data.append(aggregated_data, ignore_index=True)
    learning_data = learning_data.drop(columns='time')

    X_train, X_test, Y_train, Y_test = PrepareDataset.split_dataset(learning_data, selected_features, 'label', 0.3)
    train_data = {'x': X_train, 'y': Y_train}
    test_data = {'x': X_test, 'y': Y_test}
    knn_score, knn = MachineLearning.k_nearest_neighbours(train_data, test_data, 15)
    KNN_SCORES[period - 1] = knn_score
    # dt_score, dt = MachineLearning.decision_tree(train_data, test_data, 1)
    # DT_SCORES[period - 1] = dt_score
    svm_score, svm = MachineLearning.support_vector_machine_with_kernel(train_data, test_data, 100, 1)
    SVM_SCORES[period - 1] = svm_score
    LENGTHS[period - 1] = len(learning_data)
    print(f'Finished learning for period {period}')


threads = []
for p in PERIODS:
    process = Thread(target=calculate_score, args=[p])
    process.start()
    threads.append(process)
for process in threads:
    process.join()

plt.subplot(211)
plt.plot(PERIODS, LENGTHS)
plt.title('Number of instances in learning data')
plt.xlim(1, 20)
plt.subplot(212)
plt.plot(PERIODS, KNN_SCORES, label='KNN')
# plt.plot(PERIODS, DT_SCORES, label='DT')
plt.plot(PERIODS, SVM_SCORES, label='SVM')
plt.legend()
plt.title('Scores for different periods')
plt.xlim(1, 20)
plt.show()
