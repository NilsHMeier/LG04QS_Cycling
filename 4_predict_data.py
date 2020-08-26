import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from MachineLearning.PrepareData import PrepareDataset
from sklearn import metrics
import pickle

# Set up paths and decide which model type should be used
SOURCE_PATH = 'Data/EvaluationData/'
MODEL_PATH = 'Models/'
TRAIN_NAME = 'train_svm.sav'
EVAL_NAME = 'eval_svm.sav'

# Load models from pickle files
best_train_model = pickle.load(open(MODEL_PATH + TRAIN_NAME, 'rb'))
best_eval_model = pickle.load(open(MODEL_PATH + EVAL_NAME, 'rb'))

# Create datatable with data to predict
predict_datatable = PrepareDataset.prepare_evaluation_dataset('Data/EvaluationData/', 'label', ['LP', 'NM', 'NP'], 15, 15)
# predict_datatable = PrepareDataset.fill_datatable('Data/AggregatedData/', 'label', ['LP'])

# Plot normalized confusion matrix
metrics.plot_confusion_matrix(best_eval_model, predict_datatable.drop(columns=['label']), predict_datatable['label'], normalize='true')
print('Score:', best_eval_model.score(predict_datatable.drop(columns=['label']), predict_datatable['label']))
plt.show()
best_train_score = 0
best_eval_score = 0
# Let the models predict the loaded data and save the best achieved score
for i in range(100):
    train_score = best_train_model.score(predict_datatable.drop(columns=['label']), predict_datatable['label'])
    if train_score > best_train_score:
        best_train_score = train_score
    eval_score = best_eval_model.score(predict_datatable.drop(columns=['label']), predict_datatable['label'])
    if eval_score > best_eval_score:
        best_eval_score = eval_score
print(f'Best train score is {best_train_score}')
print(f'Best eval score is {best_eval_score}')
