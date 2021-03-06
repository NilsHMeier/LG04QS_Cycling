from MachineLearning.PrepareData import PrepareDataset
from MachineLearning.LearningAlgorithms import MachineLearning
from MachineLearning.FeatureSelection import FeatureSelection
import pandas as pd # noqa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from Util.Data_Vizualizer import Vizualizer
from mpl_toolkits.mplot3d import Axes3D  # noqa
import pickle

# Set path & names where to save the models and choose which type of learning to do
SOURCE_PATH = 'Data/AggregatedData/'
MODEL_PATH = 'Models/'
DO_KNN = False
KNN_FILE_NAME = 'knn.sav'
DO_DT = False
DT_FILE_NAME = 'dt.sav'
DO_SVM = False
SVM_FILE_NAME = 'svm.sav'
# Decide which features should be placed in training datatable
STAT_FEATURES = True
FREQ_FEATURES = True

# Create datatable for training the model
data_preparing = PrepareDataset(SOURCE_PATH, STAT_FEATURES, FREQ_FEATURES, 'label', ['x', 'y', 'z'])
train_datatable = data_preparing.fill_datatable(['LP', 'NB', 'NM'])
# Select features to use or leave the list empty to use all features
selected_features = []
# Divide the datatable into train and validation data and create dictionaries of them
if len(selected_features) == 0:
    X_train, X_test, Y_train, Y_test = data_preparing.split_dataset_default(train_datatable, 0.3)
else:
    X_train, X_test, Y_train, Y_test = data_preparing.split_dataset_selected(train_datatable, selected_features, 0.3)
print('X_train columns:', X_train.columns)
train_data = {'x': X_train, 'y': Y_train}
test_data = {'x': X_test, 'y': Y_test}

# Create datatable with test data to evaluate the model
evaluation_datatable = data_preparing.prepare_evaluation_dataset('Data/EvaluationData/', ['NM', 'LP'], 15, 15)
# evaluation_datatable = data_preparing.fill_datatable(['NM'])
# Make sure evaluation data has the same form as training data
evaluation_datatable = evaluation_datatable.drop(columns=[col for col in evaluation_datatable.columns
                                                          if col != 'label' and col not in X_train.columns])

# Run feature selection to determine the best features
max_features = 27
features, scores = FeatureSelection.forward_selection(max_features, X_train, Y_train, X_test, Y_test)
print(f'Best {max_features} features are {features} with scores {scores}')
plt.subplot(111)
plt.plot(list(range(1, max_features+1)), scores)
plt.xticks(list(range(1, max_features+1)), features, rotation='vertical')
plt.xlabel('Number of features')
plt.ylabel('Score')
plt.xlim(1, max_features)
plt.show()

if DO_KNN:
    print('- - - Running learning for KNN - - -')
    # Do Training Optimization
    print('Optimizing training score for KNN')
    best_score = 0
    best_knn = None
    best_k = None
    # Change the number of neighbours and find the best model
    for k in np.arange(1, 50, 1):
        score, knn = MachineLearning.k_nearest_neighbours(train_data, test_data, k)
        if score > best_score:
            best_score = score
            best_k = k
            best_knn = knn
    print(f'Best training score for KNN is {best_score} with k={best_k}')
    print(f'KNN = {best_knn}')
    # Plot confusion matrix with validation data
    metrics.plot_confusion_matrix(best_knn, test_data['x'], test_data['y'], normalize='true')
    plt.title('Training confusion matrix')
    plt.show()
    # Save the model to the set path
    pickle.dump(best_knn, open(MODEL_PATH + 'train_' + KNN_FILE_NAME, 'wb'))
    # Do Evaluation Optimization
    print('Optimizing evaluation score for KNN')
    best_eval = 0
    train_score = None
    best_knn = None
    best_k = None
    eval_scores = []
    train_scores = []
    for k in np.arange(1, 50, 1):
        score, knn = MachineLearning.k_nearest_neighbours(train_data, test_data, k)
        # Get the score of the model using the 'unseen' test data
        eval_score = MachineLearning.calculate_evaluation_score(evaluation_datatable, 'label', knn)
        if eval_score > best_eval:
            best_eval = eval_score
            best_k = k
            best_knn = knn
            train_score = score
        eval_scores.append(eval_score)
        train_scores.append(score)
    print(f'Best evaluation score for KNN is {best_eval} with k={best_k}')
    print(f'Training score for best evaluation score is {train_score}')
    print(f'KNN = {best_knn}')
    # Plot confusion matrix with test data
    metrics.plot_confusion_matrix(best_knn, evaluation_datatable.drop(columns=['label']), evaluation_datatable['label'], normalize='true')
    plt.title('Evaluation confusion matrix')
    plt.show()
    # Create a graph that shows the validation and test scores dependant on the number of neighbours
    plt.plot(np.arange(1, 50, 1), train_scores, color='b', label='Training Score')
    plt.plot(np.arange(1, 50, 1), eval_scores, color='r', label='Evaluation Score')
    plt.xlabel('Neighbours k')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    pickle.dump(best_knn, open(MODEL_PATH + 'eval_' + KNN_FILE_NAME, 'wb'))

if DO_DT:
    print('- - - Running learning for DT - - -')
    # Do Training Optimization
    print('Optimizing training score for DT')
    best_score = 0
    best_dt = None
    best_k = None
    # Change the minumum leaf size number and calculate scores
    for k in np.arange(1, 50, 1):
        score, dt = MachineLearning.decision_tree(train_data, test_data, k)
        if score > best_score:
            best_score = score
            best_k = k
            best_dt = dt
    print(f'Best training score for DT is {best_score} with k={best_k}')
    print(f'DT = {best_dt}')
    # Plot confusion matrix with test
    metrics.plot_confusion_matrix(best_dt, test_data['x'], test_data['y'], normalize='true')
    plt.title('Training confusion matrix')
    plt.show()
    pickle.dump(best_dt, open(MODEL_PATH + 'train_' + DT_FILE_NAME, 'wb'))
    # Performing Gridsearch
    print('Doing gridsearch for DT')
    score, dt = MachineLearning.dt_gridsearch(train_data, test_data)
    print('Score:', score)
    print(f'Best estimator = {dt}')
    # Evaluation Optimization
    print('Optimizing evaluation score for DT')
    best_eval = 0
    train_score = None
    best_dt = None
    best_k = None
    eval_scores = []
    train_scores = []
    for k in np.arange(1, 50, 1):
        score, dt = MachineLearning.decision_tree(train_data, test_data, k)
        eval_score = dt.score(evaluation_datatable.drop(columns=['label']), evaluation_datatable['label'])
        if eval_score > best_eval:
            best_eval = eval_score
            best_k = k
            best_dt = dt
            train_score = score
        eval_scores.append(eval_score)
        train_scores.append(score)
    print(f'Best evaluation score for DT is {best_eval} with min_leaf_size={best_k}')
    print(f'Training score for best evaluation score is {train_score}')
    print(f'DT = {best_dt}')
    metrics.plot_confusion_matrix(best_dt, evaluation_datatable.drop(columns=['label']), evaluation_datatable['label'], normalize='true')
    plt.title('Evaluation confusion matrix')
    plt.show()
    plt.plot(np.arange(1, 50, 1), train_scores, color='b', label='Training Score')
    plt.plot(np.arange(1, 50, 1), eval_scores, color='r', label='Evaluation Score')
    plt.legend()
    plt.show()
    pickle.dump(best_dt, open(MODEL_PATH + 'eval_' + DT_FILE_NAME, 'wb'))

if DO_SVM:
    print('- - - Running learning for SVM - - -')
    # Cs = np.arange(1, 100, 2)
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    # Gammas = np.arange(0.01, 1, 0.02)
    Gammas = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # Training Optimization:
    print('Optimizing training score for SVM')
    best_score = 0
    best_svm = None
    best_c = None
    best_gamma = None
    for c in Cs:
        for gamma in Gammas:
            score, svm = MachineLearning.support_vector_machine_with_kernel(train_data, test_data, c, gamma)
            if score > best_score:
                best_score = score
                best_svm = svm
                best_c = c
                best_gamma = gamma
    print(f'Best training score for SVM is {best_score} with c={best_c} and gamma={best_gamma}')
    print(f'SVM = {best_svm}')
    metrics.plot_confusion_matrix(best_svm, test_data['x'], test_data['y'], normalize='true')
    plt.title('Training confusion matrix')
    plt.show()
    pickle.dump(best_svm, open(MODEL_PATH + 'train_' + SVM_FILE_NAME, 'wb'))
    # Evaluation Optimization
    print('Optimizing evaluation score for SVM')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    training_scores = []
    evaluation_scores = []
    best_svm = None
    best_eval = 0
    training_score = 0
    for c in Cs:
        for gamma in Gammas:
            score, svm = MachineLearning.support_vector_machine_with_kernel(train=train_data, test=test_data, c=c, gamma=gamma)
            eval_score = MachineLearning.calculate_evaluation_score(evaluation_datatable, 'label', svm)
            if eval_score > best_eval:
                best_svm = svm
                best_eval = eval_score
                training_score = score
            training_scores.append(score)
            evaluation_scores.append(eval_score)
            tr = ax.scatter(c, gamma, score, color='b')
            ev = ax.scatter(c, gamma, eval_score, color='r')
    plt.legend([tr, ev], ['Training Score', 'Evaluation Score'])
    ax.set_xlabel('C')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Score')
    plt.show()
    print('Best SVM:', best_svm)
    print('Best Evaluation Score:', best_eval)
    print('Training Score:', training_score)
    metrics.plot_confusion_matrix(best_svm, evaluation_datatable.drop(columns=['label']), evaluation_datatable['label'], normalize='true')
    plt.title('Evaluation confusion matrix')
    plt.show()
    pickle.dump(best_svm, open(MODEL_PATH + 'eval_' + SVM_FILE_NAME, 'wb'))
