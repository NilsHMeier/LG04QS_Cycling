import numpy as np
from MachineLearning.LearningAlgorithms import MachineLearning
from MachineLearning.PrepareData import PrepareDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Decide what should be vizualised
DO_SVM = False
DO_KNN = True

# Set the source path, load the data and prepare it for training the models
SOURCE_PATH = 'Data/AggregatedData/'
data_table = PrepareDataset.fill_datatable(source_path=SOURCE_PATH, labelcol_name='label', names=[])
selected_features = [col + feature for col in ['x', 'y', 'z'] for feature in ['_mean', '_max', '_min', '_std']]
label_col = 'label'
X_train, X_test, Y_train, Y_test = PrepareDataset.split_dataset(data_table, selected_features, label_col, 0.3)
train_data = {'x': X_train, 'y': Y_train}
test_data = {'x': X_test, 'y': Y_test}

# Load a second dataset with other data to evaluate the trained models
# training_data = PrepareDataset.fill_datatable(source_path=SOURCE_PATH, labelcol_name='label', names=['NM'])
training_data = PrepareDataset.prepare_evaluation_dataset('Data/EvaluationData/', 'label', ['LP', 'NP', 'NM'], 15, 20)

if DO_SVM:
    # Set up the different hyperparameters that should be used
    # Cs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    Cs = np.arange(1, 100, 1)
    # Gammas = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    Gammas = np.arange(0.01, 1, 0.01)
    Scores = []
    # Train a model with each possible combination of C and Gamma and save the best score
    for c in Cs:
        for gamma in Gammas:
            best_score = 0
            for i in range(0, 100):
                score, svm = MachineLearning.support_vector_machine_with_kernel(train_data, test_data, c, gamma)
                if score > best_score:
                    best_score = score
            Scores.append(best_score)
            print(f'C={c} & Gamma={gamma} -> Score={best_score}')

    # Create data used in plot
    X, Y = np.meshgrid(Cs, Gammas)
    Z = np.asarray(Scores)
    Z = Z.reshape(len(Gammas), len(Cs))

    # Create 3d surface plot showing the score with the different parameters
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='brg_r')
    # surf = ax.scatter(X, Y, Z, cmap='brg_r')
    ax.set_xlabel('C')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Score')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if DO_KNN:
    # Set up a range of Ks for training the model
    n = np.arange(1, 50, 1)
    training_scores = []
    evaluation_scores = []
    # Train a model for each possible k and save the best score
    for i in n:
        print(f'Calculating best score for {i} neighbours')
        acc, knn = MachineLearning.k_nearest_neighbours(train_data, test_data, i)
        training_scores.append(acc)
        evaluation_scores.append(knn.score(training_data.drop(columns=['label']), training_data['label']))

    # Plot the training and test scores in one graph
    plt.plot(n, training_scores, color='b', label='Validation Score')
    plt.plot(n, evaluation_scores, color='r', label='Test Score')
    plt.xlabel('k Neighbours')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


