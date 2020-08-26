from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import sklearn
import pandas as pd


class MachineLearning:
    @staticmethod
    def k_nearest_neighbours(train: dict, test: dict, k: int) -> (float, KNeighborsClassifier):
        """
        Trains a KNN model with the provided data and number of neighbours.

        :param train: Dictionary containing the feature data and the labels to train the model with.
        :param test: Dictionary containing the feature data and the labels to validate the model with.
        :param k: Number of neighbours used to train the model.
        :return: Returns the best achieved score und the KNeighboursClassifier as tuple.
        """
        best_knn = None
        best_score = 0
        for i in range(100):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train['x'], train['y'])
            accuracy = knn.score(test['x'], test['y'])
            if accuracy > best_score:
                best_knn = knn
                best_score = accuracy
        return best_score, best_knn

    @staticmethod
    def decision_tree(train: dict, test: dict, minimum_leaf_size: int) -> (float, DecisionTreeClassifier):
        """
        Trains a DT model with the provided data and minumum leaf size.

        :param train: Dictionary containing the feature data and the labels to train the model with.
        :param test: Dictionary containing the feature data and the labels to validate the model with.
        :param minimum_leaf_size: Number of minumum datapoints per leaf.
        :return: Returns the best achieved score and the DecisionTreeClassifier as tuple.
        """
        best_dt = None
        best_score = 0
        for i in range(100):
            dt = DecisionTreeClassifier(min_samples_leaf=minimum_leaf_size)
            dt.fit(train['x'], train['y'])
            accuracy = dt.score(test['x'], test['y'])
            if accuracy > best_score:
                best_dt = dt
                best_score = accuracy
        return best_score, best_dt

    @staticmethod
    def dt_gridsearch(train: dict, test: dict) -> (float, DecisionTreeClassifier):
        """
        Performs a gridsearch over a DecisionTree with the hyperparameters set within the method.

        :param train: Dictionary containing the feature data and the labels to train the model with.
        :param test: Dictionary containing the feature data and the labels to validate the model with.
        :return: Returns the best achieved score and the DecisionTreeClassifier as tuple.
        """
        params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
        grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params, verbose=3, refit=True)
        grid_search_cv.fit(train['x'], train['y'])
        accuracy = grid_search_cv.score(test['x'], test['y'])
        return accuracy, grid_search_cv.best_estimator_

    @staticmethod
    def support_vector_machine_with_kernel(train: dict, test: dict, c: float, gamma: float) -> (float, SVC):
        """
        Trains a SVM with RBF kernel with the provided data and given hyperparameters C and Gamma.

        :param train: Dictionary containing the feature data and the labels to train the model with.
        :param test: Dictionary containing the feature data and the labels to validate the model with.
        :param c: Hyperparameter C to set the penalty for misclassified data points in training. Has impact on the
            margin of decicion boundary of the model.
        :param gamma: Hyperparamter Gamma to set the similarity radius of datapoints.
        :return: Returns the best achieved score and the SupportVectorClassifier as tuple.
        """
        best_svm = None
        best_score = 0
        for i in range(100):
            svm = SVC(C=c, gamma=gamma)
            # Best: C=100, gamma=1
            svm.fit(train['x'], train['y'])
            accuracy = svm.score(test['x'], test['y'])
            if accuracy > best_score:
                best_svm = svm
                best_score = accuracy
        return best_score, best_svm

    @staticmethod
    def svm_gridsearch(train: dict, test: dict) -> (float, SVC):
        """
        Performs a gridsearch over a SupportVectorClassifier with the hyperparameters set within the method.

        :param train: Dictionary containing the feature data and the labels to train the model with.
        :param test: Dictionary containing the feature data and the labels to validate the model with.
        :return: Returns the best achieved score and the SupportVectorClassifier as tuple.
        """
        parameters = {'kernel': ['rbf', 'poly'], 'gamma': [1e-3, 1e-4],
                      'C': [1, 10, 100]}
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
        grid.fit(train['x'], train['y'])
        print(grid.best_params_)
        print(grid.best_estimator_)
        accuracy = grid.score(test['x'], test['y'])
        return accuracy, grid.best_estimator_

    @staticmethod
    def calculate_evaluation_score(evaluation_dataset: pd.DataFrame, labelcol: str, estimator) -> float:
        """
        Calculates the score of a model predicting the given test dataset.

        :param evaluation_dataset: Data to test the model with stored in a dataframe.
        :param labelcol: Name of the column containing the labels.
        :param estimator: Trained model to predict the data.
        :return: Returns the best achieved score.
        """
        best_score = 0
        # Let the model predict the evaluation data and return the best score
        for i in range(100):
            accuracy = estimator.score(evaluation_dataset.drop(columns=[labelcol]), evaluation_dataset[labelcol])
            if accuracy > best_score:
                best_score = accuracy
        return best_score
