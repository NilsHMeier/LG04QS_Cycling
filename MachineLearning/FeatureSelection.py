import copy
import numpy as np
import pandas as pd
from MachineLearning.LearningAlgorithms import MachineLearning


class FeatureSelection:
    @staticmethod
    def forward_selection(max_features, X_train, Y_train, X_test, Y_test):
        selected_features = []
        scores = []
        # Select the number of max_features
        for i in range(0, max_features):
            print(f'Selecting feature {i+1}/{max_features}.')
            best_score = 0
            best_feature = None
            # Get the features that are not selected yet
            features_left = list(set(X_train.columns)-set(selected_features))
            for feature in features_left:
                # Add the feature to the selected features
                temp_selected = copy.deepcopy(selected_features)
                temp_selected.append(feature)
                # Prepare dictionaries for machine learning
                train_data = {'x': X_train[temp_selected], 'y': Y_train}
                test_data = {'x': X_test[temp_selected], 'y': Y_test}
                # Run learning with the temp selected features
                score, estimator = MachineLearning.svm_gridsearch(train_data, test_data)
                if score > best_score:
                    best_score = score
                    best_feature = feature
            print(f'Selected feature is {best_feature}.')
            selected_features.append(best_feature)
            scores.append(best_score)
        return selected_features, scores
