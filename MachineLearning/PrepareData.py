import pandas as pd
import numpy as np
import os
import copy
from sklearn.model_selection import train_test_split
from PreProcessing.PreProcess import PreProcess
from PreProcessing.FeatureEngineering import FeatureEngineering


class PrepareDataset:
    # Set the features as class attributes
    frequency_features = ['mean_PSD', 'max_PSD', 'PSD_above_20', 'weighted_mean']
    statistical_features = ['mean', 'max', 'min', 'std']

    def __init__(self, source_path: str, stat: bool, freq: bool, label_col: str, cols: list):
        self.SOURCE_PATH = source_path
        # Check which features should be used and store them in features
        if stat and freq:
            self.features = PrepareDataset.statistical_features + PrepareDataset.frequency_features
        elif stat:
            self.features = PrepareDataset.statistical_features
        else:
            self.features = PrepareDataset.frequency_features
        self.LABEL_COL = label_col
        self.COLS = cols

    def fill_datatable(self, names: list) -> pd.DataFrame:
        """
        Creates a datatable with data from the path set in object for training a model. Uses the features
        that were set when creating the object

        :param names: Use a list with names out of ['LP', 'NB', 'NM'] to specify the used data.
                In case of all names just use an empty list.
        :return: Returns a dataframe containing all features and the label.
        """
        # Create a new datatable
        data_table = self.create_datatable()
        # Set a counter and get the total number of files that will be added to the datatable
        file_counter = 1
        total_files = len(os.listdir(self.SOURCE_PATH)) if len(names) == 0 else \
            np.sum(1 if file[3:5] in names else 0 for file in os.listdir(self.SOURCE_PATH))
        for file in os.listdir(self.SOURCE_PATH):
            # Check if any names are specified and if the file needs to be processed
            if not len(names) == 0 and file[3:5] not in names:
                continue
            print('Adding file', file, 'to dataset', f'({file_counter}/{total_files})')
            # Load the csv file, drop unneeded columns, add the label col and append it to the datatable
            data = pd.read_csv(self.SOURCE_PATH + file, index_col=0)
            data = data.drop(columns=[col for col in data.columns if col not in data_table.columns])
            data[self.LABEL_COL] = file[0:2]
            data_table = data_table.append(data, ignore_index=True)
            file_counter += 1
        return data_table

    def create_datatable(self) -> pd.DataFrame:
        """
        Creates an empty dataframe with all features and a label col.

        :return: Empty dataframe with columns.
        """
        cols = list(f'{col}_{feature}' for col in self.COLS for feature in self.features)
        cols.append(self.LABEL_COL)
        data_table = pd.DataFrame(columns=cols)
        return data_table

    def split_dataset_default(self, dataset: pd.DataFrame, test_size: float) -> tuple:
        """
        Splits a dataset into training and test data using train_test_split of sklearn.model_selection.
        Keeps all the features and cols set when creating the object.

        :param dataset: Dataframe containing all the data.
        :param test_size: Percentage that will be used for test data.
        :return: Returns a tuple of four dataframes like x_train, x_test, y_train, y_test
        """
        feature_cols = list(f'{col}_{feature}' for col in self.COLS for feature in self.features)
        feature_data = dataset.drop(columns=[col for col in dataset.columns if col not in feature_cols])
        label_data = dataset[self.LABEL_COL]
        return train_test_split(feature_data, label_data, test_size=test_size, shuffle=True)

    def split_dataset_selected(self, dataset: pd.DataFrame, selected_features: list, test_size: float) -> tuple:
        """
        Splits a dataset into training and test data using train_test_split of sklearn.model_selection.
        Only uses the given selected features and drops all others.

        :param dataset: Dataframe containing all the data.
        :param selected_features: Features that will be used for splitting the data.
        :param test_size: Percentage that will be used for test data.
        :return: Returns a tuple of four dataframes like x_train, x_test, y_train, y_test
        """
        feature_data = dataset.drop(columns=[col for col in dataset.columns if col not in selected_features])
        label_data = dataset[self.LABEL_COL]
        return train_test_split(feature_data, label_data, test_size=test_size, shuffle=True)

    def prepare_evaluation_dataset(self, source_path: str, names: list,
                                   start_offset: int, end_offset: int) -> pd.DataFrame:
        """
        Method to create a dataset from a given path containing raw unprocessed data. Runs preprocessing and
        feature engineering on the data and appends it to the dataset.

        :param source_path: Path to the folder containing the data.
        :param names: Specify names that should be used.
        :param start_offset: Period that will be deleted at the start of raw data.
        :param end_offset: Period that will be deleted at the end of raw data.
        :return: Returns a dataframe containing features and labels of the raw data.
        """
        # Create empty dataframe to store the features and labels in
        data_table = pd.DataFrame()
        file_counter = 1
        total_files = np.sum(1 if file[3:5] in names else 0 for file in os.listdir(source_path))
        # Process all files in the given folder
        for file in os.listdir(source_path):
            # Check if the file should be processed.
            if file[3:5] not in names:
                continue
            print('Adding file', file, 'to dataset', f'({file_counter}/{total_files})')
            # Read the raw data in and prepare it for preprocessing
            raw_data = pd.read_csv(source_path + file)
            raw_data.columns = ['time', 'x', 'y', 'z', 'abs']
            raw_data.drop(columns=['abs'])
            # Process an offset on the data with the given start and end values
            max_time = raw_data['time'].max()
            offset_data = raw_data[((raw_data['time'] >= start_offset) & (raw_data['time'] < max_time - end_offset))]
            offset_data['time'] = offset_data['time'] - start_offset
            offset_data.index = offset_data.index - offset_data.index.min()
            # Put the SPC on the data
            coefficient_data = PreProcess.process_suspension_coefficient(
                offset_data, PreProcess.suspension_coefficients[file[3:5]], ['x', 'y', 'z'])
            # Process the bandpass filter
            filtered_data = PreProcess.process_filter(
                coefficient_data, PreProcess.frequencies[file[3:5]], ['x', 'y', 'z'])
            # Remove outliers
            outlier_data = PreProcess.process_outlier_detection(filtered_data, ['x', 'y', 'z'])
            # Generate features out of the processed data, add label col & delete time col and append it to dataset
            engineer = FeatureEngineering('', '', 10)
            aggregated_data = engineer.aggregate_data(outlier_data, self.COLS, file[3:5])
            aggregated_data[self.LABEL_COL] = file[0:2]
            aggregated_data = aggregated_data.drop(columns=['time'])
            data_table = data_table.append(aggregated_data, ignore_index=True)
            file_counter += 1
        used_features = list(f'{col}_{feature}' for col in self.COLS for feature in self.features)
        data_table = data_table.drop(columns=[col for col in data_table.columns
                                              if col != 'label' and col not in used_features])
        return data_table
