import pandas as pd
import numpy as np
import os
import copy
from sklearn.model_selection import train_test_split
from PreProcessing.PreProcess import PreProcess
from PreProcessing.FeatureEngineering import FeatureEngineering


class PrepareDataset:
    @staticmethod
    def fill_datatable(source_path: str, labelcol_name: str, names: list) -> pd.DataFrame:
        """
        Method to create a datatable with data from a given path for training a model. Assumes that features are the
        same as the one generated in feature engineering.

        :param source_path: Path to the folder containing the files to be added. Has to end with /
        :param labelcol_name: Name of the col that will contain the label of the features.
        :param names: Use a list with names out of ['LP', 'NB', 'NM'] to specify the used data.
                In case of all names just use an empty list.
        :return: Returns a dataframe containing all features and the label.
        """
        # Create a new datatable with the given label col name
        data_table = PrepareDataset.create_datatable(labelcol_name)
        # Set a counter and get the total number of files that will be added to the datatable
        file_counter = 1
        total_files = len(os.listdir(source_path)) if len(names) == 0 else \
            np.sum(1 if file[3:5] in names else 0 for file in os.listdir(source_path))
        for file in os.listdir(source_path):
            # Check if any names are specified and if the file needs to be processed
            if not len(names) == 0 and file[3:5] not in names:
                continue
            print('Adding file', file, 'to dataset', f'({file_counter}/{total_files})')
            # Load the csv file, drop unneeded columns, add the label col and append it to the datatable
            data = pd.read_csv(source_path + file, index_col=0)
            data = data.drop(columns=[col for col in data.columns if col not in data_table.columns])
            data[labelcol_name] = file[0:2]
            data_table = data_table.append(data, ignore_index=True)
            file_counter += 1
        return data_table

    @staticmethod
    def create_datatable(label_col: str) -> pd.DataFrame:
        """
        Method to create an empty dataframe with all features and a label col.

        :param label_col: Name of the label col.
        :return: Empty dataframe with columns.
        """
        cols = list(col + feature for col in ['x', 'y', 'z'] for feature in ['_mean', '_max', '_min', '_std'])
        cols.append(label_col)
        data_table = pd.DataFrame(columns=cols)
        return data_table

    @staticmethod
    def split_dataset(dataset: pd.DataFrame, feature_cols: list, label_col: str, test_size: float) -> tuple:
        """
        Method to split a dataset into training and test data using train_test_split of sklearn.model_selection.

        :param dataset: Dataframe containing all the data.
        :param feature_cols: List of the cols that contain features.
        :param label_col: Name of the label col.
        :param test_size:
        :return: Returns a tuple of four dataframes like x_train, x_test, y_train, y_test
        """
        feature_data = dataset.drop(columns=[col for col in dataset.columns if col not in feature_cols])
        label_data = dataset[label_col]
        return train_test_split(feature_data, label_data, test_size=test_size, shuffle=True)

    @staticmethod
    def prepare_evaluation_dataset(sourcepath: str, labelcol_name: str, names: list,
                                   start_offset: int, end_offset: int) -> pd.DataFrame:
        """
        Method to create a dataset from a given path containing raw unprocessed data. Runs preprocessing and
        featureengineering on the data and appends it to the dataset.

        :param sourcepath: Path to the folder containing the data.
        :param labelcol_name: Name of the label col
        :param names: Specify names that should be used.
        :param start_offset: Period that will be deleted at the start of raw data.
        :param end_offset: Period that will be deleted at the end of raw data.
        :return: Returns a dataframe containing features and labels of the raw data.
        """
        # Create empty dataframe to store the features and labels in
        data_table = PrepareDataset.create_datatable(labelcol_name)
        file_counter = 1
        total_files = np.sum(1 if file[3:5] in names else 0 for file in os.listdir(sourcepath))
        # Process all files in the given folder
        for file in os.listdir(sourcepath):
            # Check if the file should be processed.
            if file[3:5] not in names:
                continue
            print('Adding file', file, 'to dataset', f'({file_counter}/{total_files})')
            # Read the raw data in and prepare it for preprocessing
            raw_data = pd.read_csv(sourcepath + file)
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
            aggregated_data = FeatureEngineering.aggregate_data(outlier_data, ['x', 'y', 'z'], 10)
            aggregated_data[labelcol_name] = file[0:2]
            aggregated_data = aggregated_data.drop(columns=['time'])
            data_table = data_table.append(aggregated_data, ignore_index=True)
            file_counter += 1
        return data_table
