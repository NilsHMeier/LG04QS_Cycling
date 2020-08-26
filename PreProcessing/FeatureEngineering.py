import os
import pandas as pd
import numpy as np


class FeatureEngineering:
    @staticmethod
    def process_files(source_path: str, destination_path: str, period: int):
        """
        Method to run the feature engineering on all files in the given source path and save the results
        in the given destination path. Uses period as time range to calculate the features on.

        :param source_path: Source path of the data that will be processed. Has to end with /
        :param destination_path: Path to the folder the results will be saved in. Has to end with /
        :param period: Period of time for the features.
        """
        # Create counter and get number of files to show user the progress
        file_counter = 1
        total_files = len(os.listdir(source_path))
        # Do feature engineering for all files in given directory
        for file in os.listdir(source_path):
            print('Doing feature engineering on file', file, f'({file_counter}/{total_files})')
            # Read data in, do feature engineering and save the results to given directory
            processed_data = pd.read_csv(source_path + file, index_col=0)
            aggregated_data = FeatureEngineering.aggregate_data(processed_data, ['x', 'y', 'z'], period)
            aggregated_data.to_csv(destination_path + file)
            file_counter += 1

    @staticmethod
    def create_dataset(cols: list, period: int, max_time: int) -> pd.DataFrame:
        """
        Method to create a new dataframe with timestamps in to save aggregated data.

        :param cols: Cols to create the feature cols for.
        :param period: Period of time for the timestamps.
        :param max_time: Maxmimum time of the dataset to aggregate. Has to be an integer.
        :return: Returns the dataframe with timestap col named 'time'. All other values are NANs.
        """
        # Create new dataframe with timestamps in time col
        aggregated_data = pd.DataFrame(list(range(0, max_time, period)), columns=['time'])
        # Create all other cols and fill them with NANs
        for col in cols:
            aggregated_data[f'{col}_mean'] = np.nan
            aggregated_data[f'{col}_max'] = np.nan
            aggregated_data[f'{col}_min'] = np.nan
            aggregated_data[f'{col}_std'] = np.nan
        return aggregated_data

    @staticmethod
    def aggregate_data(raw_data: pd.DataFrame, cols: list, period: int) -> pd.DataFrame:
        """
        The method creates a new datatable with timestamps to the maximum time of the given raw data.
        For each timestamp / period it calculates mean, max, min and std and saves it in the datatable.

        :param raw_data: Data to do the feature engineering on.
        :param cols: Specific cols to use in calculating the features.
        :param period: Period of time used for the timestamps.
        :return: Returns the filled datatable with the calculated features.
        """
        # Create new datatable with timestamps to max time
        data_table = FeatureEngineering.create_dataset(cols, period, int(raw_data['time'].max()))
        # For each timestamp calculate the features
        for timestamp in data_table['time']:
            # Select the relevant rows of the raw data
            relevant_rows = raw_data[((raw_data['time'] >= timestamp) & (raw_data['time'] < timestamp + period))]
            # Calucalte mean, max, min and std of each col and save it in created datatable
            for col in cols:
                data_table.loc[timestamp / period, str(col) + str('_mean')] = np.mean(relevant_rows[col])
                data_table.loc[timestamp / period, str(col) + str('_max')] = np.max(relevant_rows[col])
                data_table.loc[timestamp / period, str(col) + str('_min')] = np.min(relevant_rows[col])
                data_table.loc[timestamp / period, str(col) + str('_std')] = np.std(relevant_rows[col])
        return data_table

    @staticmethod
    def process_multithreading(file: str):
        """
        Does the feature engineering on a given single file. Can be used for multithreading.
        Source and destination path have to be set within the method.

        :param file: Name of the file to process, has to be a CSV file.
        """
        # Set source and destination path
        source_path = 'Data/ProcessedData/'
        destination_path = 'Data/AggregatedData/'
        print('Doing feature engineering on file', file)
        # Read data in, do feature engineering and save results to the set directory
        processed_data = pd.read_csv(source_path + file, index_col=0)
        aggregated_data = FeatureEngineering.aggregate_data(processed_data, ['x', 'y', 'z'], 10)
        aggregated_data.to_csv(destination_path + file)
