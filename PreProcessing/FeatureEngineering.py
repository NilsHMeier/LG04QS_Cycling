import os
import pandas as pd
import numpy as np


class FeatureEngineering:
    # Set the features as class attributes
    frequency_features = ['mean_PSD', 'max_PSD', 'PSD_above_20', 'weighted_mean']
    statistical_features = ['mean', 'max', 'min', 'std', 'range']

    def __init__(self, source_path: str, destination_path: str, period: int):
        """
        Creates an object of FeatureEngineering with the given parameters used in feature engineering.

        :param source_path: Path to the folder containing the files. Has to end with /.
        :param destination_path: Path to the folder to save the results in. Has to end with /.
        :param period: Period used in feature engineering.
        """
        self.SOURCE_PATH = source_path
        self.DESTINATION_PATH = destination_path
        self.PERIOD = period

    def process_files(self):
        """
        Method to run the feature engineering on all files in the source path and save the results
        in the destination path set when creating the object. Uses period as time range to calculate the features on.
        """
        # Create counter and get number of files to show user the progress
        file_counter = 1
        total_files = len(os.listdir(self.SOURCE_PATH))
        # Do feature engineering for all files in given directory
        for file in os.listdir(self.SOURCE_PATH):
            print('Doing feature engineering on file', file, f'({file_counter}/{total_files})')
            # Read data in, do feature engineering and save the results to given directory
            processed_data = pd.read_csv(self.SOURCE_PATH + file, index_col=0)
            aggregated_data = self.aggregate_data(processed_data, ['x', 'y', 'z'], file[3:5])
            aggregated_data.to_csv(self.DESTINATION_PATH + file)
            file_counter += 1

    def create_dataset(self, cols: list, max_time: int) -> pd.DataFrame:
        """
        Method to create a new dataframe with timestamps in to save aggregated data.

        :param cols: Cols to create the feature cols for.
        :param max_time: Maximum time of the dataset to aggregate. Has to be an integer.
        :return: Returns the dataframe with timestamp col named 'time'. All other values are NANs.
        """
        # Create new dataframe with timestamps in time col
        aggregated_data = pd.DataFrame(list(range(0, max_time, self.PERIOD)), columns=['time'])
        features = FeatureEngineering.statistical_features + FeatureEngineering.frequency_features
        # Create all other cols and fill them with NANs
        for col in cols:
            for feature in features:
                aggregated_data[f'{col}_{feature}'] = np.nan
        return aggregated_data

    def aggregate_data(self, raw_data: pd.DataFrame, cols: list, cyclist: str) -> pd.DataFrame:
        """
        The method creates a new datatable with timestamps to the maximum time of the given raw data.
        For each timestamp / period it calculates statistical and frequency and saves it in the datatable.

        :param raw_data: Data to do the feature engineering on.
        :param cols: Specific cols to use in calculating the features.
        :return: Returns the filled datatable with the calculated features.
        :param cyclist: String holding two characters to identify who took the sample to set the frequency.
        """
        # Create new datatable with timestamps to max time
        data_table = self.create_dataset(cols, int(raw_data['time'].max()))
        f = 200 if cyclist in ['NB', 'NP'] else 100
        # For each timestamp calculate the features
        for timestamp in data_table['time']:
            # Select the relevant rows of the raw data
            relevant_rows = raw_data[((raw_data['time'] >= timestamp) & (raw_data['time'] < timestamp + self.PERIOD))]
            # Calculate mean, max, min and std of each col and save it in created datatable
            for col in cols:
                stat_features = self.calculate_statistical_features(relevant_rows[col])
                freq_features = self.calculate_frequency_features(relevant_rows[col], f)
                features = {**stat_features, **freq_features}
                for feature in features:
                    data_table.loc[timestamp / self.PERIOD, f'{col}_{feature}'] = features[feature]
        return data_table

    @staticmethod
    def calculate_statistical_features(values: pd.Series):
        """
        Calculates the statistical features for the given values.

        :param values: Values to calculate the features on.
        :return: Returns a dictionary containing the features with the names as keys.
        """
        results = {'mean': np.mean(values), 'max': np.max(values), 'min': np.min(values),
                   'range': np.max(values) - np.min(values), 'std': np.std(values)}
        return results

    @staticmethod
    def calculate_frequency_features(values: pd.Series, f: int):
        """
        Calculates the frequency features for the given values using the capturing frequency for FFT.

        :param values: Values to calculate the features on.
        :param f: Frequency the sample has been recorded with.
        :return: Returns a dictionary containing the features with the names as keys.
        """
        dt = 1 / f
        n = len(values)
        fhat = np.fft.fft(values, n)
        PSD = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)
        data = pd.DataFrame(data={'freq': freq, 'PSD': PSD, 'FHAT': fhat})
        data['weighted'] = data['freq'] * data['PSD']
        data = data[((data['freq'] >= 5) & (data['freq'] <= 30))]
        results = {'mean_PSD': np.mean(data['PSD']).real, 'max_PSD': np.max(data['PSD']).real,
                   'PSD_above_20': sum(1 if psd.real > 20 else 0 for psd in data['PSD']),
                   'weighted_mean': np.mean(data['weighted']).real}
        return results

    def process_multithreading(self, file: str):
        """
        Does the feature engineering on a given single file. Can be used for multithreading.

        :param file: Name of the file to process, has to be a CSV file.
        """
        print('Doing feature engineering on file', file)
        # Read data in, do feature engineering and save results to the set directory
        processed_data = pd.read_csv(self.SOURCE_PATH + file, index_col=0)
        aggregated_data = self.aggregate_data(processed_data, ['x', 'y', 'z'], file[3:5])
        aggregated_data.to_csv(self.DESTINATION_PATH + file)
