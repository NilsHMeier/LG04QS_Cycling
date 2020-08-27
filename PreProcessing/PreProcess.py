import copy
import math
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import special
from Util.Data_Vizualizer import Vizualizer


class PreProcess:
    suspension_coefficients = {'LP': 0, 'NM': 0.3082, 'NB': 0.4805, 'NP': 0.44}
    frequencies = {'NB': 200, 'NM': 100, 'LP': 100, 'NP': 200}

    @staticmethod
    def process_new_files(source_path: str, destination_path: str, cut_path: str):
        """
        Runs all steps of preprocessing on new files in source_path, which were not processed before.

        :param source_path: Source of the data to be processed. Has to end with /
        :param destination_path: Target path to save the processed data in. Has to end with /
        :param cut_path: Path to save the cropped files in. Has to end with /
        """
        destination_files = os.listdir(destination_path)
        file_counter = 1
        total_files = len(os.listdir(source_path))
        # Do Preprocessing for all files in given directory
        for file in os.listdir(source_path):
            # Check if file was already processed. If that's the case just skip it.
            if file in destination_files:
                print('Skipping file', file, f'({file_counter}/{total_files})')
                file_counter += 1
                continue
            # Starting proprocessing on file
            print('Running preprocessing on file', file, f'({file_counter}/{total_files})')

            # Read in the file and modify the dataframe
            raw_data = pd.read_csv(source_path + file)
            raw_data.columns = ['time', 'x', 'y', 'z', 'abs']
            # Run all steps of preprocessing
            raw_data = PreProcess.process_offset(raw_data)
            # Save cropped data to be able to apply changes later on
            raw_data.to_csv(cut_path + file)
            coefficient = PreProcess.suspension_coefficients[file[3:5]]
            raw_data = PreProcess.process_suspension_coefficient(raw_data, coefficient, ['x', 'y', 'z'])
            raw_data = PreProcess.process_filter(raw_data, PreProcess.frequencies[file[3:5]])
            raw_data = PreProcess.process_outlier_detection(raw_data, ['x', 'y', 'z'])
            # Save processed data to CSV file
            raw_data.to_csv(destination_path + file)
            file_counter += 1

    @staticmethod
    def process_offset(raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Crops the data to the needed period. User has to enter start end stop values
        after taking a look on the data.

        :param raw_data: Data to process the offset on.
        :return: Returns the cropped data with index and time col starting at 0.
        :rtype: pd.Dataframe
        """
        # Copy raw_data and prepare it for cropping
        processed_data = copy.deepcopy(raw_data)
        processed_data.columns = ['time', 'x', 'y', 'z', 'abs']
        processed_data = processed_data.drop(columns=['abs'])

        # Show data that the user can decide which offset to choose
        Vizualizer.plot_line(processed_data, 'time', ['x', 'y', 'z'], 3, 1)

        # Let the user enter start and end time
        start = int(input("Start (s): "))
        end = int(input("End (s): "))
        # Select the relevant rows. Change index and time column, that they start from 0 again
        processed_data = processed_data[(processed_data['time'] >= start) & (processed_data['time'] <= end)]
        processed_data['time'] = processed_data['time'] - start
        processed_data.index = processed_data.index - processed_data.index.min()
        return processed_data

    @staticmethod
    def process_suspension_coefficient(raw_data: pd.DataFrame, coefficient: float, cols: list) -> pd.DataFrame:
        """
        Applies the suspension coefficient on the data.

        :param raw_data: Data to apply the coefficient on.
        :param coefficient: Coefficient to use.
        :param cols: Cols in data to apply coefficient on.
        :return: Data after coefficient has been applied.
        """
        processed_data = copy.deepcopy(raw_data)
        # Apply coefficient for each col
        for col in cols:
            processed_data[col] = raw_data[col] / (1 - coefficient)
        return processed_data

    # noinspection PyDefaultArgument
    @staticmethod
    def process_filter(raw_data: pd.DataFrame, f: int, cols: list = ['x', 'y', 'z']) -> pd.DataFrame:
        """
        Applies a bandpass-filter on the given cols in the data. Filters out frequencies that are
        smaller than 5 or greater than 30 Hz.

        :param raw_data: Data to apply the filter on.
        :param f: Frequency the data has been recorded with.
        :param cols: Cols to apply the filter on.
        :return: Returns the filtered data.
        """
        processed_data = copy.deepcopy(raw_data)
        # Create 'blueprint' filter for the data with 5 & 30 as cutter freqs and 10 as filter ordner
        sos = signal.butter(10, (5, 30), btype='bandpass', fs=f, output='sos')
        # Apply filter to each col
        for col in cols:
            processed_data[col] = signal.sosfilt(sos, raw_data[col])
        return processed_data

    @staticmethod
    def process_outlier_detection(data: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Finds and replaces outliers in the given data by distribution-based outlier detection.

        :param data: Data to do the outlier detection on.
        :param cols: Specific cols to use.
        :return: Data with replaced outliers by interpolation.
        """
        data_table = copy.deepcopy(data)
        for col in cols:
            # Compute the mean & standard deviation and set criterion
            mean = data_table[col].mean()
            std = data_table[col].std()
            criterion = 0.005
            # Consider the deviation for the data points
            deviation = abs(data_table[col] - mean) / std
            # Express the upper and lower bounds
            low = -deviation / math.sqrt(2)
            high = deviation / math.sqrt(2)
            prob = []
            mask = []
            # Pass all rows in the dataset.
            for i in data_table.index:
                # Determine the probability of observing the point
                prob.append(1.0 - 0.5 * (special.erf(high[i]) - special.erf(low[i])))
                # And mark as an outlier when the probability is below our criterion
                mask.append(prob[i] < criterion)
            data_table[col + '_outlier'] = mask
            data_table.loc[data_table[f'{col}_outlier'], col] = np.nan
            del data_table[col + '_outlier']
            data_table = PreProcess.impute_interpolate(data_table, col)
        return data_table

    @staticmethod
    def impute_interpolate(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fills missing values after outlier detection with interpolated values.

        :param dataset: Data with outliers as NANs.
        :param col: Col in which to impute values.
        :return: Returns data without NANs.
        """
        # Fill NANs with interpolated values
        dataset[col] = dataset[col].interpolate()
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset

    @staticmethod
    def apply_changes(file: str):
        """
        Applies changes on a single file quickly, in case parameters in preprocessing have changed.
        Can be used for multithreading.

        :param file: Name of the file. Has to end with .csv
        """
        # Set source and destination path
        SOURCE_PATH = 'Data/CuttedData/'
        DESTINATION_PATH = 'Data/ProcessedData/'
        # Tell the user which file is processed
        print(f'Processing file {file}')
        # Read the file in and do all steps of preprocessing exept offset
        data = pd.read_csv(SOURCE_PATH + file, index_col=0)
        coefficient = PreProcess.suspension_coefficients[file[3:5]]
        data = PreProcess.process_suspension_coefficient(data, coefficient, ['x', 'y', 'z'])
        data = PreProcess.process_filter(data, 200 if file[3:5] == 'NB' else 100)
        data = PreProcess.process_outlier_detection(data, ['x', 'y', 'z'])
        data.to_csv(DESTINATION_PATH + file)
