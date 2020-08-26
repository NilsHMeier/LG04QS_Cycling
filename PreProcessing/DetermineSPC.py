import pandas as pd
import numpy as np
from PreProcessing.PreProcess import PreProcess


class DetermineSPC:
    @staticmethod
    def calculate_similarity(ref_data: pd.DataFrame, data_to_fit: pd.DataFrame, SPC: float) -> float:
        """
        Method to calculate the similarity between two datasets with a given SPC

        :param ref_data: Reference data for calculating similarity
        :param data_to_fit: Data that the SPC will be processed on
        :param SPC: Coefficient for data_to_fit
        :return: Similarity between datasets after SPC was processed
        :rtype: float
        """
        # Process SPC on data_to_fit
        data_to_fit = PreProcess.process_suspension_coefficient(data_to_fit, SPC, ['x', 'y', 'z'])
        differences = []
        # Calculate absolute difference of mean, std, max and min for each col
        for col in ['x', 'y', 'z']:
            differences.append(ref_data[col].mean() - data_to_fit[col].mean())
            differences.append(ref_data[col].std() - data_to_fit[col].std())
            differences.append(ref_data[col].max() - data_to_fit[col].max())
            differences.append(ref_data[col].min() - data_to_fit[col].min())
        # Calculate the amount of all differences
        similarity = np.sqrt(np.sum(np.square(dif) for dif in differences))
        return similarity

    @staticmethod
    def determine_minSPC(data_ref: pd.DataFrame, data_to_fit: pd.DataFrame) -> (float, float):
        """
        Method to determine the best SPC and the minimum similarity for two datasets

        :param data_ref: Reference data for fitting the SPC
        :param data_to_fit: Data that has to be best fitted to reference data
        :return: Returns the best SPC an the minimum similarity like SPC, SIM
        :rtype: tuple
        """
        # Create variables for saving the SPC and similarity
        best_SPC = np.inf
        min_sim = np.inf
        # Calculate the similarity for each SPC in range [0,1)
        for SPC in np.arange(0, 1, 0.01):
            sim = DetermineSPC.calculate_similarity(ref_data=data_ref, data_to_fit=data_to_fit, SPC=SPC)
            # If calculated similarity is smaller than smallest so far, save SPC and SIM
            if sim < min_sim:
                best_SPC = SPC
                min_sim = sim
        return best_SPC, min_sim
