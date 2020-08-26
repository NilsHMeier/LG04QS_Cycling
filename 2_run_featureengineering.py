from PreProcessing.FeatureEngineering import FeatureEngineering
from multiprocessing import Pool
import os

# Set path for source and destination folder
SOURCE_PATH = 'Data/ProcessedData/'
DESTINATION_PATH = 'Data/AggregatedData/'
# Decide to use multithreading and set the number of threads
MULTITHREADING: bool = True
THREADS: int = 6

# Calling method in package FeatureEngineering to use no multithreading
if not MULTITHREADING:
    FeatureEngineering.process_files(SOURCE_PATH, DESTINATION_PATH, 5)
    print('Feature engineering completed successfully!')
# Start feature engineering with multithreading
if MULTITHREADING and __name__ == '__main__':
    with Pool(THREADS) as p:
        p.map(FeatureEngineering.process_multithreading, os.listdir(SOURCE_PATH))
    print('Feature engineering completed successfully!')
