from PreProcessing.FeatureEngineering import FeatureEngineering
from multiprocessing import Pool
import os

# Set path for source and destination folder
SOURCE_PATH = 'Data/ProcessedData/'
DESTINATION_PATH = 'Data/AggregatedData/'
# Set the type period for feature engineering
PERIOD = 10
# Decide to use multithreading and set the number of threads
MULTITHREADING: bool = True
THREADS: int = 4
# Create object of class FeatureEngineering
engineer = FeatureEngineering(SOURCE_PATH, DESTINATION_PATH, PERIOD)
# Calling method of engineer to use no multithreading
if not MULTITHREADING:
    engineer.process_files()
    print('Feature engineering completed successfully!')
# Start feature engineering with multithreading
if MULTITHREADING and __name__ == '__main__':
    with Pool(THREADS) as p:
        p.map(engineer.process_multithreading, os.listdir(SOURCE_PATH))
    print('Feature engineering completed successfully!')
