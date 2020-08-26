from PreProcessing.PreProcess import PreProcess
import pandas as pd
import os
from multiprocessing import Pool

# Set paths for data source, intermediate results and final results
SOURCE_PATH = 'Data/RawData/'
DESTINATION_PATH = 'Data/ProcessedData/'
CUT_PATH = 'Data/CuttedData/'

# Set parameter to decide what to do. True = Process new files. False = Apply Changes
NEW_FILES = False

# Call to start processing new files
if NEW_FILES:
    PreProcess.process_new_files(SOURCE_PATH, DESTINATION_PATH, CUT_PATH)

# Call to apply changes on data. Uses real multithreading.
if not NEW_FILES and __name__ == "__main__":
    with Pool(6) as p:
        p.map(PreProcess.apply_changes, os.listdir(SOURCE_PATH))
    print('Changes applied successfully!')
