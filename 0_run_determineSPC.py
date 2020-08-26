import pandas as pd
import numpy as np
import os
import copy
from PreProcessing.PreProcess import PreProcess
from PreProcessing.FeatureEngineering import FeatureEngineering
import matplotlib.pyplot as plt
from PreProcessing.DetermineSPC import DetermineSPC
from multiprocessing import Pool

# Setup values
BASE_PATH = 'Data/RawData/'
REFERENCE = 'LP'
TO_DETERMINE = 'NM'
UNDERGROUND_TYPES = ['AS', 'RW', 'WW', 'SC', 'KO']

# Create empty lists to store results
best_SPCs_general = []
underground_SPCs = []

# Find mean SPC for each type in declared list
for u_type in UNDERGROUND_TYPES:
    best_SPCs_specific = []
    # Find all files from REFERENCE with type u_type
    for reference_file in os.listdir(BASE_PATH):
        if not reference_file.startswith(u_type) or not reference_file[3:5] == REFERENCE:
            continue
        # Load and prepare reference data
        reference_data = pd.read_csv(BASE_PATH + reference_file)
        reference_data.columns = ['time', 'x', 'y', 'z', 'abs']
        reference_data = PreProcess.process_outlier_detection(reference_data, ['x', 'y', 'z'])
        # Find all files from TO_DETERMINE with type u_type
        for to_fit_file in os.listdir(BASE_PATH):
            if not to_fit_file.startswith(u_type) or not to_fit_file[3:5] == TO_DETERMINE:
                continue
            # Load and prepare to_fit data
            to_fit_data = pd.read_csv(BASE_PATH + to_fit_file)
            to_fit_data.columns = ['time', 'x', 'y', 'z', 'abs']
            to_fit_data = PreProcess.process_outlier_detection(to_fit_data, ['x', 'y', 'z'])
            # Call method to get minimum similarity and best SPC
            best_SPC, min_sim = DetermineSPC.determine_minSPC(data_ref=reference_data, data_to_fit=to_fit_data)
            print(reference_file, '->', to_fit_file, 'Best SPC:', best_SPC, 'Minimum similarity:', min_sim)
            # Append SPC to lists
            best_SPCs_specific.append(best_SPC)
            if not best_SPC == 0:
                best_SPCs_general.append(best_SPC)
    # Calculate and print mean SPC for u_type
    if len(best_SPCs_specific) > 0:
        print('Mean SPC for underground type', u_type, 'is', np.mean(best_SPCs_specific))
        underground_SPCs.append(np.mean(best_SPCs_specific))
    else:
        print(f'No files to fit for underground type {u_type}')
        underground_SPCs.append('-/-')
print(15 * '- - - ')
# Print all specific type mean SPC and overall mean SPC
for i in range(0, len(underground_SPCs)):
    print('Mean SPC for', UNDERGROUND_TYPES[i], '->', underground_SPCs[i])
print('Mean general SPC for person', TO_DETERMINE, 'is', np.mean(best_SPCs_general))
