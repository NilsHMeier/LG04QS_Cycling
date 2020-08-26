import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PreProcessing.DetermineSPC import DetermineSPC
from PreProcessing.PreProcess import PreProcess

# Load and prepare reference data
ref_data = pd.read_csv('Data/RawData/AS_LP_Landwehr Bardowick.csv')
ref_data.columns = ['time', 'x', 'y', 'z', 'abs']
ref_data = PreProcess.process_outlier_detection(ref_data, ['x', 'y', 'z'])

# Load and prepare to_fit data
to_fit_data = pd.read_csv('Data/RawData/AS_NM_Landwehr Bardowick.csv')
to_fit_data.columns = ['time', 'x', 'y', 'z', 'abs']
to_fit_data = PreProcess.process_outlier_detection(to_fit_data, ['x', 'y', 'z'])

best_SPC = np.inf
min_sim = np.inf
SPCs = []
similarities = []
# Calculate similarity for each SPC from [0,1) with 0.01 step size
for SPC in np.arange(-1, 1, 0.01):
    print(f'Calculating similarity for SPC={SPC}')
    sim = DetermineSPC.calculate_similarity(ref_data=ref_data, data_to_fit=to_fit_data, SPC=SPC)
    if sim < min_sim:
        best_SPC = SPC
        min_sim = sim
    # Save SPC and similarity in lists to plot them later on
    SPCs.append(SPC)
    similarities.append(sim)

# Plot SPC on x axis and calculated similarity on  axis
plt.plot(SPCs, similarities)
print(f'Minimum similarity {min_sim} with SPC {best_SPC}')
plt.xlim(-0.4, 0.6)
plt.xlabel('Suspension Coefficient SPC')
plt.ylim(2, 15)
plt.ylabel('Similarity SIM')
plt.savefig('Figures/DetermineSPC_AS.png')
plt.show()

# Create plot showing the factor each SPC represents
spc_range = np.arange(0, 1, 0.001)
plt.plot(spc_range, 1 / (1 - spc_range))
plt.ylim(0, 10)
plt.xlabel('SPC')
plt.ylabel('Factor')
plt.show()
