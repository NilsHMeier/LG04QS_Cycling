# LG04QS_Cycling
This repository provides the code of our Project "Ground recognition via machine Learning".
The code is split into the parts of data preprecossing, feature engineering and machine learning.
1. In the preprocessing the samples are cropped to the right period and processed with a suspension coefficient. Furthermore the data gets filtered by a Bandpass-Filter and outliers are removed via distribution-based outlier detection.
2. During the feature engineering the samples get split into periods of 10s. On each snippet the mean, max, min and standard deviation get calculated as features.
3. In the last step the features are used to train different machine learning models (KNN, DT & SVM) and optimize them for validation- and test-data.


## Required python packages:
- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- Scipy
