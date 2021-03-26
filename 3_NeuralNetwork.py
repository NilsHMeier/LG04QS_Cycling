import tensorflow as tf
from tensorflow import keras
from MachineLearning.PrepareData import PrepareDataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from matplotlib import pyplot as plt

# Set constants
SOURCE_PATH = 'Data/AggregatedData/'
USE_STAT = True
USE_FREQ = False
LABEL_COL = 'label'
COLS = ['x', 'y', 'z']
THREE_WAY = True
USE_TEST = True

# Load train data
data_engineer = PrepareDataset(SOURCE_PATH, USE_STAT, USE_FREQ, LABEL_COL, COLS)
dataset = data_engineer.fill_datatable(['LP', 'NB', 'NM'])

# Load test dataset
evaluation_datatable = data_engineer.prepare_evaluation_dataset('Data/EvaluationData/', ['NM', 'LP'], 15, 15)
# Make sure evaluation data has the same form as training data
evaluation_datatable = evaluation_datatable.drop(columns=[col for col in evaluation_datatable.columns
                                                          if col != 'label' and col not in dataset.columns])

# Refactor labels to three way classification if constant is set
labels = {'AS': 'good', 'RW': 'medium', 'WW': 'medium', 'SC': 'medium', 'KO': 'bad'}
if THREE_WAY:
    dataset[LABEL_COL] = [labels[key] for key in dataset[LABEL_COL]]
    evaluation_datatable[LABEL_COL] = [labels[key] for key in evaluation_datatable[LABEL_COL]]

if USE_TEST:
    X_test, y_test = evaluation_datatable.drop(columns=[LABEL_COL]), evaluation_datatable[LABEL_COL]
else:
    dataset = dataset.append(evaluation_datatable, ignore_index=True)
X_train, X_val, y_train, y_val = data_engineer.split_dataset_default(dataset, 0.3)

# Encode labels
encoder = OneHotEncoder()
encoder.fit(y_train.to_numpy().reshape(-1, 1))
y_train_enc = encoder.transform(y_train.to_numpy().reshape(-1, 1)).toarray()
y_val_enc = encoder.transform(y_val.to_numpy().reshape(-1, 1)).toarray()
if USE_TEST:
    y_test_enc = encoder.transform(y_test.to_numpy().reshape(-1, 1)).toarray()

# Parameters for neural network
EPOCHS = 500
BATCH_SIZE = 50
OUTPUT_LAYER = 3 if THREE_WAY else 5
INPUT_SHAPE = 15 * USE_STAT + 12 * USE_FREQ

# Create neural network
model = keras.Sequential([
    keras.layers.Dense(32, activation=keras.activations.relu, input_shape=[INPUT_SHAPE]),
    keras.layers.Dense(16, activation=keras.activations.relu),
    keras.layers.Dense(OUTPUT_LAYER, activation=keras.activations.softmax)
])
model.compile(optimizer=keras.optimizers.Adam(0.001), loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


history = model.fit(x=X_train, y=y_train_enc, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val_enc))
print(f'Training -> {model.evaluate(x=X_train, y=y_train_enc)}')
print(f'Validation -> {model.evaluate(x=X_val, y=y_val_enc)}')
if USE_TEST:
    print(f'Test -> {model.evaluate(x=X_test, y=y_test_enc)}')

# Plot losses and accuracies
losses = history.history['loss']
val_losses = history.history['val_loss']
accuracies = history.history['accuracy']
val_accuracies = history.history['val_accuracy']
epochs = np.arange(0, EPOCHS, 1)
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(epochs, accuracies, label='Training')
ax[0].plot(epochs, val_accuracies, label='Validation')
ax[0].set(title='Accuracy', xlim=(0, EPOCHS))
ax[0].legend()
ax[1].plot(epochs, losses, label='Training')
ax[1].plot(epochs, val_losses, label='Validation')
ax[1].set(title='Loss', xlim=(0, EPOCHS))
ax[1].legend()
plt.show()
