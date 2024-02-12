# Load the trained model
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_and_preprocess_data(file_path, sequence_length):
    # Load data from .mat file
    data = scipy.io.loadmat(file_path)

    # Assuming 'sig' is the key for PPG signal data
    ppg_data = data['sig']

    # Preprocess the data (you may need to adjust this based on your specific requirements)
    # For example, normalizing the PPG signal data to have zero mean and unit variance
    ppg_data = (ppg_data - np.mean(ppg_data, axis=0)) / np.std(ppg_data, axis=0)

    # Reshape into a 3D array, including sequence_length dimension
    ppg_data_reshaped = np.reshape(ppg_data, ( sequence_length, 1))

    return ppg_data_reshaped



loaded_model = load_model('your_model.h5')


# Assuming 'new_data' is your new data
# Adjust 'sequence_length' based on the length of your new sequence
new_data_preprocessed = load_and_preprocess_data('/Users/drake/OneDrive/Documents/PPG Research/DATABASE/Training_data/DATA_02_TYPE02.mat',227100)


predictions = loaded_model.predict(new_data_preprocessed)

# Assuming 'ground_truth_labels' are available
evaluation_results = loaded_model.evaluate(new_data_preprocessed)
threshold = 0.5

# Find matching indices based on the threshold
matching_indices = np.where(predictions > threshold)[0]

# If matching_indices is not empty, it indicates the indices of data points that match the model
if len(matching_indices) > 0:
    print("Matching data found in the model.")
    # You can access the corresponding data points in the training set using matching_indices
    matching_data = X_train[matching_indices]
    # Perform further analysis or visualization with matching_data
else:
    print("No matching data found in the model.")
