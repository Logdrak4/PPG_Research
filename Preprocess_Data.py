import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

def preprocess_data(resized_data):
    # Apply bandpass filter to PPG signals
    ppg_signals_filtered = apply_bandpass_filter(resized_data)
    print("PPG signals filtered shape:", ppg_signals_filtered.shape)

    # Normalize PPG signals
    ppg_signals_normalized = normalize_signals(ppg_signals_filtered)
    print("PPG signals normalized shape:", ppg_signals_normalized.shape)

    # Average the PPG signals
    # ppg_signals_averaged = average_signals(ppg_signals_normalized)
    # print("PPG signals shape:", ppg_signals_averaged.shape)
    #
    # # Average the acceleration signals
    # accel_signals_averaged = average_signals(resized_data)
    # print("Acceleration signals shape:", accel_signals_averaged.shape)

    # Merge PPG and acceleration signals
    #merged_signals = np.concatenate((ppg_signals_normalized, accel_sig), axis=1)
    # print("Merged signals data shape:", merged_signals.shape)

    return ppg_signals_normalized

def apply_bandpass_filter(signals):
    b, a = butter(4, [0.1, 18], btype='bandpass', fs=125)
    filtered_signals = filtfilt(b, a, signals)
    return filtered_signals

def normalize_signals(signals):
    scaler = StandardScaler()
    normalized_signals = scaler.fit_transform(signals.reshape(-1, signals.shape[-1])).reshape(signals.shape)
    return normalized_signals

def average_signals(signals):
    # Check if signals is a 1D array

        # If signals is a 2D array, compute the mean along the last axis
        averaged_signals = np.mean(signals, axis=1,keepdims= True)
        return averaged_signals
# Assuming you have resized_data in the shape of (2, 500)
# Preprocess the data
