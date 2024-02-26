import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def mat_prcoessing():

    file_path = '/Users/drake/OneDrive/Documents/PPG Research/DATABASE/Training_data/DATA_01_TYPE01.mat'
    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)
    # Select data from file
    data = mat_data['sig']
    print("Keys available in the loaded .mat file:")
    print(list(mat_data.keys()))

    print("Data from the loaded .mat file:")
    for key, value in mat_data.items():
        if not key.startswith("__"):
        # np.set_printoptions(threshold=np.inf)
         print(f"Array: {key}, Shape: {value.shape}")
         print(value)

    original_data = data;
    print ("data shape test: ",data[2:4,:500])
    PPG_data = data[2:4,:500]
    #or as banavar said
    #PPG_data = data[2,:100]
    #PPG_data_2 = data[3,:100]
    print("Data size is: ", PPG_data.shape)
# # Reshape the original data into windows of length 500 with a 25 sample point (0.2 seconds) shift
#     window_length = 500
#     shift = 25
#     num_windows = (original_data.shape[1] - window_length) // shift + 1
#
# # Create an empty array to store the resized data
#     resized_data = np.zeros((2, num_windows, window_length))
#
#     #   Extract PPG and acceleration signals for each window
#     for i in range(num_windows):
#         start_idx = i * shift
#         end_idx = start_idx + window_length
#
#     # PPG signal (top signal)
#     ppg_signal = original_data[0, start_idx:end_idx]
#
#     # Acceleration signal (bottom signal)
#     accel_signal = original_data[1, start_idx:end_idx]
#
#     #  Store the signals in the resized data array
#     resized_data[0, i, :] = ppg_signal
#     resized_data[1, i, :] = accel_signal
#
#     print("Resized data shape:", resized_data.shape)

    return PPG_data
# # Access the 'sig' variable from the loaded data
# sig_data = mat_data['sig']
#
# # Extract the desired column, e.g., the first column (index 0)
# column_to_plot = sig_data[0, :800]
#
# # Plot the selected column
# plt.plot(column_to_plot, label='Selected Column')
# plt.title('Plot of Variable sig (Selected Column)')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()
original_data = mat_prcoessing()

