import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from Preprocess_Data import preprocess_data
from mat_prcoessing import mat_prcoessing

# Define the BIDNET model
def create_bidnet_model(input_shape,x_train):
    model = models.Sequential()

    # Convolutional layers
    #model.add(tf.keras.Input(shape=(x_train.shape[1],)))
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv1D(30, kernel_size=7, input_shape=(x_train.shape), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))

    for _ in range(4):

        model.add(layers.Conv1D(60, kernel_size=7, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.1))

    # Bidirectional LSTM layers
    model.add(layers.Bidirectional(layers.LSTM(64, activation='tanh', return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, activation='tanh')))

    # Dense output layer for unsupervised learning
    model.add(layers.Dense(1))
    model.add(layers.Softmax())
    return model



def train_model(X_train):

    # Train the model
    history = model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.2)

    plot_training_history(history)
    # Save the trained model
    model.save('model.keras')
    print("Training completed.")



# sequence_length = 227622  # Adjust this based on your actual sequence length
#
# X_train = load_and_preprocess_data('/Users/drake/OneDrive/Documents/PPG Research/DATABASE/Training_data/DATA_01_TYPE01.mat', sequence_length)





#takes in data from matlab file to process
# Assuming resized_data has shape (6, 37937)
# We want to reshape it to (2, 500)

# Reshape the data
#resized_data = original_data[2:4, :500]  # Take the first 500 samples from each channel
#print("Resized data shape:", resized_data.shape)

# Transpose the data to have shape (2, 500)
#resized_data = resized_data.T.reshape(2, 500)

original_data = mat_prcoessing()
preprocessed_data = preprocess_data(original_data)
# Define input shape based on required dimensions

input_shape = (2, 500)
# Create the BIDNET model
X_train = np.reshape(preprocessed_data,(2,500))
print("X_Train data shape:", X_train.shape)

model = create_bidnet_model(input_shape,X_train)

# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
print("X_train data shape:", preprocessed_data.shape)

X_train_2 = np.expand_dims(preprocessed_data,axis=0)
print("X_Train_2 data shape:", X_train_2.shape)
# Train the model
train_model(X_train)





