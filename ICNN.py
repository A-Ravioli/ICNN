from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

# Generate sample data for binary classification
np.random.seed(0)
X_train = np.random.randn(1000, 100)
y_train = np.random.randint(2, size=(1000, 1))

# Variables
input_shape = (100,)
num_layers = 50
num_neurons = 64


# Define a function to build the skip-connected neural network
def build_skip_connected_nn(input_shape, num_layers, num_neurons):
    # Define input layer
    input_layer = Input(shape=input_shape)
    prev_layer = input_layer

    # Initialize list to hold skip connections
    skip_connections = []
    # Build the neural network with skip connections
    for i in range(num_layers):
        # Dense layer
        dense_layer = Dense(num_neurons, activation="relu")(prev_layer)

        # Add skip connection from previous layers
        if len(skip_connections) > 0:
            dense_layer = Add()([dense_layer] + skip_connections)

        # Append the current layer to the skip connections list
        skip_connections.append(dense_layer)

        # Update the previous layer
        prev_layer = dense_layer

    # Output layer
    output_layer = Dense(1, activation="sigmoid")(prev_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# EX: Build the skip-connected neural network model
# model = build_skip_connected_nn(input_shape, num_layers, num_neurons)
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# model.summary()
