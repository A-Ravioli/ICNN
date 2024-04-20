import numpy as np
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model


# Define a traditional neural network
def build_traditional_nn(input_shape, num_layers, num_neurons):
    # Define input layer
    input_layer = Input(shape=input_shape)
    prev_layer = input_layer

    # Build the neural network
    for _ in range(num_layers):
        # Dense layer
        prev_layer = Dense(num_neurons, activation="relu")(prev_layer)

    # Output layer
    output_layer = Dense(1, activation="sigmoid")(prev_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# EX: Build the traditional neural network model
# traditional_model = build_traditional_nn(input_shape=(100,), num_layers=5, num_neurons=64)

# Compile the traditional model
# traditional_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the traditional model
# traditional_history = traditional_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)
