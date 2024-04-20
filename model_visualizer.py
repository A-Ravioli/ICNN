import tensorflow as tf


def convert_model_to_graph(model, input_shape=(100,)):
    # Create a TensorFlow tensor for the input
    input_tensor = tf.keras.layers.Input(shape=(None,) + input_shape)
    output_tensor = model(input_tensor)  # Forward pass through the model

    # Save the graph
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


# EX: Convert the model to a graph and save it for visualization
# convert_model_to_graph(model)
