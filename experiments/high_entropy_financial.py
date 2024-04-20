from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn_builder import build_traditional_nn
from ICNN import build_skip_connected_nn
import matplotlib.pyplot as plt
import yfinance as yf

# Download stock data for a particular ticker symbol using yfinance
ticker_symbol = "AAPL"
stock_data = yf.download(ticker_symbol, start="2020-01-01", end="2021-01-01")

# Drop any rows with missing values
stock_data.dropna(inplace=True)

# Extract features and target (e.g., closing prices)
features = stock_data.drop(columns=["Close"]).values
target = stock_data["Close"].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Set input shape, number of layers, and neurons per layer for traditional NN
input_shape = X_train_scaled.shape[1]
num_layers = 5
num_neurons = 64

# Build the traditional neural network model
traditional_model = build_traditional_nn(input_shape, num_layers, num_neurons)

# Compile the traditional model
traditional_model.compile(optimizer="adam", loss="mean_squared_error")

# Train the traditional model
traditional_history = traditional_model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    verbose=0,
)

# Build the skip-connected neural network model
skip_connected_model = build_skip_connected_nn(input_shape, num_layers, num_neurons)

# Compile the skip-connected model
skip_connected_model.compile(optimizer="adam", loss="mean_squared_error")

# Train the skip-connected model
skip_connected_history = skip_connected_model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    verbose=0,
)

# Plot training and validation loss for both models
plt.plot(
    traditional_history.history["loss"],
    label="Traditional NN Training Loss",
    color="blue",
)
plt.plot(
    traditional_history.history["val_loss"],
    label="Traditional NN Validation Loss",
    linestyle="--",
    color="blue",
)

plt.plot(
    skip_connected_history.history["loss"],
    label="Skip-Connected NN Training Loss",
    color="orange",
)
plt.plot(
    skip_connected_history.history["val_loss"],
    label="Skip-Connected NN Validation Loss",
    linestyle="--",
    color="orange",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Comparison")
plt.legend()
plt.show()
