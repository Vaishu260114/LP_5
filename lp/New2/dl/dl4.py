import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout

# Importing the training set
data_train = pd.read_csv("Google_Stock_Price_Train.csv")
train = data_train.loc[:, ["Open"]].values

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

# Create a data structure with timesteps
timesteps = 50  # Changed from 5 to 50 (more typical for stock prediction)
x_train = []
y_train = []
for i in range(timesteps, len(train_scaled)):
    x_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape input to be 3D [samples, timesteps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create RNN Model
regressor = Sequential()

# First RNN layer
regressor.add(SimpleRNN(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Second RNN layer
regressor.add(SimpleRNN(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Third RNN layer
regressor.add(SimpleRNN(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Fourth RNN layer
regressor.add(SimpleRNN(units=50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units=1))

# Compile the RNN
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
regressor.fit(x_train, y_train, epochs=100, batch_size=32)  # Increased epochs, larger batch size

# Getting the real stock price of test set
data_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = data_test.loc[:, ["Open"]].values

# Prepare test inputs
data_total = pd.concat((data_train["Open"], data_test["Open"]), axis=0)
inputs = data_total[len(data_total) - len(data_test) - timesteps:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

# Create test data structure
x_test = []
for i in range(timesteps, timesteps + len(data_test)):
    x_test.append(inputs[i-timesteps:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()