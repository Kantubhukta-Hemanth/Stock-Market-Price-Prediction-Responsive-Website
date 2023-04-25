#Import the modules
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import joblib

# Specify the start and end dates for the data
start_date = '2005-01-01'
end_date =  datetime.datetime.now().strftime('%Y-%m-%d')

# Use the DataReader function to import the data from Yahoo Finance
yf.pdr_override()
df = web.get_data_yahoo(['HDFC.NS'], start=start_date, end=end_date)

# Print the first 5 rows of the data
print(df.head())

#Reset index so that Date column can be used as an index
df = df.reset_index()

# Select 75% of the dataframe to be the training data
training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])

# Select the remaining 25% of the dataframe to be the testing data
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.75) : int(len(df))])

# Print the shape of the training data
print(training_data.shape)

# Print the shape of the testing data
print(testing_data.shape)

# Import the MinMaxScaler class from the scikit-learn library
from sklearn.preprocessing import MinMaxScaler

# Create an instance of the MinMaxScaler class, with a feature range of 0 to 1
scaler = MinMaxScaler(feature_range = (0, 1))

# Use the scaler to transform the training data into a scaled array of values
train_data_array = scaler.fit_transform(training_data)

# Create empty lists to store the input and output data for the training set
x_train = []
y_train = []

# Loop through the training data starting from index 100
for i in range(100, train_data_array.shape[0]):
    # Append the previous 100 values as the input data
    x_train.append(train_data_array[i-100 : i])
    # Append the next value as the output data
    y_train.append(train_data_array[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Print the shapes of x_train and y_train
print(x_train.shape, y_train.shape)

# Import necessary modules from Keras library
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model

# Define a Sequential model
model = Sequential()

# Add the first LSTM layer with 50 units, ReLU activation function, input shape, and return sequences
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))

# Add a Dropout layer with a 20% rate
model.add(Dropout(0.2))

# Add a second LSTM layer with 60 units, ReLU activation function, and return sequences
model.add(LSTM(units=60, activation='relu', return_sequences=True))

# Add a Dropout layer with a 30% rate
model.add(Dropout(0.3))

# Add a third LSTM layer with 80 units, ReLU activation function, and return sequences
model.add(LSTM(units=80, activation='relu', return_sequences=True))

# Add a Dropout layer with a 40% rate
model.add(Dropout(0.4))

# Add a fourth LSTM layer with 120 units and ReLU activation function
model.add(LSTM(units=120, activation='relu'))

# Add a Dropout layer with a 50% rate
model.add(Dropout(0.5))

# Add a Dense output layer with a single unit and ReLU activation function
model.add(Dense(units=1, activation='relu'))

# Print a summary of the neural network model
model.summary()

# Compile the model with the Adam optimizer and mean squared error loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model for 30 epochs using the training data
model.fit(x_train, y_train, epochs=30)

# Save the trained model to a file named 'HDFC.h5'
model.save('HDFC.h5')

# Load the saved model from the file named 'HDFC.h5' into a new variable named 'new'
new = load_model('HDFC.h5')

# Select the last 100 days of the training data
last_100_days = training_data.tail(100)

# Combine the last 100 days of the training data with the testing data
final_df = last_100_days.append(testing_data, ignore_index=True)

# Scale the input data using a scaler
input_data = scaler.fit_transform(final_df)

# Create x_test and y_test arrays for testing the model
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions on x_test using the trained model
y_predict = new.predict(x_test)

# Rescale the predicted and actual values to their original scale
scale_factor = 1/scaler.scale_
y_predict = y_predict * scale_factor
y_test = y_test * scale_factor

# Import the r2_score function from the sklearn.metrics module
from sklearn.metrics import r2_score

# Calculate the R-squared (R2) score between the test values (y_test) and the predicted values (y_predict)
r2 = r2_score(y_test, y_predict)

# Print the R-squared (R2) score
print("R-squared (R2):", r2)

# Import the mean_absolute_error function from the sklearn.metrics module
from sklearn.metrics import mean_absolute_error

# Calculate the Mean Absolute Error (MAE) between the test values (y_test) and the predicted values (y_predict)
mae = mean_absolute_error(y_test, y_predict)

# Print the MAE
print("MAE:", mae)