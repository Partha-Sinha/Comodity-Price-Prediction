# Used RNN(LSTM)

#Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('azd.csv')
training_set_price = dataset.iloc[:, 7:10].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(training_set_price[:, 0:3])
training_set_price[:, 0:3] = imputer.transform(training_set_price[:, 0:3])

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_price = sc.fit_transform(training_set_price)
 
# Getting the inputs and the outputs
X_train = training_set_price[0:3804]
y_train = training_set_price[1:3805]

# Reshaping
X_train = np.reshape(X_train, (3804, 1, 3))

# Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 3)))

# Adding the output Layer
regressor.add(Dense(units = 3))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Getting the predicited min price
input_min = dataset.iloc[3803:3804, 7:10].values
input_min = sc.transform(input_min)
input_min = np.reshape(input_min, (1, 1, 3))
predicted_price = regressor.predict(input_min)
predicted_price = sc.inverse_transform(predicted_price)
input_min = predicted_price
for x in range(29):
    input_min = sc.transform(input_min)
    input_min = np.reshape(input_min, (1, 1, 3))
    predicted_price_1 = regressor.predict(input_min)
    predicted_price_1= sc.inverse_transform(predicted_price_1)
    input_min = predicted_price_1
    predicted_price = np.concatenate((predicted_price, predicted_price_1), axis = 0)

np.savetxt('Predicted_Price.csv', predicted_price, fmt='%.2f', delimiter=',', header="MinPrice,  MaxPrice,  ModalPrice")



