# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:42:44 2021

@author: ASHUTOSH
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('18-04-2019-TO-16-04-2021RELIANCEALLN.csv', header=0, index_col=2)
dataset=dataset.drop(columns=['Series','Symbol'])
print(dataset.head())
# dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year
# dataset['Month'] = pd.DatetimeIndex(dataset['Date']).month
# dataset['Week'] = pd.DatetimeIndex(dataset['Date']).week
# dataset['Day'] = pd.DatetimeIndex(dataset['Date']).day
# dataset['Dayofweek'] = pd.DatetimeIndex(dataset['Date']).day_name()
# dataset['Dayofyear'] = pd.DatetimeIndex(dataset['Date']).dayofyear
# dataset['Is_month_end'] = pd.DatetimeIndex(dataset['Date']).is_month_end
# dataset['Is_month_start'] = pd.DatetimeIndex(dataset['Date']).is_month_start
# dataset['Is_quarter_end'] = pd.DatetimeIndex(dataset['Date']).is_quarter_end
# dataset['Is_quarter_start'] = pd.DatetimeIndex(dataset['Date']).is_quarter_start
# dataset['Is_year_end'] = pd.DatetimeIndex(dataset['Date']).is_year_end
# dataset['Is_year_start'] = pd.DatetimeIndex(dataset['Date']).is_year_start
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,11] = encoder.fit_transform(values[:,11])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_days = 10
n_features = 12
# frame as supervised learning
reframed = series_to_supervised(scaled, n_days, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
# values1=values[:400,:]
# values2=values[400:,:]
# print(values[:400,:])
# # n_train_hours = 365 * 24
train = values[:395, :]
test = values[395:, :]
# split into input and outputs
n_obs = n_days * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -11:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
print('prediction=')
print(inv_yhat)
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -11:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
print('actual=')
print(inv_y)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)