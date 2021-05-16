# import packages
import pandas as pd
import numpy as np

# to plot within notebook
import matplotlib.pyplot as plt
# %matplotlib inline

# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# read the file
df = pd.read_csv('18-04-2019-TO-16-04-2021RELIANCEALLN.csv')

# print the head
df.head()

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%d-%b-%Y')
df.index = df['Date']

# plot
plt.figure(figsize=(16, 8))
plt.plot(df['Close Price'], label='Close Price history')

print('\n Shape of the data:')
print(df.shape)
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close Price'])
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close Price'][i] = data['Close Price'][i]

# NOTE: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.

# splitting into train and validation
train = new_data[:395]
valid = new_data[395:]

# shapes of training set
print('\n Shape of training set:')
print(train.shape)

# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)

# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0, valid.shape[0]):
    a = train['Close Price'][len(train) - 248 + i:].sum() + sum(preds)
    b = a / 248
    preds.append(b)

# checking the results (RMSE value)
rms = np.sqrt(np.mean(np.power((np.array(valid['Close Price']) - preds), 2)))
print('\n RMSE value on validation set:')
print(rms)

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Close Price'])
plt.plot(valid[['Close Price', 'Predictions']])

# sorting
data = df.sort_index(ascending=True, axis=0)

# creating a separate dataset
new_data = pd.DataFrame(index=range(0, len(df)),
                        columns=['Date', 'Close Price', 'Turnover', 'No. of Trades', '% Dly Qt to Traded Qty'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close Price'][i] = data['Close Price'][i]
    new_data['Turnover'][i] = data['Turnover'][i]
    new_data['No. of Trades'][i] = data['No. of Trades'][i]
    new_data['% Dly Qt to Traded Qty'][i] = data['% Dly Qt to Traded Qty'][i]

new_data['Year'] = pd.DatetimeIndex(new_data['Date']).year
new_data['Month'] = pd.DatetimeIndex(new_data['Date']).month
new_data['Week'] = pd.DatetimeIndex(new_data['Date']).week
new_data['Day'] = pd.DatetimeIndex(new_data['Date']).day
new_data['Dayofweek'] = pd.DatetimeIndex(new_data['Date']).day_name()
new_data['Dayofyear'] = pd.DatetimeIndex(new_data['Date']).dayofyear
new_data['Is_month_end'] = pd.DatetimeIndex(new_data['Date']).is_month_end
new_data['Is_month_start'] = pd.DatetimeIndex(new_data['Date']).is_month_start
new_data['Is_quarter_end'] = pd.DatetimeIndex(new_data['Date']).is_quarter_end
new_data['Is_quarter_start'] = pd.DatetimeIndex(new_data['Date']).is_quarter_start
new_data['Is_year_end'] = pd.DatetimeIndex(new_data['Date']).is_year_end
new_data['Is_year_start'] = pd.DatetimeIndex(new_data['Date']).is_year_start
new_data.head()

switcher = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

for i in range(0, len(new_data)):
    prev = new_data['Dayofweek'][i]
    new_data['Dayofweek'][i] = switcher.get(new_data['Dayofweek'][i], "Invalid day")
#     if(new_data['Dayofweek'][i]=="Invalid day"):
#         print(prev)
print(new_data.head())

new_data['mon_fri'] = 0
for i in range(0, len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0
print(new_data.head())

# split into train and validation
train = new_data[:395]
valid = new_data[395:]

x_train = train.drop(['Date', 'Close Price'], axis=1)
y_train = train['Close Price']
x_valid = valid.drop(['Date', 'Close Price'], axis=1)
y_valid = valid['Close Price']

# implement linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

# make predictions and find the rmse
preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
rms

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[395:].index
train.index = new_data[:395].index

plt.plot(train['Close Price'])
plt.plot(valid[['Close Price', 'Predictions']])

# importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

# using gridsearch to find the best parameter
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# fit the model and make predictions
model.fit(x_train, y_train)
preds = model.predict(x_valid)

# rmse
rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
rms

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Close Price', 'Predictions']])
plt.plot(train['Close Price'])

from pmdarima import auto_arima
data = df.sort_index(ascending=True, axis=0)

train = data[:395]
valid = data[395:]

training = train['Close Price']
validation = valid['Close Price']

model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)

forecast = model.predict(n_periods=99)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

rms=np.sqrt(np.mean(np.power((np.array(valid['Close Price'])-np.array(forecast['Prediction'])),2)))
rms

#plot
plt.plot(train['Close Price'])
plt.plot(valid['Close Price'])
plt.plot(forecast['Prediction'])

#importing prophet
from fbprophet import Prophet

#creating dataframe
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close Price'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close Price'][i] = data['Close Price'][i]

new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
new_data.index = new_data['Date']

#preparing data
new_data.rename(columns={'Close Price': 'y', 'Date': 'ds'}, inplace=True)

#train and validation
train = new_data[:395]
valid = new_data[395:]

#fit the model
model = Prophet()
model.fit(train)

#predictions
close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)
#rmse
forecast_valid = forecast['yhat'][395:]
rms=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])

#plot
valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close Price'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close Price'][i] = data['Close Price'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:395,:]
valid = dataset[395:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms

#for plotting
train = new_data[:395]
valid = new_data[395:]
valid['Predictions'] = closing_price
plt.plot(train['Close Price'])
plt.plot(valid[['Close Price','Predictions']])

