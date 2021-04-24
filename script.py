#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
# %matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('18-04-2019-TO-16-04-2021RELIANCEALLN.csv')

#print the head
df.head()


#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%d-%b-%Y')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close Price'], label='Close Price history')

print('\n Shape of the data:')
print(df.shape)
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close Price'])
for i in range(0,len(data)):
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
for i in range(0,valid.shape[0]):
    a = train['Close Price'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Close Price'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Close Price'])
plt.plot(valid[['Close Price', 'Predictions']])

#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close Price','Turnover','No. of Trades','% Dly Qt to Traded Qty'])

for i in range(0,len(data)):
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

switcher={
    'Monday':0,
    'Tuesday':1,
    'Wednesday':2,
    'Thursday':3,
    'Friday':4,
    'Saturday':5,
    'Sunday':6
}

for i in range(0,len(new_data)):
    prev=new_data['Dayofweek'][i]
    new_data['Dayofweek'][i]=switcher.get(new_data['Dayofweek'][i],"Invalid day")
#     if(new_data['Dayofweek'][i]=="Invalid day"):
#         print(prev)
print(new_data.head())

new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0
print(new_data.head())

#split into train and validation
train = new_data[:395]
valid = new_data[395:]

x_train = train.drop(['Date','Close Price'], axis=1)
y_train = train['Close Price']
x_valid = valid.drop(['Date','Close Price'], axis=1)
y_valid = valid['Close Price']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[395:].index
train.index = new_data[:395].index

plt.plot(train['Close Price'])
plt.plot(valid[['Close Price', 'Predictions']])

#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Close Price', 'Predictions']])
plt.plot(train['Close Price'])