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