import math
import pandas as pd

from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from xgboost import XGBRegressor


#### Input params ##################
stk_path = "../SPY.csv"
test_size = 0.2                # proportion of dataset to be used as test set
cv_size = 0.2                  # proportion of dataset to be used as cross-validation set
N = 7                          # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

n_estimators = 100             # for the initial model before tuning. default = 100
max_depth = 3                  # for the initial model before tuning. default = 3
learning_rate = 0.1            # for the initial model before tuning. default = 0.1
min_child_weight = 1           # for the initial model before tuning. default = 1
subsample = 1                  # for the initial model before tuning. default = 1
colsample_bytree = 1           # for the initial model before tuning. default = 1
colsample_bylevel = 1          # for the initial model before tuning. default = 1
train_test_split_seed = 111    # 111
model_seed = 100

fontsize = 14
ticklabelsize = 14
####################################


# def readTrain():
#   train = pd.read_csv("data.txt", engine='python')
#   train.tail()
#   # print (train)
#   return train


# def augFeatures(train):
#   train["Date"] = pd.to_datetime(train["Date"])
#   train["year"] = train["Date"].dt.year
#   train["month"] = train["Date"].dt.month
#   train["date"] = train["Date"].dt.day
#   train["day"] = train["Date"].dt.dayofweek
#   return train

def readDF():
	df = pd.read_csv(stk_path, sep = ",")
	# Convert Date column to datetime
	df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
	# Change all column headings to be lower case, and remove spacing
	df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
	# Get month of each sample
	df['month'] = df['date'].dt.month
	# Sort by datetime
	df.sort_values(by='date', inplace=True, ascending=True)
	df.head()

def divideDataset():
	# Get sizes of each of the datasets
	num_cv = int(cv_size*len(df))
	num_test = int(test_size*len(df))
	num_train = len(df) - num_cv - num_test

	# Split into train, cv, and test
	train = df[:num_train]
	cv = df[num_train:num_train+num_cv]
	train_cv = df[:num_train+num_cv]
	test = df[num_train+num_cv:]

	return train, cv, train_cv, test

def convertToDF(data):
	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])

	# Convert the numpy array back into pandas dataframe
	data_scaled = pd.DataFrame(data_scaled, columns=['open', 'high', 'low', 'close', 'volume'])
	data_scaled[['date', 'month']] = data[['date', 'month']]
	data_scaled.head()

	return data_scaled

readDF()
train, cv, train_cv, test = divideDataset()
train_scaled = convertToDF(train)
cv_scaled = convertToDF(cv)
train_cv_scaled = convertToDF(train_cv)
test_scaled = convertToDF(test)

# Combine back train_scaled, cv_scaled, test_scaled together
df_scaled = pd.concat([train_scaled, cv_scaled, test_scaled], axis=0)
df_scaled.head()

# Get difference between high and low of each day
df_scaled['range_hl'] = df_scaled['high'] - df_scaled['low']
df_scaled.drop(['high', 'low'], axis=1, inplace=True)

# Get difference between open and close of each day
df_scaled['range_oc'] = df_scaled['open'] - df_scaled['close']

df_scaled.head()

# Add a column 'order_day' to indicate the order of the rows by date
df_scaled['order_day'] = [x for x in list(range(len(df_scaled)))]

# merging_keys
merging_keys = ['order_day']

# List of columns that we will use to create lags
lag_cols = ['open', 'close', 'range_hl', 'range_oc', 'volume']


shift_range = [x+1 for x in range(N)]

for shift in tqdm_notebook(shift_range):
    train_shift = df_scaled[merging_keys + lag_cols].copy()

    # E.g. order_day of 0 becomes 1, for shift = 1.
    # So when this is merged with order_day of 1 in df_scaled, this will represent lag of 1.
    train_shift['order_day'] = train_shift['order_day'] + shift

    foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
    train_shift = train_shift.rename(columns=foo)

    df_scaled = pd.merge(df_scaled, train_shift, on=merging_keys, how='left') #.fillna(0)

del train_shift


# Remove the first N rows which contain NaNs
df_scaled = df_scaled[N:]
    
df_scaled.head()

df_scaled.info()

features = [
    "open_lag_1",
    "close_lag_1",
    "range_hl_lag_1",
    "range_oc_lag_1",
    "volume_lag_1",
    "open_lag_2",
    "close_lag_2",
    "range_hl_lag_2",
    "range_oc_lag_2",
    "volume_lag_2",
    "open_lag_3",
    "close_lag_3",
    "range_hl_lag_3",
    "range_oc_lag_3",
    "volume_lag_3",
    "open_lag_4",
    "close_lag_4",
    "range_hl_lag_4",
    "range_oc_lag_4",
    "volume_lag_4",
    "open_lag_5",
    "close_lag_5",
    "range_hl_lag_5",
    "range_oc_lag_5",
    "volume_lag_5",
    "open_lag_6",
    "close_lag_6",
    "range_hl_lag_6",
    "range_oc_lag_6",
    "volume_lag_6",
    "open_lag_7",
    "close_lag_7",
    "range_hl_lag_7",
    "range_oc_lag_7",
    "volume_lag_7"
]

target = "close"

# Split into train, cv, and test
train = df_scaled[:num_train]
cv = df_scaled[num_train:num_train+num_cv]
train_cv = df_scaled[:num_train+num_cv]
test = df_scaled[num_train+num_cv:]

# Split into X and y
X_train = train[features]
y_train = train[target]
X_cv = cv[features]
y_cv = cv[target]
X_train_cv = train_cv[features]
y_train_cv = train_cv[target]
X_sample = test[features]
y_sample = test[target]

# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8 # width 10, height 8

ax = train.plot(x='date', y='close', style='b-', grid=True)
ax = cv.plot(x='date', y='close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='close', style='g-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test'])
ax.set_xlabel("date")
ax.set_ylabel("USD (scaled)")


# # Train the model using XGBoost


# Create the model
model = XGBRegressor(seed=model_seed,
                      n_estimators=n_estimators,
                      max_depth=max_depth,
                      learning_rate=learning_rate,
                      min_child_weight=min_child_weight)

# Train the regressor
model.fit(X_train, y_train)


# In[19]:


# Do prediction on train set
est = model.predict(X_train)

# Calculate RMSE
math.sqrt(mean_squared_error(y_train, est))


# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8 # width 10, height 8

est_df = pd.DataFrame({'est': est, 
                       'date': train['date']})

ax = train.plot(x='date', y='close', style='b-', grid=True)
ax = cv.plot(x='date', y='close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'est'])
ax.set_xlabel("date")
ax.set_ylabel("USD (scaled)")


# Do prediction on test set
est = model.predict(X_cv)

# Calculate RMSE
math.sqrt(mean_squared_error(y_cv, est))


# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8 # width 10, height 8

est_df = pd.DataFrame({'est': est,
                       'y_cv': y_cv,
                       'date': cv['date']})

ax = train.plot(x='date', y='close', style='b-', grid=True)
ax = cv.plot(x='date', y='close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'est'])
ax.set_xlabel("date")
ax.set_ylabel("USD (scaled)")
plt.show()



