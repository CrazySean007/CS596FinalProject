# coding: utf-8

import math
import matplotlib
import numpy as np
import pandas as pd

from datetime import date, datetime, time, timedelta
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Input params #
stk_path = "../data/GOOG.csv"
test_size = 0.2  # proportion of dataset to be used as test set
cv_size = 0.2  # proportion of dataset to be used as cross-validation set
Nmax = 7  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
N = 5
# Nmax is the maximum N we are going to test

start_time = datetime.timestamp()

# # Common functions
def get_preds_lin_reg(df, target_col, N, pred_min, offset):
    """
    Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe with the values you want to predict. Can be of any length.
        target_col : name of the column you want to predict e.g. 'adj_close'
        N          : get prediction at timestep t using values from t-1, t-2, ..., t-N
        pred_min   : all predictions should be >= pred_min
        offset     : for df we only do predictions for df[offset:]. e.g. offset can be size of training set
    Outputs
        pred_list  : the predictions for target_col. np.array of length len(df)-offset.
    """
    # Create linear regression object
    regression = LinearRegression(fit_intercept=True)

    pred_list = []

    for i in range(offset, len(df[target_col])):
        X_train = np.array(range(len(df[target_col][i - N:i])))  # e.g. [0 1 2 3 4]
        y_train = np.array(df[target_col][i - N:i])  # e.g. [2944 3088 3226 3335 3436]
        X_train = X_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        regression.fit(X_train, y_train)  # Train the model
        pred = regression.predict(np.array(N).reshape(-1, 1))

        pred_list.append(pred[0][0])  # Predict the footfall using the model

    # If the values are < pred_min, set it to be pred_min
    pred_list = np.array(pred_list)

    return pred_list


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # Load data

df = pd.read_csv(stk_path, sep=",")

# Convert Date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df_date = df['Date']

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Get month of each sample
df['month'] = df['date'].dt.month

# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

df = df.drop(["date"], axis=1)
df = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

df = pd.concat([df_date.rename('date'), df], axis=1)
df.head(10)

# # Split into train, dev and test set

# Get sizes of each of the datasets
num_cv = int(cv_size * len(df))
num_test = int(test_size * len(df))
num_train = len(df) - num_cv - num_test
print("num_train = " + str(num_train))
print("num_cv = " + str(num_cv))
print("num_test = " + str(num_test))

# Split into train, cv, and test
train = df[:num_train]
cv = df[num_train:num_train + num_cv]
train_cv = df[:num_train + num_cv]
test = df[num_train + num_cv:]
print("train.shape = " + str(train.shape))
print("cv.shape = " + str(cv.shape))
print("train_cv.shape = " + str(train_cv.shape))
print("test.shape = " + str(test.shape))


# # Predict using Linear Regression

RMSE = []
R2 = []
mape = []

est_list = get_preds_lin_reg(train_cv, 'adj_close', N, 0, num_train)

cv['est' + '_N' + str(N)] = est_list
RMSE.append(math.sqrt(mean_squared_error(est_list, cv['adj_close'])))
R2.append(r2_score(cv['adj_close'], est_list))
mape.append(get_mape(cv['adj_close'], est_list))

print('RMSE = ' + str(RMSE))
print('R2 = ' + str(R2))
print('MAPE = ' + str(mape))
cv.head()

# Set optimum N
N_opt = 5

# # Plot predictions on dev set

# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8  # width 10, height 8

ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
ax = cv.plot(x='date', y='est_N5', style='m-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'predictions with N=5'])
ax.set_xlabel("date")
ax.set_ylabel("USD")



#Final Model

#
# est_list = get_preds_lin_reg(df, 'adj_close', N_opt, 0, num_train+num_cv)
# test['est' + '_N' + str(N_opt)] = est_list
# print("RMSE = %0.3f" % math.sqrt(mean_squared_error(est_list, test['adj_close'])))
# print("R2 = %0.3f" % r2_score(test['adj_close'], est_list))
# print("MAPE = %0.3f%%" % get_mape(test['adj_close'], est_list))
# test.head()
#
# # Plot adjusted close over time, only for test set
# rcParams['figure.figsize'] = 10, 8 # width 10, height 8
# matplotlib.rcParams.update({'font.size': 14})
#
# ax = test.plot(x='date', y='adj_close', style='gx-', grid=True)
# ax = test.plot(x='date', y='est_N5', style='rx-', grid=True, ax=ax)
# ax.legend(['test', 'predictions using linear regression'], loc='upper left')
# ax.set_xlabel("date")
# ax.set_ylabel("USD")
plt.show()

# Save as csv
test_lin_reg = cv

test_lin_reg.to_csv("./test_lin_reg.csv")


end_time = datetime.timestamp();

print(end_time - start_time)
# # Findings
# * On the dev set, the lowest RMSE is 1.2 which is achieved using N=1, ie. using value on day t-1 to predict value on day t
# * On the dev set, the next lowest RMSE is 1.36 which is achieved using N=5, ie. using values from days t-5 to t-1 to predict value on day t
# * We will use N_opt=5 in this work since our aim here is to use linear regression
# * On the test set, the RMSE is 1.42 and MAPE is 0.707% using N_opt=5
