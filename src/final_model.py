# coding: utf-8

import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from mpi4py import MPI



# Input params #
stk_path = "../data/GSPC.csv"
# stk_path = "./data/SPY.csv"
# stk_path = "./data/GSPC.csv"
# train : cv : test = 6 : 2 : 2
test_size = 0.1  # proportion of dataset to be used as test set
cv_size = 0.05  # proportion of dataset to be used as cross-validation set
N = 30  # past days
start_time = time.time()


np.random.seed(10)


# parameters needed in MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def load_data():
    # # Load data
    df = pd.read_csv(stk_path, sep=",")
    # df.drop(['Close'], axis = 1)
    # Convert Date column to datetime
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df_date = df['Date']

    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

    # Get month of each sample
    df['month'] = df['date'].dt.month
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Sort by datetime
    df.sort_values(by='date', inplace=True, ascending=True)

    df = df.drop(["date"], axis=1)
    df = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

    df = pd.concat([df_date.rename('date'), df], axis=1)
    df.head(10)

    # # Split into train, dev and test set

    # Get sizes of each of the datasets
    num_cv = int(cv_size * len(df))        # train for meta-regressor
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

    return df, train, cv, train_cv, test, num_cv, num_test, num_train


def buildTrain(train, pastDay=30, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0] - futureDay - pastDay):
        X_train.append(np.array(train.iloc[i:i + pastDay]))
        Y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay]["adj_close"]))
    return np.array(X_train), np.array(Y_train)


def shuffle(X, Y):
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def lin_regression_model():

    def dim3_to_dim2(data):
        x, y, z = data.shape
        return data.reshape((x, y * z))

    df, train, cv, train_cv, test, num_cv, num_test, num_train = load_data()

    # # Predict using Linear Regression
    RMSE = []
    R2 = []
    mape = []

    train = train.drop(["date"], axis=1)
    cv = cv.drop(["date"], axis=1)
    test = test.drop(["date"], axis=1)
    X_train, y_train = buildTrain(train, N, 1)
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = buildTrain(cv, N, 1)
    X_val, y_val = shuffle(X_val, y_val)
    X_test, y_test = buildTrain(test, N, 1)
    X_test, y_test = shuffle(X_test, y_test)

    print("++++++++++++++++++++Lin_Reg++++++++++++++++++++++++")
    print(X_test)
    print(y_test)
    print("++++++++++++++++++++++++++++++++++++++++++++")

    model = LinearRegression(fit_intercept=True)
    X_train = dim3_to_dim2(X_train)
    X_val = dim3_to_dim2(X_val)
    X_test = dim3_to_dim2(X_test)
    model.fit(X_train, y_train)

    trainPredict = model.predict(X_train).flatten()
    valPredict = model.predict(X_val).flatten()
    testPredict = model.predict(X_test).flatten()

    train_size = trainPredict.size
    train_x = [i for i in range(train_size)]
    test_size = valPredict.size
    test_x = [i for i in range(test_size)]


    y_val = y_val.flatten()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    plt.figure(1)
    plt.plot(test_x, y_val, label="y_val")
    plt.plot(test_x, valPredict, label="predict")
    plt.xlabel("index")  # row axios
    plt.ylabel("Adj Closing price")  # column axios
    plt.legend(loc="best")  # sample
    plt.show()

    valPredict = pd.DataFrame(valPredict, columns=['est_lin_reg'])
    testPredict = pd.DataFrame(testPredict, columns=['est_lin_reg'])
    y_val = pd.DataFrame(y_val, columns=['adj_close'])
    y_test = pd.DataFrame(y_test, columns=['adj_close'])

    return model, valPredict, testPredict, y_val, y_test


def lstm_model():

    def buildManyToOneModel(shape):
        model = Sequential()
        model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
        # output shape: (1, 1)
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        return model

    df, train, cv, train_cv, test, num_cv, num_test, num_train = load_data()

    train = train.drop(["date"], axis=1)
    cv = cv.drop(["date"], axis=1)
    test = test.drop(["date"], axis=1)
    # change the last day and next day
    X_train, y_train = buildTrain(train, N, 1)
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = buildTrain(cv, N, 1)
    X_val, y_val = shuffle(X_val, y_val)
    X_test, y_test = buildTrain(test, N, 1)
    X_test, y_test = shuffle(X_test, y_test)

    print("++++++++++++++++++++LSTM++++++++++++++++++++++++")
    print(X_test)
    print(y_test)
    print("++++++++++++++++++++++++++++++++++++++++++++")

    model = buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_val, y_val), callbacks=[callback])
    # model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_val, y_val))

    y_val = y_val.flatten()
    y_train = y_train.flatten()

    trainPredict = model.predict(X_train).flatten()
    valPredict = model.predict(X_val).flatten()
    testPredict = model.predict(X_test).flatten()

    train_size = trainPredict.size
    train_x = [i for i in range(train_size)]
    test_size = valPredict.size
    test_x = [i for i in range(test_size)]

    plt.figure(1)
    plt.plot(test_x, y_val, label="y_val")
    plt.plot(test_x, valPredict, label="lstm_predict")
    plt.xlabel("index")  # row axios
    plt.ylabel("Closing price")  # column axios
    plt.legend(loc="best")  # sample
    plt.show()

    valPredict = pd.DataFrame(valPredict, columns=['est_lstm'])
    testPredict = pd.DataFrame(testPredict, columns=['est_lstm'])
    y_val = pd.DataFrame(y_val, columns=['adj_close'])
    y_test = pd.DataFrame(y_test, columns=['adj_close'])

    return model, valPredict, testPredict


'''
    Meta-Regressor: merge two model together

    example: ensemble_model(df_lin, df_lstm, y_val, 'est_lin_reg', 'est', 'adj_close', y_target_col)
    will return the model that has been trained
'''


def ensemble_model(model1_val,
                   model1_test,
                   model2_val,
                   model2_test,
                   y_val,
                   y_test,
                   model1_target_col,
                   model2_target_col,
                   y_target_col):

    # # Common functions
    def make_walkforward_model(X, y):
        model = LassoCV(positive=True)
        X_train = pd.concat([X[model1_target_col], X[model2_target_col]], axis=1)
        y_train = y.loc[:len(y)]
        model.fit(X_train, y_train)
        return model

    def predict(model, X):
        X_test = pd.concat([X[model1_target_col], X[model2_target_col]], axis=1)
        return model.predict(X_test).flatten()

    def prepare_Xy(X_raw, y_raw):
        ''' Utility function to drop any samples without both valid X and y values'''
        Xy = X_raw.join(y_raw).replace({np.inf: None, -np.inf: None}).dropna()
        X = Xy.iloc[:, :-1]
        y = Xy.iloc[:, -1]
        return X, y

    def get_model_predict(df1, df2, y_data):
        y = y_data[y_target_col]
        X = pd.concat([df1[model1_target_col], df2[model2_target_col]], axis=1)
        print(X)
        return X, y

    models_X_val, y_val = get_model_predict(model1_val, model2_val, y_val)
    X_ens, y_ens = prepare_Xy(models_X_val, y_val)
    ensemble_models = make_walkforward_model(X_ens, y_ens)

    models_X_test, y_test = get_model_predict(model1_test, model2_test, y_test)
    models_X_test, y_test = prepare_Xy(models_X_test, y_test)
    testPredict = predict(ensemble_models, models_X_test)
    testPredict = pd.DataFrame(testPredict, columns=['est_ensem'])

    return ensemble_models, testPredict


if rank == 0:

    # load data from the file and start a timer
    start_time = time.time()
    data = []
    print("\n*********************************************\n                    Recv\n*********************************************\n")

    # size = comm.recv(source=1, tag=1)
    # lin_reg_val_predict = np.empty(size, dtype='float64')
    lin_reg_val_predict = comm.recv(source=1, tag=2)
    lin_reg_val_predict = pd.DataFrame(lin_reg_val_predict, columns = ["est_lin_reg"])
    # print(
    #     "\n*********************************************\n                    Recv\n*********************************************\n")
    # print("lin_reg mark1 received")
    lin_reg_test_predict = comm.recv(source=1, tag=3)
    lin_reg_test_predict = pd.DataFrame(lin_reg_test_predict, columns = ["est_lin_reg"])
    # print(
    #     "\n*********************************************\n                    Recv\n*********************************************\n")
    # print("lin_reg mark2 received")
    y_val = comm.recv(source=1, tag=4)
    y_val = pd.DataFrame(y_val, columns = ["adj_close"])
    # print(
    #     "\n*********************************************\n                    Recv\n*********************************************\n")
    # print("lin_reg mark3 received")
    y_test = comm.recv(source=1, tag=5)
    y_test = pd.DataFrame(y_test, columns = ["adj_close"])
    # print(
    #     "\n*********************************************\n                    Recv\n*********************************************\n")
    # print("lin_reg mark4 received")
    lstm_val_predict = comm.recv(source=2, tag=6)
    lstm_val_predict = pd.DataFrame(lstm_val_predict, columns = ["est_lstm"])
    # print(
    #     "\n*********************************************\n                    Recv\n*********************************************\n")
    # print("lstm mark1 received")
    lstm_test_predict = comm.recv(source=2, tag=7)
    lstm_test_predict = pd.DataFrame(lstm_test_predict, columns = ["est_lstm"])
    # print(
    #     "\n*********************************************\n                    Recv\n*********************************************\n")
    # print("lstm mark12 received")
    ensemble_models, ensem_test_predict = ensemble_model(lin_reg_val_predict, lin_reg_test_predict, lstm_val_predict,
                                                         lstm_test_predict, y_val, y_test, 'est_lin_reg', 'est_lstm',
                                                            'adj_close')

    test_size = ensem_test_predict.size
    test_x = [i for i in range(test_size)]
    plt.figure(1)
    plt.plot(test_x, y_test, label="y_test")
    plt.plot(test_x, lin_reg_test_predict, label="lin_predict")
    plt.plot(test_x, lstm_test_predict, label="lstm_predict")
    plt.plot(test_x, ensem_test_predict, label="ensem_predict")
    plt.xlabel("index")  # row axios
    plt.ylabel("Closing price")  # column axios
    plt.legend(loc="best")  # sample
    # plt.show()

    RMSE = []
    R2 = []
    MAPE = []
    SMAPE = []

    for model_predict in (lin_reg_test_predict, lstm_test_predict, ensem_test_predict):
        RMSE.append(math.sqrt(mean_squared_error(y_test, model_predict)))
        R2.append(r2_score(y_test, model_predict))
        MAPE.append(get_mape(y_test, model_predict))
        SMAPE.append(get_smape(y_test, model_predict))

    print("RMSE" + str(RMSE))
    print("R2" + str(R2))
    print("MAPE" + str(MAPE))
    print("SMAPE" + str(SMAPE))

    end_time = time.time()
    print("time elapsed: ", end_time - start_time)



elif rank == 1:
    # time.sleep(50)
    lin_reg_model, lin_reg_val_predict, lin_reg_test_predict, y_val, y_test = lin_regression_model()

    lin_reg_val_predict = lin_reg_val_predict.values.flatten()
    lin_reg_val_predict = np.array(lin_reg_val_predict)
    comm.send(lin_reg_val_predict, dest=0, tag=2)
    print("lin_reg mark1")

    lin_reg_test_predict = lin_reg_test_predict.values.flatten()
    lin_reg_test_predict = np.array(lin_reg_test_predict)
    comm.send(lin_reg_test_predict, dest=0, tag=3)
    print("lin_reg mark2")

    y_val = y_val.values.flatten()
    y_val = np.array(y_val)
    comm.send(y_val, dest=0, tag=4)
    print("lin_reg mark3")

    y_test = y_test.values.flatten()
    y_test = np.array(y_test)
    comm.send(y_test, dest=0, tag=5)
    print("lin_reg mark4")


elif rank == 2:
    lstm_model, lstm_val_predict, lsmt_test_predict = lstm_model()
    # print("*********************************************\nLSTM model*********************************************\n",
    #       lstm_val_predict)
    # print("\n*********************************************\nLSTM model*********************************************\n")

    lstm_val_predict = lstm_val_predict.values.flatten()
    lstm_val_predict = np.array(lstm_val_predict)
    comm.send(lstm_val_predict, dest=0, tag=6)
    print("lstm mark1")

    lsmt_test_predict = lsmt_test_predict.values.flatten()
    lsmt_test_predict = np.array(lsmt_test_predict)
    comm.send(lsmt_test_predict, dest=0, tag=7)
    print("lstm mark2")