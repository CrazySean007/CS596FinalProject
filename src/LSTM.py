import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))


def readTrain():
    train = pd.read_csv("../data/GSPC.csv", engine='python')
    train.tail()
    return train


def augFeatures(train):
    train["Date"] = pd.to_datetime(train["Date"])
    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month
    train["date"] = train["Date"].dt.day
    train["day"] = train["Date"].dt.dayofweek
    print(train.keys())
    print(train)
    return train


def normalize(train):
    train = train.drop(["Date"], axis=1)
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    # train_norm = scaler.fit_transform(train)
    # train_norm = pd.DataFrame(train_norm ,columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'year', 'month', 'date',
    #      'day'])
    return train_norm


def buildTrain(train, pastDay=30, futureDay=1):
    X_train, Y_train = [], []
    # print("shape0: ", train.shape[0])
    for i in range(train.shape[0] - futureDay - pastDay):
        X_train.append(np.array(train.iloc[i:i + pastDay]))
        Y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay]["Adj Close"]))
    return np.array(X_train), np.array(Y_train)


def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def splitData(X, Y, rate):
    X_train = X[int(X.shape[0] * rate):]
    Y_train = Y[int(Y.shape[0] * rate):]
    X_val = X[:int(X.shape[0] * rate)]
    Y_val = Y[:int(Y.shape[0] * rate)]
    return X_train, Y_train, X_val, Y_val


def divideAndNormalizeData(train, rate):
    train = train.drop(["Date"], axis=1)
    trainData = train[int(train.shape[0] * rate):]
    testData = train[:int(train.shape[0] * rate)]
    return trainData, testData


def buildOneToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    # output shape: (1, 1)
    model.add(TimeDistributed(Dense(1)))  # or use model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model


def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model


start_time = time.time()

train = readTrain()
train_Aug = augFeatures(train)
train_Aug.drop(["Date"], axis=1)
train_norm = normalize(train_Aug)
# change the last day and next day
X_train, Y_train = buildTrain(train_norm, 30, 1)
# X_train, Y_train = shuffle(X_train, Y_train)
# because no return sequence, Y_train and Y_val shape must be 2 dimension
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
# model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val))

end_time = time.time()
print("time elapsed: ", end_time - start_time)

# predictModel = train_norm
# Y_train = Y_train.flatten()

# trainPredict = model.predict(X_train).flatten()
testPredict = model.predict(X_val).flatten()
Y_val = Y_val.flatten()

# train_size = trainPredict.size
# train_x = [i for i in range(train_size)]

test_size = testPredict.size
test_x = [i for i in range(test_size)]

plt.figure(1)
# plt.plot(train_x, Y_train, label="trainData")
# plt.plot(train_x, trainPredict, label="testData")
plt.plot(test_x, Y_val, label="actual Price")
plt.plot(test_x, testPredict, label="Predict Price")
plt.xlabel("index")  # 横坐标名字
plt.ylabel("Closing price")  # 纵坐标名字
plt.legend(loc="best")  # 图例
plt.show()
