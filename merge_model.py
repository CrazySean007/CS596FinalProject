from sklearn.linear_model import LassoCV
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

'''
    Meta-Regressor: merge two model together
    
    example: ensemble_model(df_lin, df_lstm, 'est_N5', 'est', 'adj_close')
    will return the model that has been trained
'''


def ensemble_model(model1_predict, model2_predict, model1_target_col, model2_target_col, y_target_col):

    # # Common functions
    def make_walkforward_model(X, y):
        model = LassoCV(positive=True)
        X_train = pd.concat([X[model1_target_col], X[model2_target_col]], axis=1)
        y_train = y.loc[:len(y)]
        model.fit(X_train, y_train)
        return model

    def prepare_Xy(X_raw,y_raw):
        ''' Utility function to drop any samples without both valid X and y values'''
        Xy = X_raw.join(y_raw).replace({np.inf:None,-np.inf:None}).dropna()
        X = Xy.iloc[:,:-1]
        y = Xy.iloc[:,-1]
        return X,y

    def get_model_predict(df1, df2):
        y = df1[y_target_col]
        X = pd.concat([df1['date'], df1[model1_target_col], df2[model2_target_col]], axis=1)
        return X, y

    models_predict, real_price = get_model_predict(model1_predict, model2_predict)
    X_ens, y_ens = prepare_Xy(models_predict, real_price)

    ensemble_models = make_walkforward_model(X_ens, y_ens)

    return ensemble_models
