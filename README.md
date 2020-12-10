# CS596FinalProject

Final project repo of CSCI 596 project



## Background

Financial markets are mostly random. However, they are not completely random. There are many small inefficiencies and patterns in the market, which can be identified and used to gain a weak advantage in the market.These advantages are rarely large enough to trade alone, and transaction costs and overheads can easily cover our revenue. However, when we can combine many of these small advantages so that we may avoid many risks and make benefits.


## Set up

1. Install tensorflow@2.0

   `$ pip install tensorflow`
   
2.  Install sklearn, numpy, pandas, matplotlib
3. Install mpi4py(For windows users, please download [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) and set environment path)	

   `$ pip install mpi4py`
   
4. It’s also recommended to install anaconda to prepare for the environment

## Run

### Serial Version

`$ python final_model_serial.py`

### Parallel Version

`$ mpiexec -n 3 python final_model.py`



## Data

The data we used is S&P 500 Index（from Dec.1969 to Dec. 2020) , named "GSPC.csv" in the /data folder.

The data is downloaded at Yahoo Finance. (https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC)

 We used S&P 500 Index because the index data is more general and stable, which is not easily affected by individual purpose.



## Structure

Basic Algorithms and Stack Generalization

### Basic process

1. We train many different models with different training algorithms, hyperparameter and characteristics to predict
2. Train a meta-regressor to combine and utilize the prediction results of all models to form a unified and stable result

### Advantages

1. By weakening those models that seem to overfit the data, the model has greater generalization capabilities for out-of-sample (that is, invisible) data. This is achieved by allowing the meta-model to learn which base model predictions perform well (or poorly) outside of the sample, and weighting the models appropriately.
2. Stack generalization is very suitable for the challenges faced when forecasting in noisy, non-stationary, and unstable financial markets, and helps solve the problem of overfitting
3. SG is very flexible for all kinds of machine learning ideas for it can utilize different training models as base models. What matters is to create the meta regressor to utilize the advantages and avoid the disadvantages of each trained model.

![img](https://lh6.googleusercontent.com/VXs_RsvPrZgBHNVilABhn9lhGHcoysV6RwgKxTR5Bg2PH0XvFRQHCPOr5iais5ZO-p0rH1fFxv04ysaalpbSHr-sgB6jGwusSWNekldVJ76x1v39HxWt_6oYRNXkZv01tGh2Fg7s)



### Basic Algorithm

#### Linear Regression

a common model for predicting stock markets, which is building a function to calculate the result by using different features.

![image029](https://lh3.googleusercontent.com/UZHIRX-iE14PX-C_9XEKh5bP59BMvNRXZBXsq4zm97_pJNyHOwqc4mfwLtWJxyhjTWYBJKTtsE2QAAps2SeTb41Is9Ter0-W1vFxnJ6u9G42VGnxD49FzbWtLoruuoCbgYDdNQcV)

#### LSTM

Long short-term memory (Long short-term memory, LSTM) is a special RNN, it mainly solves the problem of gradient vanishing and gradient explosion in the training process of long sequences. LSTM can perform better in longer sequences than ordinary RNNs.

![img](https://lh6.googleusercontent.com/t6uEKzGXjwCJsC4d5VnqL1xJlLuiN3ygNidsXog8afrzFLcGKUkt2pUmNk_vb4rTVG28Ytx37aOAJszR-rY3drs-e71THGt-ehHigMhl0IMX3Cwig5Q9W1hj8qO5SnkgNRu6DZAS)

### MetaRegressor

#### Process:

After getting results of different base algorithms, we can ensemble them with different weights. By training the weights with data, the final model will be our results.The training process can also be helped by the basic algorithms above. However, in this step, we decided to complete it by **LassoCV**, which is a python module automatically running meta regression.



## Project Workflow

1. Load the data file and process the data.

   a. Training data for Linear_regression and LSTM models 

   b. Training data for meta_regressor

   c. Test data to evaluate all the models

 2. Build training data by setting past 30 days data as input and current Close price as output

 3. Shuffle the data to remove timeline influences

 4. Send data (a) to train linear and LSTM model

 5. Utilize data (b) to train the meta regression model

 6. Input data (c) to the meta regressor to make predictions

 7. Evaluate the model using predictions and actual values



### Serial Version

![img](https://lh4.googleusercontent.com/fMQ0FU5DflQ7dFop-5htMQbdN-teOPdisa83MFvWcVr-9as8wFuRfC_Lpk_vrOYgwL_knKrWh9FMxD4WNj7P5VuGPgqSEgb-MqYwN7JnhuDT_oTPiLdIxyckNxEciEbafiPnBRmJ)

### Parallel Version (Using MPI):

![img](https://lh6.googleusercontent.com/GKOqZA_lc0eQlQZ9y4EUPeIZo8roX7Yqj0xBH1cA2pxQJo-MfQVkQafTlGxfZPm2Q1hREPjapoTDIYI1GmXjzHBzpPcq_vAVffY0yibAZNcy2iY-WTdl5uRk1_Z2AWETR2BMSIe6)

### Shffuling

The reason for Shuffle is to prevent the order of data input from affecting network training. Increase the randomness, improve the generalization performance of the network, avoid the occurrence of regular data, which causes the gradient of the weight update to be too extreme, and avoid over-fitting or under-fitting of the final model.



As it is given in the picture below, the training data is overfitted in Linear Regression, because of the consecutive timeline.

![img](https://lh3.googleusercontent.com/o6CoNa8b20fb4nOxUtGcJbZJTMwKqtaD--ba9l5c2o28ywWliJWiXaF2yPVoYksZH-1PD3NOI623lpNK0hfAN_8AzaybnH6JujsNU7yynjhB789oyR-g4cbf7afj4sjsWcTMuIWz)

## Result

### Final Model

The comparasion of different models for prediction.

(The fluctuation is because of data shuffling)

![img](https://lh5.googleusercontent.com/1LcNiJ2PZj5QxJR9zSzC19faXp-oBw7OWe4J4GI6E88A1HNcfzXR-cA3OzKH0Bf5D0ko1kWOoxqF_05RmbIH4y2BJ39NeKD-zFWOw6yOuL307ChQVbPwMrsxtU5kTe1E_S-DC88a)



Zoom version of the graph, we can see the effect that the red line (ensembled model) fits better with blue line (original data) than orange line (Linear Regression Model) and green line (LSTM Model) do.

![img](https://lh5.googleusercontent.com/GCDoBQsSgRnQOL62iXjjp0McLnBMKVHLHHVpo5JxaLWrYKDGxYJupEGzhvvLvOVkX1D-tu938rzLy5HZ0IaAWMOaf0Eef3zH_Q2N8B-2GYsrH2_jTkyywTyxSocwAYroLPshTkZW)



## Conclusion

RMSE(Root Mean Square Error):	

<img src="https://lh4.googleusercontent.com/zenTjapjO4kOoMOe-uyybQulcDkmAwfks9rhW5MjlAyJgEjhqNU25zq1TV1w89XBSO4OOKVVYGyUlZt5GPm7vJRBs7XakstDANMQz0zYC_XnaZj3XXitfrtgyoxespWP5aDELaeD" alt="img" style="zoom:50%;" />

MAPE(Mean Absolute Percentage Error):

<img src="https://lh6.googleusercontent.com/5xrL6AULFHxn43aYhIoSuSGyURVetDiGsNem7Wtr44DGuz1iKrQfsDIZR1bMiN-eIeMOZ5jeNSR4JstbsSYip7ONLkQyAe3DNB32VkCvqoJ0XO5YDlB5yCtGPO_3x9YpgYJic4MM" alt="img" style="zoom:50%;" />



| Evaluation | Linear Regression | LSTM   | Final(Meta-Regressor) |
| ---------- | ----------------- | ------ | --------------------- |
| RMSE       | 0.1601            | 0.0135 | 0.0103                |
| MAPE (%)   | 28.05             | 2.089  | 1.407                 |





The new model has smaller RMSE and MSE values for the validation data, indicating that it has better generalization ability and prediction accuracy. We can conclude that the new model has the best effect, followed by LSTMmodel, and linear regression model has the worst effect.



|                  | Serial | Parallel (with MPI) |
| ---------------- | ------ | ------------------- |
| Running Time (s) | 90.80  | 176.66              |



Since the linear regression model is extremely fast compared with the LSTM model, we use “sleep(50)” in rank 1 to balance the running time. 

   Speedup: 176.66 / 98.8 = 1.95


   Efficiency: Speedup / Num_Ranks = 1.95 / 3 = 64.9%




## Future Work

In the future, we can add more machine learning models into the regression-model part to have better prediction and generalization power. Also, the linear model is too fast, which decreases the efficiency because for most of the time, rank 1 which runs linear regression stays idle. So in the future, we can better balance the base models to make the best use of the CPU resources. Most importantly, although CUDA is not used due to the lack of compatible GPUs, we can use CUDA for accelerating training the program, especially Neural Network models.

