# CS596FinalProject
Final project repo of CSCI 596 project

## Background

**Financial markets are mostly random**. However, they are not completely random. There are many small inefficiencies and patterns in the market, which can be identified and used to gain a weak advantage in the market.

These advantages are rarely large enough to trade alone, and transaction costs and overheads can easily cover our revenue. However, when we can combine many of these small advantages so that **we may avoid many risks and make benefits**.

<img src="https://pic1.zhimg.com/v2-46987ab76079ac483abcd35bde8aa277_1440w.jpg?source=172ae18b" width="40%"/>

## Structure 
### Stacked Generalization

#### Basic process: 
  1. We train many **different models with different training algorithms**, hyperparameter and characteristics to predict
  2. **Train a meta-regressor** to combine and utilize the prediction results of all models to form a unified and stable result

#### Advantage:
  1. By **weakening those models that seem to overfit the data**, the model has greater generalization capabilities for out-of-sample (that is, invisible) data. This is achieved by allowing the meta-model to learn which base model predictions perform well (or poorly) outside of the sample, and weighting the models appropriately.
  2. Stack generalization is very **suitable** for the challenges faced when forecasting in **noisy, non-stationary, and unstable financial markets**, and helps solve the problem of overfitting
  3. SG is **very flexible** for all kinds of machine learning ideas for it can utilize different training models as base models. What matters is to create the meta regressor to utilize the advantages and avoid the disadvantages of each trained model.

<div align="center">
  <img src="https://img.chainnews.com/material/images/665f1c1e2f04e3408b294292c7b88cd6.jpg-article#pic_center" width="80%"/>
</div>

## Base Algorithm

### Linear Regression

a common model for predicting stock markets, which is building a function to calculate the result by using different features.

![image029](http://people.duke.edu/~rnau/regintro_files/image029.png)

### LSTM
Long short-term memory (Long short-term memory, LSTM) is a special RNN, it mainly solves the problem of gradient vanishing and gradient explosion in the training process of long sequences. LSTM can perform better in longer sequences than ordinary RNNs.
<div align="center">
  <img src="https://pic4.zhimg.com/80/v2-e4f9851cad426dfe4ab1c76209546827_1440w.jpg#pic_center" width="80%"/>
</div>

### More Algorithms

Because of the flexibility of the SG model, we can find other algorithms in the future, to enhance this model and make the result more accurate.



## Parallel Computing

### mpi4py

Mpi4py is a module that helps to run MPI in python environments.

This module helps us train and run base algorithms in parallel.

![MPI usage](images/mpi.jpg)

The document can be found in https://mpi4py.readthedocs.io/en/stable/index.html .

### Cuda Python

To run CUDA Python, we need to use CUDA Toolkit, which can be found in https://developer.nvidia.com/how-to-cuda-python%20.


### CuDNN

The cuDNN is a library developed by NVIDA, which combines CUDA with Deep Neural Network Library. (https://developer.nvidia.com/cudnn) 

  #### CuDNNLSTM

  In this project, we may use CuDNNLSTM, the cuDNN library specifically for LSTM algorithm as we use in the base step.






