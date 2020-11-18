# CS596FinalProject
Final project repo of CSCI 596 project

## Background

Financial markets are mostly random. However, they are not completely random. There are many small inefficiencies and patterns in the market, which can be identified and used to gain a weak advantage in the market.

These advantages are rarely large enough to trade alone, and transaction costs and overheads can easily cover our revenue. However, when we can combine many of these small advantages, the benefits may be huge!

## Possible algorithms 
### Stacked Generalization

#### Basic process: 
  1. We train many different models with different training algorithms, hyperparameter and characteristics to predict
  2. Train a meta-regressor to combine and utilize the prediction results of all models to form a unified and stable result
  
#### Advantage:
  1. By weakening those models that seem to overfit the data, the model has greater generalization capabilities for out-of-sample (that is, invisible) data. This is achieved by allowing the meta-model to learn which base model predictions perform well (or poorly) outside of the sample, and weighting the models appropriately.
  2. Stack generalization is very suitable for the challenges faced when forecasting in noisy, non-stationary, and unstable financial markets, and helps solve the problem of overfitting
  3. SG is very flexible for all kinds of machine learning ideas for it can utilize different training models as base models. What matters is to create the meta regressor to utilize the advantages and avoid the disadvantages of each trained model.

<div align="center" style="text-align: 'center'">
  <img src="https://img.chainnews.com/material/images/665f1c1e2f04e3408b294292c7b88cd6.jpg-article#pic_center" width="80%"/>
</div>
