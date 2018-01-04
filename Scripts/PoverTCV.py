"""
The target variables for countries B and C are highly unbalanced, but for A it is a little more balanced.
- use Stratified K Fold to create CV that takes account of the unbalance
- use different metrics to measure model performance
    - confusion matrix
    - Precision/Recall/ROC Curves

Treat each country seperately since each country will use a different model.




"""

import pandas as pd
import numpy as np
import PoverTHelperTools

# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import lightgbm as lgb


def StratifiedKF(df, n_splits, model):
    """
    df: train dataframe with `poor` and `country` still in columns. will be dropped in this function, and `poor` will be used to define y in this func
    model: classifier with sklearn-like API 

    """
    X = df.drop(['poor', 'country'], axis = 1)
    X = PoverTHelperTools.pre_process_data(X)
    y = df['poor'].values.astype(int)

    skf = StratifiedKFold(n_splits=n_splits, random_state = 25)
    for i, (train_index, val_index) in enumerate(skf.split(X,y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # models here
        print('*****\nTraining model on Fold {}'.format(i))
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)

        valid_logloss = log_loss(y_val, preds)
        train_logloss = log_loss(y_train, model.predict_proba(X_train))

        print('\nTrain Log Loss for Fold {}: {}'.format(i, train_logloss))
        print('Validation Log Loss for Fold {}: {}\n'.format(i, valid_logloss))


