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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def StratifiedKF(df, n_splits, model,categorical = 'label_encoder',enforce_cols = None, fillmean = None, verbose = False):
    """
    df: train dataframe with `poor` and `country` still in columns. will be dropped in this function, and `poor` will be used to define y in this func
    model: classifier with sklearn-like API 

    """
    avg_train_logloss = []
    avg_valid_logloss = []
    y_valid_preds = np.array([])

    X = df.drop(['poor', 'country'], axis = 1)
    y = df['poor'].values.astype(int)

    skf = StratifiedKFold(n_splits=n_splits, random_state = 25)
    for i, (train_index, val_index) in enumerate(skf.split(X,y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # pre process data here after the split to avoid information leak
        print('Preproc train')
        X_train = PoverTHelperTools.pre_process_data(X_train, enforce_cols = enforce_cols, categorical = categorical, fillmean = fillmean)
        print('Preproc valid')
        X_val = PoverTHelperTools.pre_process_data(X_val, enforce_cols = enforce_cols, categorical = categorical, fillmean = fillmean)
        
        # models here
        print('*****\nTraining model on Fold {}'.format(i))
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)
        # print(len(preds))
        # print(preds)
        valid_logloss = log_loss(y_val, preds[:,1])
        train_logloss = log_loss(y_train, model.predict_proba(X_train)[:,1])
        avg_train_logloss.append(train_logloss)
        avg_valid_logloss.append(valid_logloss)
        
        y_valid_preds = np.concatenate((y_valid_preds,preds[:,1]))

        if verbose:
            print('\nTrain Log Loss for Fold {}: {}'.format(i, train_logloss))
            print('Validation Log Loss for Fold {}: {}\n'.format(i, valid_logloss))

    print('******\nAverage Train Log Loss: {}'.format(np.mean(avg_train_logloss)))
    print('Average Validation Log Loss: {}\n'.format(np.mean(avg_valid_logloss)))
    
    return y, y_valid_preds

def mean_log_loss(y_true_a, y_pred_a,
                  y_true_b, y_pred_b,
                  y_true_c, y_pred_c):
    # average of each countries log loss
    a_logloss = log_loss(y_true_a, y_pred_a)
    b_logloss = log_loss(y_true_b, y_pred_b)
    c_logloss = log_loss(y_true_c, y_pred_c)
    # average of those 3 countries
    # return np.sum([a_logloss, b_logloss, c_logloss])/3
    print('Weighted average')
    return np.sum([a_logloss*len(y_true_a), b_logloss*len(y_true_b), c_logloss*len(y_true_c)])/(np.sum([len(y_true_a), len(y_true_b), len(y_true_c)]))

def print_confusion_matrix(confusion_matrix, class_names, figsize = (6,5), fontsize=14, ax = None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap. Copied from shaypal5/confusion_matrix_pretty_print.py
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    # fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", ax = ax)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # return fig
