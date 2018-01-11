"""
The target variables for countries B and C are highly unbalanced, but for A it is a little more balanced.
- use Stratified K Fold to create CV that takes account of the unbalance
- use different metrics to measure model performance
    - confusion matrix
    - Precision/Recall/ROC Curves

Treat each country seperately since each country will use a different model.




"""
import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np
from PoverTHelperTools import *

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def resample_data(df, how = 'down'):
    """
    Upsample or Downsample data to balance classes for CV
    
    """
    df_majority = df[df['poor'] == 0]
    df_minority = df[df['poor'] == 1]

    if how == 'down':
        print('Downsampling data...')
        df_majority_downsampled = resample(df_majority,
                                           replace = False, # sample without replacement
                                           n_samples = df_minority.shape[0], # to match minority class
                                           random_state = 144
                                           )
        df_resampled = pd.concat([df_majority_downsampled, df_minority])
    elif how == 'up':
        print('Upsampling data...')
        df_minority_upsampled = resample(df_minority,
                                          replace = True, #sample with replacement
                                          n_samples = df_majority.shape[0], # to match majority class amount
                                          random_state = 144)
        df_resampled = pd.concat([df_majority, df_minority_upsampled])

    return df_resampled

def run_CV(X,y,model,func, n_splits = 3, how = 'up', categorical = 'label_encoder'):
    """
    Proper CV for unbalanced target

    
    X: training set with poor still in the column. Drop country
    y: target (`poor`)
    model: classifier to use
    func: feature engineering function i.e. feature_eng_a, etc
    how: upsample or downsample
    categorical: how to treat categorical features
    
    Returns: a list of logloss values of each fold
    """
    logloss = []
    skf = StratifiedKFold(n_splits = n_splits, random_state = 144)
    for i, (train_idx, val_idx) in enumerate(skf.split(X,y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # # SMOTE
        # X_train = X_train.drop('poor', axis = 1) # drop target
        # cat_columns = X_train.select_dtypes(['object']).columns
        # X_train[cat_columns] = X_train[cat_columns].apply(LabelEncoder().fit_transform)
        # orig_cols = X_train.columns # SMOTE will return a numpy array. Store the column names here to recreate the dataframe for feature engineering/transforms below
        # X_train, y_train = SMOTE().fit_sample(X_train,y_train)
        # # recreate dataframe
        # X_train = pd.DataFrame(X_train, columns = orig_cols)

        if how:
            # resample to balance data
            X_resampled = resample_data(X_train, how = how)
            # store the targets now that they are balanced
            y_train = X_resampled['poor']
            # drop target from train
            X_train = X_resampled.drop('poor', axis = 1)
        ####### feature engineering goes blow this comment:
       
        func(X_train)
       
        ###### end feature eng
        X_train = pre_process_data(X_train, normalize_num='standardize', categorical = categorical)
        assert X_train.shape[0] == y_train.shape[0]

        model.fit(X_train, y_train)
        # standardize X_val to predict
        X_val = pre_process_data(X_val,normalize_num= 'standardize', enforce_cols=X_train.columns, categorical = categorical)
        preds = model.predict_proba(X_val)
        
        logloss.append(log_loss(y_val, preds[:,1]))
    
    return logloss

def mean_log_loss(a_logloss, b_logloss, c_logloss):
    """
    a_logloss, b_logloss, c_logloss: lists of logloss from different folds

    Returns: scalar. mean log loss value

    """
    # hard code weights because size of test set isn't going to change and this way don't need to load test sets just to get shapes
    a_weight = 4041/8832 # hhold_a_test.shape[0]/total number of test samples 
    b_weight = 1604/8832
    c_weight = 3187/8832
    return np.average([np.average(a_logloss), np.average(b_logloss), np.average(c_logloss)], weights = [a_weight, b_weight, c_weight])


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
