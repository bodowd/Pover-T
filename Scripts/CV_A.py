import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, f1_score, recall_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

#########

def skfCV_A():
    # Load data
    print('Loading data...')
    hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
    hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

    indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()

    #########
    # drop columns
    # columns with lots of NaNs
    indiv_a_train.drop('OdXpbPGJ', axis = 1, inplace = True)

    # these features have overlapping distributions. improved CV just a little bit
    hhold_a_train.drop(['YFMZwKrU', 'OMtioXZZ'], axis = 1, inplace = True)

    #########
    # make training sets

    logloss = [] # append log loss scores to this array
    f1 = []
    recall = []

    X = hhold_a_train.drop('country', axis = 1)
    y = hhold_a_train['poor'].values
    indiv_X = indiv_a_train.drop(['poor','country'], axis = 1)

    # begin CV loop
    n_splits = 3
    skf = StratifiedKFold(n_splits = n_splits, random_state = 2, shuffle = True)
    for train_index, val_index in skf.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # get the samples in indiv set matching indexes of the hhold set in the fold
        indiv_X_train = indiv_X[indiv_X.index.get_level_values('id').isin(X_train.index.values)]
        indiv_X_val = indiv_X[indiv_X.index.get_level_values('id').isin(X_val.index.values)]
        # double check that the index values in the hhold data appear at least once in the individual index (indiv index has many duplicates of id because it is multi index)
        assert any(i in indiv_X_train.index.get_level_values('id').values for i in X_train.index.values)
        X_train.drop('poor', axis = 1, inplace = True)
        X_val.drop('poor', axis = 1, inplace = True)

        # make new features from indiv data set
        X_train = num_indiv(X_train, indiv_X_train)
        X_val = num_indiv(X_val, indiv_X_val)

        # standardize the numerical columns
        num_columns = ['TiwRslOh', 'num_indiv']
        X_train[num_columns] = standardize(X_train[num_columns])
        X_val[num_columns] = standardize(X_val[num_columns])

        # one hot encoding
        # concatenate train and test to do the one hot encoding. Train and test don't have the same categorical values so one hot encoding gives different number of features on the different sets
        X_train['nEsgxvAq'] = X_train['nEsgxvAq'].astype('str') # treat this feature as categorical
        X_val['nEsgxvAq'] = X_val['nEsgxvAq'].astype('str')
        print(X_train.head(1))
        tmp = pd.concat((X_train, X_val))
        tmp = pd.get_dummies(tmp)
        print(tmp.shape)
        X_train = tmp.iloc[:X_train.shape[0]]
        print(X_train.head(1))
        X_val = tmp.iloc[X_train.shape[0]:]

        # model
        params = {'n_estimators':100, 'max_depth':5, 'reg_alpha':0.5, 'reg_lambda': 0.5,
                  'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5, 'random_state' : 144,'eval_metric' : 'logloss', 'verbose': 2}

        clf = xgb.XGBClassifier(**params)

        clf.fit(X_train, y_train)

        preds = clf.predict_proba(X_val)

        # metrics
        logloss.append(log_loss(y_val, preds[:,1]))

        # convert probs to 0/1 for the following metrics
        preds_01 = (preds[:,1] > 0.5)
        f1.append(f1_score(y_val, preds_01))

        recall.append(recall_score(y_val, preds_01))

    print('Average logloss for A: ', np.average(logloss))
    print('log losses for each fold: ', logloss)
    print('average f1: ', np.average(f1))
    print('f1 for each fold: ', f1)
    print('average recall: ', np.average(recall))
    print('recall for each fold: ', recall)
    return logloss, f1, recall

if __name__ == '__main__':
    skfCV_A()
