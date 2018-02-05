import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb

def run_a_model():
    hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
    hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

    indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()
    # need to load indiv test sets here to make the new feats for the test set for submission
    indiv_a_test, indiv_b_test, indiv_c_test = load_indiv_test()

    ####--- Drop columns that we won't need at all ####
    # drop columns
    # columns with lots of NaNs
    indiv_a_train.drop('OdXpbPGJ', axis = 1, inplace = True)
    indiv_a_test.drop('OdXpbPGJ', axis = 1, inplace = True)

    # these features have overlapping distributions. improved CV just a little bit
    hhold_a_train.drop(['YFMZwKrU', 'OMtioXZZ'], axis = 1, inplace = True)
    hhold_a_test.drop(['YFMZwKrU', 'OMtioXZZ'], axis = 1, inplace = True)

    #### end drop columns #####

    # make training sets
    X_train = hhold_a_train.drop(['poor','country'], axis = 1) # if you want to resample, need to leave the poor column in here so that minority/majority class can be calculated for resampling
    y_train = hhold_a_train['poor'].values
    indiv_X_train = indiv_a_train.drop(['poor','country'], axis = 1)

    # make test sets
    X_test = hhold_a_test.drop('country', axis = 1)
    indiv_X_test = indiv_a_test.drop('country', axis = 1)

    # make new features from the individual sets
    # number of individuals
    X_train = num_indiv(X_train, indiv_X_train)
    X_test = num_indiv(X_test, indiv_X_test)


    ## standardizing remaining columns
    # standardize only the numerical columns
    num_columns = ['TiwRslOh', 'num_indiv']
    X_train[num_columns] = standardize(X_train[num_columns])
    X_test[num_columns] = standardize(X_test[num_columns])

    # one hot encoding
    # concatenate train and test to do the one hot encoding. Train and test don't have the same categorical values so one hot encoding gives different number of features on the different sets
    X_train['nEsgxvAq'] = X_train['nEsgxvAq'].astype('str')
    X_test['nEsgxvAq'] = X_test['nEsgxvAq'].astype('str')
    print(X_train.head(1))
    tmp = pd.concat((X_train, X_test))
    tmp = pd.get_dummies(tmp)
    print(tmp.shape)
    X_train = tmp.iloc[:X_train.shape[0]]
    print(X_train.head(1))
    X_test = tmp.iloc[X_train.shape[0]:]

    # # mean target encoding
    # cat_columns = X_train.select_dtypes(include = ['object', 'category'])
    # for col in cat_columns:
        # train_means = X_train.groupby(col).poor.mean()
        # X_train[col] = X_train[col].map(train_means)
        # X_test[col] = X_test[col].map(train_means)

    # X_train.drop('poor', axis = 1, inplace = True)
    # print(X_train.columns.values)

    ### end features
    params = {'n_estimators':100, 'max_depth':5, 'reg_alpha':0.5, 'reg_lambda': 0.5,
              'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5, 'random_state' : 144,'eval_metric' : 'logloss', 'verbose': 2}

    clf = xgb.XGBClassifier(**params)

    # fit
    clf.fit(X_train, y_train)

    # predict
    preds = clf.predict_proba(X_test)

    return preds

if __name__ == '__main__':
    run_a_model()
