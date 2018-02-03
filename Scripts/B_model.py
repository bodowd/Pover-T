# V1 - RF.  LB: 0.27340 xgbA rfB rfC
import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def run_b_model():
    hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
    hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

    indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()
    # need to load indiv test sets here to make the new feats for the test set for submission
    indiv_a_test, indiv_b_test, indiv_c_test = load_indiv_test()

    #### Drop columns that we won't need at all ######
    # columns with lots of NaNs
    hhold_b_train.drop(['FGWqGkmD', 'BXOWgPgL', 'umkFMfvA', 'McFBIGsm', 'IrxBnWxE', 'BRzuVmyf', 'dnlnKrAg', 'aAufyreG', 'OSmfjCbE'], axis = 1, inplace=True)
    hhold_b_test.drop(['FGWqGkmD', 'BXOWgPgL', 'umkFMfvA', 'McFBIGsm', 'IrxBnWxE', 'BRzuVmyf', 'dnlnKrAg', 'aAufyreG', 'OSmfjCbE'], axis = 1, inplace=True)

    # drop columns with only 1 unique value
    hhold_b_train.drop(['ZehDbxxy', 'qNlGOBmo', 'izDpdZxF', 'dsUYhgai'], axis = 1, inplace = True)
    hhold_b_test.drop(['ZehDbxxy', 'qNlGOBmo', 'izDpdZxF', 'dsUYhgai'], axis = 1, inplace = True)
    # no seperation between classes
    hhold_b_train.drop(['qrOrXLPM','NjDdhqIe', 'rCVqiShm', 'ldnyeZwD',
           'BEyCyEUG', 'VyHofjLM', 'GrLBZowF', 'oszSdLhD',
           'NBWkerdL','vuQrLzvK','cDhZjxaW', # added 1_17
           'IOMvIGQS'], axis = 1, inplace = True)
    hhold_b_test.drop(['qrOrXLPM','NjDdhqIe', 'rCVqiShm', 'ldnyeZwD',
           'BEyCyEUG', 'VyHofjLM', 'GrLBZowF', 'oszSdLhD',
           'NBWkerdL','vuQrLzvK','cDhZjxaW', # added 1_17
           'IOMvIGQS'], axis = 1, inplace = True)

    # correlated features
    hhold_b_train.drop(['ZvEApWrk'], axis = 1, inplace = True)
    hhold_b_test.drop(['ZvEApWrk'], axis = 1, inplace = True)

    # lots of NaNs
    indiv_b_train.drop(['BoxViLPz', 'qlLzyqpP', 'unRAgFtX', 'TJGiunYp', 'WmKLEUcd', 'DYgxQeEi', 'jfsTwowc', 'MGfpfHam', 'esHWAAyG', 'DtcKwIEv', 'ETgxnJOM', 'TZDgOhYY', 'sWElQwuC', 'jzBRbsEG', 'CLTXEwmz', 'WqEZQuJP', 'DSttkpSI', 'sIiSADFG', 'uDmhgsaQ', 'hdDTwJhQ', 'AJgudnHB', 'iZhWxnWa', 'fyfDnyQk', 'nxAFXxLQ', 'mAeaImix', 'HZqPmvkr', 'tzYvQeOb', 'NfpXxGQk'], axis = 1, inplace = True)

    indiv_b_test.drop(['BoxViLPz', 'qlLzyqpP', 'unRAgFtX', 'TJGiunYp', 'WmKLEUcd', 'DYgxQeEi', 'jfsTwowc', 'MGfpfHam', 'esHWAAyG', 'DtcKwIEv', 'ETgxnJOM', 'TZDgOhYY', 'sWElQwuC', 'jzBRbsEG', 'CLTXEwmz', 'WqEZQuJP', 'DSttkpSI', 'sIiSADFG', 'uDmhgsaQ', 'hdDTwJhQ', 'AJgudnHB', 'iZhWxnWa', 'fyfDnyQk', 'nxAFXxLQ', 'mAeaImix', 'HZqPmvkr', 'tzYvQeOb', 'NfpXxGQk'], axis = 1, inplace = True)

    # need to rename because there are same column names in hhold and indiv
    indiv_b_train['wJthinfa_2'] = indiv_b_train['wJthinfa']
    indiv_b_train.drop('wJthinfa', axis = 1, inplace = True)

    indiv_b_test['wJthinfa_2'] = indiv_b_test['wJthinfa']
    indiv_b_test.drop('wJthinfa', axis = 1, inplace = True)


    print('Dropping all categoricals')

    cat_columns = list(hhold_b_train.select_dtypes(include = ['object']).columns)
    cat_columns.remove('country') # keep country. It gets selected by line above
    hhold_b_train.drop(cat_columns, axis = 1, inplace = True)
    hhold_b_test.drop(cat_columns, axis = 1, inplace = True)

    print(hhold_b_train.shape)
    #### end drop columns #####

    ### make training sets
    X_train = hhold_b_train.drop(['poor','country'], axis = 1) # if you want to resample, need to leave the poor column in here so that minority/majority class can be calculated for resampling
    y_train = hhold_b_train['poor'].values


    indiv_X_train = indiv_b_train.drop(['poor','country'], axis = 1)

    # make test sets
    X_test = hhold_b_test.drop('country', axis = 1)
    indiv_X_test = indiv_b_test.drop('country', axis = 1)

    # store cat columns and numerical columns for later use
    # cat_columns = X_train.select_dtypes(include = ['object']).columns
    # num_columns = X_train.select_dtypes(include = ['int64', 'float64']).columns

    # make new features from the individual sets
    # number of individuals
    X_train = num_indiv(X_train, indiv_X_train)
    X_test = num_indiv(X_test, indiv_X_test)

    # label encode individual train/test set
    # indiv_X_train, indiv_cat_columns = labelencode_cat(indiv_X_train)
    # indiv_X_test, indiv_cat_columns = labelencode_cat(indiv_X_test)

    ## standardizing remaining columns
    # standardize only the numerical columns
    num_columns = ['num_indiv']

    X_train[num_columns] = standardize(X_train[num_columns])
    X_test[num_columns] = standardize(X_test[num_columns])

    # concatenate train and test to do the one hot encoding. Train and test don't have the same categorical values so one hot encoding gives different number of features on the different sets
    new_cats = list(set(X_train.columns.values) - set(num_columns))
    print(new_cats)
    X_train[new_cats] = X_train[new_cats].astype('str')
    X_test[new_cats] = X_test[new_cats].astype('str')
    print(X_train.head(1))
    tmp = pd.concat((X_train, X_test))
    tmp = pd.get_dummies(tmp)
    print(tmp.shape)
    X_train = tmp.iloc[:X_train.shape[0]]
    print(X_train.head(1))
    X_test = tmp.iloc[X_train.shape[0]:]

    # label encode remaining cat columns. Don't want to redo what was encoded in individual set already
    # X_train[cat_columns] = X_train[cat_columns].apply(LabelEncoder().fit_transform)
    # X_test[cat_columns] = X_test[cat_columns].apply(LabelEncoder().fit_transform)

    ### end features

    # params = {'n_estimators':400, 'max_depth':5, 'reg_alpha':0.5, 'reg_lambda': 0.5,
	   # 'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5, 'random_state' = 144,
	      # 'eval_metric' : 'logloss', 'verbose': 2
	   # }
    params = {'n_estimators':400, 'max_depth':3, 'reg_alpha':0, 'reg_lambda':1, 'min_child_weight': 5} # xgb params
    clf = xgb.XGBClassifier(**params)
    # clf = lgb.LGBMClassifier(n_estimators = 50, objective = 'binary', num_threads = 4, learning_rate = 0.05)
    # clf = SVC(class_weight = 'balanced', probability = True, random_state = 2)
    # fit
    clf.fit(X_train, y_train)

    # predict
    preds = clf.predict_proba(X_test)

    return preds
if __name__ == '__main__':
    run_b_model()

