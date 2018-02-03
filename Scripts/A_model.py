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
    # columns with lots of NaNs
    indiv_a_train.drop('OdXpbPGJ', axis = 1, inplace = True)
    indiv_a_test.drop('OdXpbPGJ', axis = 1, inplace = True)

    # these features have overlapping distributions. improved CV just a little bit
    hhold_a_train.drop(['YFMZwKrU',
	# 'nEsgxvAq', # added 1_17
	'OMtioXZZ'], axis = 1, inplace = True)
    hhold_a_test.drop(['YFMZwKrU',
	# 'nEsgxvAq', # added 1_17 . removed again 1_17. See if it helps to have it in there, while dropping all categoricals in B
	'OMtioXZZ'], axis = 1, inplace = True)

    # cat_columns = hhold_a_train.select_dtypes(include = ['object']).columns
    # cat_to_keep = ['zFkComtB','DxLvCGgv', 'YTdCRVJt', 'QyBloWXZ', 'ZRrposmO', 'ggNglVqE', 'JwtIxvKg', 'bMudmjzJ', 'LjvKYNON', 'HHAeIHna', 'CrfscGZl', 'ZnBLVaqz', 'pCgBHqsR', 'wEbmsuJO', 'IZFarbPw', 'GhJKwVWC', 'qgxmqJKa', 'xkUFKUoW', 'phwExnuQ', 'ptEAnCSs', 'kLkPtNnh', 'DbUNVFwv', 'PWShFLnY', 'uRFXnNKV', 'UXhTXbuS', 'vRIvQXtC']
    # cat_to_keep.append('country')
    # cat_to_drop = list(set(cat_to_keep)^set(cat_columns))
    # hhold_a_train.drop(cat_to_drop, axis = 1, inplace = True)
    # hhold_a_test.drop(cat_to_drop, axis = 1, inplace = True)

    #### end drop columns #####


    # make training sets
    X_train = hhold_a_train.drop(['poor','country'], axis = 1) # if you want to resample, need to leave the poor column in here so that minority/majority class can be calculated for resampling
    y_train = hhold_a_train['poor'].values
    indiv_X_train = indiv_a_train.drop(['poor','country'], axis = 1)

    # make test sets
    X_test = hhold_a_test.drop('country', axis = 1)
    indiv_X_test = indiv_a_test.drop('country', axis = 1)

    # store cat columns and numerical columns for later use
    cat_columns = X_train.select_dtypes(include = ['object']).columns
    # num_columns = X_train.select_dtypes(include = ['int64', 'float64']).columns

    # make new features from the individual sets
    # number of individuals
    X_train = num_indiv(X_train, indiv_X_train)
    X_test = num_indiv(X_test, indiv_X_test)

    # label encode individual train/test set
    indiv_X_train, indiv_cat_columns = labelencode_cat(indiv_X_train)
    indiv_X_test, indiv_cat_columns = labelencode_cat(indiv_X_test)

    ## standardizing remaining columns
    # standardize only the numerical columns
    num_columns = ['TiwRslOh', 'num_indiv']
    X_train[num_columns] = standardize(X_train[num_columns])
    X_test[num_columns] = standardize(X_test[num_columns])

    # label encode remaining cat columns. Don't want to redo what was encoded in individual set already
    # X_train[cat_columns] = X_train[cat_columns].apply(LabelEncoder().fit_transform)
    # X_test[cat_columns] = X_test[cat_columns].apply(LabelEncoder().fit_transform)

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

    ### end features
    params = {'n_estimators':100, 'max_depth':5, 'reg_alpha':0.5, 'reg_lambda': 0.5,
              'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5, 'random_state' : 144,'eval_metric' : 'logloss', 'verbose': 2}

    clf = xgb.XGBClassifier(**params)
    # clf = lgb.LGBMClassifier(n_estimators = 50, objective = 'binary', num_threads = 4, learning_rate = 0.05)

    # clf = SVC(probability = True, random_state = 2)
    # fit
    clf.fit(X_train, y_train)

    # predict
    preds = clf.predict_proba(X_test)

    return preds

if __name__ == '__main__':
    run_a_model()
