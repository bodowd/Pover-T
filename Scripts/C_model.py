import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

def run_c_model():
    hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
    hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

    indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()
    # need to load indiv test sets here to make the new feats for the test set for submission
    indiv_a_test, indiv_b_test, indiv_c_test = load_indiv_test()

    #### Drop columns that we won't need at all #######
    # drop columns
    # drop columns with only one unique value
    hhold_c_train.drop(['GRGAYimk', 'DNnBfiSI', 'laWlBVrk', 'XAmOFyyg', 'gZWEypOM', 'kZmWbEDL', 'tTScFJYA', 'xyzchLjk', 'MtkqdQSs', 'enTUTSQi', 'kdkPWxwS', 'HNRJQbcm'], axis =1 , inplace = True)
    hhold_c_test.drop(['GRGAYimk', 'DNnBfiSI', 'laWlBVrk', 'XAmOFyyg', 'gZWEypOM', 'kZmWbEDL', 'tTScFJYA', 'xyzchLjk', 'MtkqdQSs', 'enTUTSQi', 'kdkPWxwS', 'HNRJQbcm'], axis =1 , inplace = True)
    # features with overlapping distributions
    # drop overlapping distributions
    hhold_c_train.drop(['LhUIIEHQ', 'PNAiwXUz', 'NONtAKOM', 'WWuPOkor', 'CtFxPQPT', 'qLDzvjiU', 'detlNNFh', 'tXjyOtiS', 'EQtGHLFz', 'cmjTMVrd', 'hJrMTBVd', 'IRMacrkM', 'EQSmcscG', 'aFKPYcDt', 'BBPluVrb', 'gAZloxqF', 'vSqQCatY', 'phbxKGlB','snkiwkvf','ZZGQNLOX', 'POJXrpmn', 'jmsRIiqp', 'izNLFWMH', 'nTaJkLaJ'], axis =1, inplace = True)
    hhold_c_test.drop(['LhUIIEHQ', 'PNAiwXUz', 'NONtAKOM', 'WWuPOkor', 'CtFxPQPT', 'qLDzvjiU', 'detlNNFh', 'tXjyOtiS', 'EQtGHLFz', 'cmjTMVrd', 'hJrMTBVd', 'IRMacrkM', 'EQSmcscG', 'aFKPYcDt', 'BBPluVrb', 'gAZloxqF', 'vSqQCatY', 'phbxKGlB','snkiwkvf','ZZGQNLOX', 'POJXrpmn', 'jmsRIiqp', 'izNLFWMH', 'nTaJkLaJ'], axis =1, inplace = True)

    print('Dropping all categoricals')

    cat_columns = list(hhold_c_train.select_dtypes(include = ['object']).columns)
    cat_columns.remove('country') # keep country. It gets selected by line above. gets dropped later. If it is here, code will get messed up later in one hot encoding
    hhold_c_train.drop(cat_columns, axis = 1, inplace = True)
    hhold_c_test.drop(cat_columns, axis = 1, inplace = True)

    #### end drop columns #####

    # make training sets
    X_train = hhold_c_train.drop(['poor','country'], axis = 1) # if you want to resample, need to leave the poor column in here so that minority/majority class can be calculated for resampling
    y_train = hhold_c_train['poor'].values
    indiv_X_train = indiv_c_train.drop(['poor','country'], axis = 1)

    # make test sets
    X_test = hhold_c_test.drop('country', axis = 1)
    indiv_X_test = indiv_c_test.drop('country', axis = 1)

    # make new features from the individual sets
    # number of individuals
    X_train = num_indiv(X_train, indiv_X_train)
    X_test = num_indiv(X_test, indiv_X_test)

    X_train['num_indiv'] = pd.cut(X_train['num_indiv'], [0, 3, 5, 10, 13, 20], labels=['0-3', '3-5', '5-10', '10-13', '16+'])
    X_test['num_indiv'] = pd.cut(X_test['num_indiv'], [0, 3, 5, 10, 13, 20], labels=['0-3', '3-5', '5-10', '10-13', '16+'])


    ## standardizing remaining columns
    # standardize only the numerical columns
    num_columns = ['xFKmUXhu', 'kLAQgdly', 'mmoCpqWS']
    X_train[num_columns] = standardize(X_train[num_columns])
    X_test[num_columns] = standardize(X_test[num_columns])

    # add these to num columns so they DON'T get selected to one hot encoding below at new_cats, but add it after standardizing so they don't get standardized
    num_columns.append('DBjxSUvf')
    num_columns.append('num_indiv')
    # log transform
    X_train['DBjxSUvf'] = np.log(X_train['DBjxSUvf'])
    X_test['DBjxSUvf'] = np.log(X_test['DBjxSUvf'])

    # one hot encoding
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


    ### end features

    params = {'n_estimators':400, 'max_depth':5, 'reg_alpha':0.5, 'reg_lambda': 0.5, 'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5, 'random_state' : 144, 'eval_metric' : 'logloss', 'verbose': 2 }

    clf = xgb.XGBClassifier(**params)

    # fit
    clf.fit(X_train, y_train)

    # predict
    preds = clf.predict_proba(X_test)

    return preds

if __name__ == '__main__':
    run_c_model()

