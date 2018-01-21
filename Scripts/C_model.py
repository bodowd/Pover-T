import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.ensemble import RandomForestClassifier

def run_c_model():
    hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
    hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

    indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()
    # need to load indiv test sets here to make the new feats for the test set for submission
    indiv_a_test, indiv_b_test, indiv_c_test = load_indiv_test() 
    
    #### Drop columns that we won't need at all #######
    # # remove some outliers
    # hhold_c_train = hhold_c_train[hhold_c_train['GIwNbAsH'] > -30]
    # hhold_c_train = hhold_c_train[hhold_c_train['DBjxSUvf'] < 50000]    
    # drop columns with only one unique value
    hhold_c_train.drop(['GRGAYimk', 'DNnBfiSI', 'laWlBVrk', 'XAmOFyyg', 'gZWEypOM', 'kZmWbEDL', 'tTScFJYA', 'xyzchLjk', 'MtkqdQSs', 'enTUTSQi', 'kdkPWxwS', 'HNRJQbcm'], axis =1 , inplace = True)
    hhold_c_test.drop(['GRGAYimk', 'DNnBfiSI', 'laWlBVrk', 'XAmOFyyg', 'gZWEypOM', 'kZmWbEDL', 'tTScFJYA', 'xyzchLjk', 'MtkqdQSs', 'enTUTSQi', 'kdkPWxwS', 'HNRJQbcm'], axis =1 , inplace = True)
    # features with overlapping distributions 
    # drop overlapping distributions
    hhold_c_train.drop(['LhUIIEHQ', 'PNAiwXUz', 'NONtAKOM', 'WWuPOkor',
           'CtFxPQPT', 'qLDzvjiU', 'detlNNFh', 'tXjyOtiS',
           'EQtGHLFz', 'cmjTMVrd', 'hJrMTBVd', 'IRMacrkM',
           'EQSmcscG', 'aFKPYcDt', 'BBPluVrb', 'gAZloxqF', 'vSqQCatY',
           'phbxKGlB','snkiwkvf','ZZGQNLOX', 'POJXrpmn', 'jmsRIiqp', 'izNLFWMH', 'nTaJkLaJ'], axis =1, inplace = True)
    hhold_c_test.drop(['LhUIIEHQ', 'PNAiwXUz', 'NONtAKOM', 'WWuPOkor',
           'CtFxPQPT', 'qLDzvjiU', 'detlNNFh', 'tXjyOtiS',
           'EQtGHLFz', 'cmjTMVrd', 'hJrMTBVd', 'IRMacrkM',
           'EQSmcscG', 'aFKPYcDt', 'BBPluVrb', 'gAZloxqF', 'vSqQCatY',
           'phbxKGlB','snkiwkvf','ZZGQNLOX', 'POJXrpmn', 'jmsRIiqp', 'izNLFWMH', 'nTaJkLaJ'], axis =1, inplace = True)
    
    print('Dropping all categoricals')
    
    cat_columns = list(hhold_c_train.select_dtypes(include = ['object']).columns)
    cat_columns.remove('country') # keep country. It gets selected by line above
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
    
    # store cat columns and numerical columns for later use
    # cat_columns = X_train.select_dtypes(include = ['object']).columns
    # num_columns = X_train.select_dtypes(include = ['int64', 'float64']).columns

    # make new features from the individual sets
    # number of individuals
    X_train = num_indiv(X_train, indiv_X_train)
    X_test = num_indiv(X_test, indiv_X_test)

    # label encode individual train/test set
    indiv_X_train, indiv_cat_columns = labelencode_cat(indiv_X_train)
    indiv_X_test, indiv_cat_columns = labelencode_cat(indiv_X_test)
    
    # log transform
    X_train['DBjxSUvf'] = np.log(X_train['DBjxSUvf'])
    X_test['DBjxSUvf'] = np.log(X_test['DBjxSUvf'])
    # X_train['nTaJkLaJ'] = np.log(X_train['nTaJkLaJ']+10) # might not be that good of feature. Poor 0/1 overlaps quite a bit
    # X_test['nTaJkLaJ'] = np.log(X_test['nTaJkLaJ']+10) # might not be that good of feature. Poor 0/1 overlaps quite a bit

    ## standardizing remaining columns
    # standardize only the numerical columns
    num_columns = ['xFKmUXhu', 'kLAQgdly', 'mmoCpqWS', 'DBjxSUvf'] 
    X_train[num_columns] = standardize(X_train[num_columns])
    X_test[num_columns] = standardize(X_test[num_columns])


    # label encode remaining cat columns. Don't want to redo what was encoded in individual set already
    # X_train[cat_columns] = X_train[cat_columns].apply(LabelEncoder().fit_transform)
    # X_test[cat_columns] = X_test[cat_columns].apply(LabelEncoder().fit_transform)

    ### end features
    
    # params = {'n_estimators':400, 'max_depth':5, 'reg_alpha':0.5, 'reg_lambda': 0.5, 
           # 'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5, 'random_state' = 144, 
              # 'eval_metric' : 'logloss', 'verbose': 2
           # }
    params = {'n_estimators':200, 'criterion': 'entropy'}
    # clf = xgb.XGBClassifier(**params)
    clf = RandomForestClassifier(**params)

    # fit
    clf.fit(X_train, y_train)

    # predict
    preds = clf.predict_proba(X_test)

    return preds

