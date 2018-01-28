import pandas as pd
import numpy as np

from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import xgboost as xgb
###################
SEED = 0
NFOLDS = 3

class SKlearnHelper(object):
    def __init__(self, clf, seed = 0, params = None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        # return self.clf.predict(x)
        return self.clf.predict_proba(x)[:,1]

    def fit(self, x, y):
        return self.clf.fit(x,y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x,y).feature_importances_)



def get_oof(clf, x_train, y_train, x_test, ntrain, ntest,skf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train[train_index]
        x_te = x_train.iloc[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis = 0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1,1)

###################
def run_b_model():
    # load data
    hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
    hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

    indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()
    # need to load indiv test sets here to make the new feats for the test set for submission
    indiv_a_test, indiv_b_test, indiv_c_test = load_indiv_test()

    #### prepare data
    #### Drop columns that we won't need at all ######
    # columns with lots of NaNs
    hhold_b_train.drop(['FGWqGkmD', 
         'BXOWgPgL',
         'umkFMfvA',
         'McFBIGsm',
         'IrxBnWxE',
         'BRzuVmyf',
         'dnlnKrAg', 
         'aAufyreG',
         'OSmfjCbE'], axis = 1, inplace=True)
    hhold_b_test.drop(['FGWqGkmD', 
         'BXOWgPgL',
         'umkFMfvA',
         'McFBIGsm',
         'IrxBnWxE',
         'BRzuVmyf',
         'dnlnKrAg',
         'aAufyreG',
         'OSmfjCbE'], axis = 1, inplace=True)

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

    #### end drop columns #####
    
    # begin features
    y_train = hhold_b_train['poor'].values
    X_train = hhold_b_train.drop(['poor', 'country'], axis = 1)
    X_test = hhold_b_test.drop('country', axis = 1)
    indiv_X_train = indiv_b_train.drop(['poor','country'], axis = 1)
    indiv_X_test = indiv_b_test.drop('country', axis = 1)


    # for get_oof
    ntrain = hhold_b_train.drop(['poor', 'country'], axis = 1).shape[0]
    ntest = hhold_b_test.drop('country', axis = 1).shape[0]
    skf = StratifiedKFold(n_splits = NFOLDS, random_state = SEED)

    # store cat columns and numerical columns for later use
    cat_columns = X_train.select_dtypes(include = ['object']).columns

    # make new features from the individual sets
    # number of individuals
    X_train = num_indiv(X_train, indiv_X_train)
    X_test = num_indiv(X_test, indiv_X_test)

    # label encode individual train/test set
    indiv_X_train, indiv_cat_columns = labelencode_cat(indiv_X_train)
    indiv_X_test, indiv_cat_columns = labelencode_cat(indiv_X_test)

    ## standardizing remaining columns
    # standardize only the numerical columns
    num_columns = ['num_indiv']
    X_train[num_columns] = standardize(X_train[num_columns])
    X_test[num_columns] = standardize(X_test[num_columns])


    ### end features

    # generate base first-level models
    xgb_params = {'n_estimators':400, 'max_depth':5, 'reg_alpha':0.5,
        'reg_lambda': 0.5,'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5}
    rf_params = {'n_estimators':200, 'criterion': 'entropy'}
    svc_params = {'probability': True}

    # xgb.XGBClassifier()
    xgb_clf = SKlearnHelper(clf=xgb.XGBClassifier, seed = SEED, params = xgb_params)
    rf = SKlearnHelper(clf = RandomForestClassifier, seed = SEED, params = rf_params)
    svc = SKlearnHelper(clf = SVC, seed = SEED, params = svc_params)

    # get out of fold predictions
    xgb_oof_train, xgb_oof_test = get_oof(xgb_clf, X_train, y_train, X_test, ntrain, ntest, skf)
    rf_oof_train, rf_oof_test = get_oof(rf, X_train, y_train, X_test, ntrain, ntest, skf)
    svc_oof_train, svc_oof_test = get_oof(svc, X_train, y_train, X_test, ntrain, ntest, skf)

    # first level output
    base_predictions_train = pd.DataFrame({'XGB': xgb_oof_train.ravel(),
        'RF': rf_oof_train.ravel(),
        'SVC': svc_oof_train.ravel()
        })

    print(base_predictions_train.head())

    print(base_predictions_train.corr().values)

    #### concatenate first-level train and test predictions to pass into second level
    x_train = np.concatenate((xgb_oof_train, rf_oof_train, svc_oof_train), axis = 1)
    x_test = np.concatenate((xgb_oof_test, rf_oof_test, svc_oof_test), axis = 1)

    # second level model
    lr_clf = LogisticRegression()
    lr_clf.fit(x_train, y_train)
    predictions = lr_clf.predict_proba(x_test)

    print(predictions)
    return predictions

