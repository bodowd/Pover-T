# notes:
# resample data: when up or down sampling, it makes the log loss way worse, but gives non zero F1 and recall. However, without resampling, this will get low log loss BUT 0.0 for f1 and recall. Shows the problem of log loss on unbalanced targets. The xgb is not predicting anything above 0.5 (no 1's)

import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, f1_score, recall_score, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

#########

def skfCV_B():
    # Load data
    print('Loading data...')
    hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
    hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

    indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()

    #########
    # drop columns
    # columns with lots of NaNs
    hhold_b_train.drop(['FGWqGkmD', 'BXOWgPgL', 'umkFMfvA', 'McFBIGsm', 'IrxBnWxE', 'BRzuVmyf', 'dnlnKrAg', 'aAufyreG', 'OSmfjCbE'], axis = 1, inplace=True)

    # drop columns with only 1 unique value
    hhold_b_train.drop(['ZehDbxxy', 'qNlGOBmo', 'izDpdZxF', 'dsUYhgai'], axis = 1, inplace = True)

    # no seperation between classes
    hhold_b_train.drop(['qrOrXLPM','NjDdhqIe', 'rCVqiShm', 'ldnyeZwD', 'BEyCyEUG', 'VyHofjLM', 'GrLBZowF', 'oszSdLhD', 'NBWkerdL','vuQrLzvK','cDhZjxaW', 'IOMvIGQS'], axis = 1, inplace = True)

    # correlated features
    hhold_b_train.drop(['ZvEApWrk'], axis = 1, inplace = True)

    # lots of NaNs
    indiv_b_train.drop(['BoxViLPz', 'qlLzyqpP', 'unRAgFtX', 'TJGiunYp', 'WmKLEUcd', 'DYgxQeEi', 'jfsTwowc', 'MGfpfHam', 'esHWAAyG', 'DtcKwIEv', 'ETgxnJOM', 'TZDgOhYY', 'sWElQwuC', 'jzBRbsEG', 'CLTXEwmz', 'WqEZQuJP', 'DSttkpSI', 'sIiSADFG', 'uDmhgsaQ', 'hdDTwJhQ', 'AJgudnHB', 'iZhWxnWa', 'fyfDnyQk', 'nxAFXxLQ', 'mAeaImix', 'HZqPmvkr', 'tzYvQeOb', 'NfpXxGQk'], axis = 1, inplace = True)

    # need to rename because there are same column names in hhold and indiv
    indiv_b_train['wJthinfa_2'] = indiv_b_train['wJthinfa']
    indiv_b_train.drop('wJthinfa', axis = 1, inplace = True)

    # drop all categoricals
    cat_columns = list(hhold_b_train.select_dtypes(include = ['object']).columns)
    cat_columns.remove('country') # keep country. It gets selected by line above
    hhold_b_train.drop(cat_columns, axis = 1, inplace = True)

    #########
    # make training sets

    logloss = [] # append log loss scores to this array
    f1 = []
    recall = []

    X = hhold_b_train.drop('country', axis = 1)
    y = hhold_b_train['poor'].values
    indiv_X = indiv_b_train.drop(['poor','country'], axis = 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.05)
    # begin CV loop
    # n_splits = 3
    # skf = StratifiedKFold(n_splits = n_splits, random_state = 2, shuffle = True)
    # for train_index, val_index in skf.split(X,y):
        # X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        # y_train, y_val = y[train_index], y[val_index]

    # get the samples in indiv set matching indexes of the hhold set in the fold
    indiv_X_train = indiv_X[indiv_X.index.get_level_values('id').isin(X_train.index.values)]
    indiv_X_val = indiv_X[indiv_X.index.get_level_values('id').isin(X_val.index.values)]
    # double check that the index values in the hhold data appear at least once in the individual index (indiv index has many duplicates of id because it is multi index)
    assert any(i in indiv_X_train.index.get_level_values('id').values for i in X_train.index.values)

    X_resampled = resample_data(X_train, how = 'up')
    y_train = X_resampled['poor']
    X_train = X_resampled.drop('poor', axis = 1)
    X_val.drop('poor', axis = 1, inplace = True)

    # X_train.drop('poor', axis = 1, inplace = True)
    # X_val.drop('poor', axis = 1, inplace = True)

    # make new features from indiv data set
    X_train = num_indiv(X_train, indiv_X_train)
    X_val = num_indiv(X_val, indiv_X_val)

    num_columns = ['num_indiv']
    X_train[num_columns] = standardize(X_train[num_columns])
    X_val[num_columns] = standardize(X_val[num_columns])

    # X_train['num_indiv'] = pd.cut(X_train['num_indiv'], [0, 3, 5, 10, 13, 20], labels=['0-3', '3-5', '5-10', '10-13', '16+'])
    # X_val['num_indiv'] = pd.cut(X_val['num_indiv'], [0, 3, 5, 10, 13, 20], labels=['0-3', '3-5', '5-10', '10-13', '16+'])

    # one hot encoding
    # concatenate train and test to do the one hot encoding. Train and test don't have the same categorical values so one hot encoding gives different number of features on the different sets
    X_train['wJthinfa'] = X_train['wJthinfa'].astype('str') # treat this feature as categorical
    X_val['wJthinfa'] = X_val['wJthinfa'].astype('str')

    # print(X_train.head(1))
    tmp = pd.concat((X_train, X_val))
    tmp = pd.get_dummies(tmp)
    X_train = tmp.iloc[:X_train.shape[0]]
    # print(X_train.head(1))
    X_val = tmp.iloc[X_train.shape[0]:]

    params = {'n_estimators':400, 'max_depth':3, 'reg_alpha':0, 'reg_lambda':1, 'min_child_weight': 5} # xgb params
    clf = xgb.XGBClassifier(**params)

    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_val)

    # metrics
    # log loss
    logloss.append(log_loss(y_val, preds[:,1]))

    # convert probs to 0/1 for the following metrics
    preds_01 = (preds[:,1] > 0.5)
    f1.append(f1_score(y_val, preds_01))

    recall.append(recall_score(y_val, preds_01))

    print('Confusion Matrix: \n')
    print(confusion_matrix(y_val, preds_01))
    print('')
    print('AUC: ', roc_auc_score(y_val, preds_01))
    print('Average logloss for B: ', np.average(logloss))
    print('log losses for each fold: ', logloss)
    print('average f1: ', np.average(f1))
    print('f1 for each fold: ', f1)
    print('average recall: ', np.average(recall))
    print('recall for each fold: ', recall)

    print('Predictions: \n')
    print(preds_01)
    print(preds[:,1])
    return logloss, f1, recall

if __name__ == '__main__':
    skfCV_B()

