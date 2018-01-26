import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, f1_score, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

#### Load data
hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()
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
# hhold_b_test.drop(['FGWqGkmD', 
     # 'BXOWgPgL',
     # 'umkFMfvA',
     # 'McFBIGsm',
     # 'IrxBnWxE',
     # 'BRzuVmyf',
     # 'dnlnKrAg',
     # 'aAufyreG',
     # 'OSmfjCbE'], axis = 1, inplace=True)

# drop columns with only 1 unique value
hhold_b_train.drop(['ZehDbxxy', 'qNlGOBmo', 'izDpdZxF', 'dsUYhgai'], axis = 1, inplace = True)
# hhold_b_test.drop(['ZehDbxxy', 'qNlGOBmo', 'izDpdZxF', 'dsUYhgai'], axis = 1, inplace = True)
# no seperation between classes
hhold_b_train.drop(['qrOrXLPM','NjDdhqIe', 'rCVqiShm', 'ldnyeZwD',
       'BEyCyEUG', 'VyHofjLM', 'GrLBZowF', 'oszSdLhD',
       'NBWkerdL','vuQrLzvK','cDhZjxaW', # added 1_17
       'IOMvIGQS'], axis = 1, inplace = True)
# hhold_b_test.drop(['qrOrXLPM','NjDdhqIe', 'rCVqiShm', 'ldnyeZwD',
       # 'BEyCyEUG', 'VyHofjLM', 'GrLBZowF', 'oszSdLhD',
       # 'NBWkerdL','vuQrLzvK','cDhZjxaW', # added 1_17
       # 'IOMvIGQS'], axis = 1, inplace = True)

# correlated features
hhold_b_train.drop(['ZvEApWrk'], axis = 1, inplace = True)
# hhold_b_test.drop(['ZvEApWrk'], axis = 1, inplace = True)

# lots of NaNs
indiv_b_train.drop(['BoxViLPz', 'qlLzyqpP', 'unRAgFtX', 'TJGiunYp', 'WmKLEUcd', 'DYgxQeEi', 'jfsTwowc', 'MGfpfHam', 'esHWAAyG', 'DtcKwIEv', 'ETgxnJOM', 'TZDgOhYY', 'sWElQwuC', 'jzBRbsEG', 'CLTXEwmz', 'WqEZQuJP', 'DSttkpSI', 'sIiSADFG', 'uDmhgsaQ', 'hdDTwJhQ', 'AJgudnHB', 'iZhWxnWa', 'fyfDnyQk', 'nxAFXxLQ', 'mAeaImix', 'HZqPmvkr', 'tzYvQeOb', 'NfpXxGQk'], axis = 1, inplace = True)

# indiv_b_test.drop(['BoxViLPz', 'qlLzyqpP', 'unRAgFtX', 'TJGiunYp', 'WmKLEUcd', 'DYgxQeEi', 'jfsTwowc', 'MGfpfHam', 'esHWAAyG', 'DtcKwIEv', 'ETgxnJOM', 'TZDgOhYY', 'sWElQwuC', 'jzBRbsEG', 'CLTXEwmz', 'WqEZQuJP', 'DSttkpSI', 'sIiSADFG', 'uDmhgsaQ', 'hdDTwJhQ', 'AJgudnHB', 'iZhWxnWa', 'fyfDnyQk', 'nxAFXxLQ', 'mAeaImix', 'HZqPmvkr', 'tzYvQeOb', 'NfpXxGQk'], axis = 1, inplace = True)

# need to rename because there are same column names in hhold and indiv 
indiv_b_train['wJthinfa_2'] = indiv_b_train['wJthinfa']
indiv_b_train.drop('wJthinfa', axis = 1, inplace = True)

# indiv_b_test['wJthinfa_2'] = indiv_b_test['wJthinfa']
# indiv_b_test.drop('wJthinfa', axis = 1, inplace = True)


print('Dropping all categoricals')

cat_columns = list(hhold_b_train.select_dtypes(include = ['object']).columns)
cat_columns.remove('country') # keep country. It gets selected by line above
hhold_b_train.drop(cat_columns, axis = 1, inplace = True)
# hhold_b_test.drop(cat_columns, axis = 1, inplace = True)

#### end drop columns #####

logloss = []
f1 = []
recall = []

X = hhold_b_train.drop(['country'], axis = 1) # need to keep poor to resample inside the CV loop
y = hhold_b_train['poor'].values
indiv_X = indiv_b_train.drop(['country'], axis = 1)

# store numerical and categorical columns. Only want to standardize the numerical ones, not the categorical which gets labelencoded
# num_columns = hhold_b_train.select_dtypes(include = ['int64', 'float64']).columns
cat_columns = hhold_b_train.select_dtypes(include = ['object']).columns

# begin CV loop
n_splits = 3
skf = StratifiedKFold(n_splits = n_splits, random_state = 144, shuffle = True)
for i, (train_idx, val_idx) in enumerate(skf.split(X,y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # get the samples in indiv set matching indexes of the hhold set in the fold
        indiv_X_train = indiv_X[indiv_X.index.get_level_values('id').isin(X_train.index.values)]
        
        # double check that the index values in the hhold data appear at least once in the individual index (indiv index has many duplicates of id because it is multi index)
        assert any(i in indiv_X_train.index.get_level_values('id').values for i in X_train.index.values)
       
        # resample data

        X_resampled = resample_data(X_train, how = 'up') # balance data with upsampling. 
        y_train = X_resampled['poor'] # resampled targets. now balaned
        X_train = X_resampled.drop('poor', axis = 1)
        X_val.drop('poor', axis = 1, inplace = True)
        # I think: don't need to resample indiv set because adding the new column will just merge to the ids in the resampled X_resampled
        # so it will match...
        
        ##########
        ## Begin features: standardizing, log transforms, labelencoder, etc go here inside CV loop 
        
        # new features
        # X_train = num_indiv(X_train, indiv_X_train) # add num_indiv column
        # X_val = num_indiv(X_val, indiv_X_train)

        # # indiv_X_train, indiv_cat_columns = labelencode_cat(indiv_X_train) # label encode categoricals in indiv train
        # # X_train = n_unique_cat(X_train, indiv_X_train, indiv_cat_columns)
        # # X_val = n_unique_cat(X_val, indiv_X_train, indiv_cat_columns)
    # #     X_train = sum_cat(X_train, indiv_X_train, indiv_cat_columns)

        # # standardize only the numerical columns
        # num_columns = ['num_indiv']
        # X_train[num_columns] = standardize(X_train[num_columns])
        # X_val[num_columns] = standardize(X_val[num_columns])

        # params = {'n_estimators':400, 'max_depth':3, 'reg_alpha':0, 'reg_lambda':1, 'min_child_weight': 5} # xgb params
        # model = xgb.XGBClassifier(**params)
        # model = KNeighborsClassifier()
        model = SVC(class_weight = 'balanced', probability = True)
        model.fit(X_train, y_train)

        # enforce X_val for prediction
        # X_val = enforce_cols(X_val, X_train)

        preds = model.predict_proba(X_val)

        logloss.append(log_loss(y_val, preds[:,1]))
       
        # other metrics
        preds_01 = (preds[:,1] > 0.5)
        f1.append(f1_score(y_val, preds_01))

        cnf_matrix = confusion_matrix(y_val, preds_01)
        print(cnf_matrix)
        
        recall.append(recall_score(y_val, preds_01))

        print(log_loss(y_val, preds_01))

print('B logloss: ', logloss)
print('Avg logloss:', np.average(logloss))
print('average f1: ', np.average(f1))
print('average recall: ', np.average(recall))

