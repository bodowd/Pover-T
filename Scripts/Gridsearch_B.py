import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.feature_selection import SelectFromModel

import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()


#### Drop columns that we won't need at all. Don't want to accidently upsample rows with these in there and then throw balance off
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
# drop columns with only 1 unique value
hhold_b_train.drop(['ZehDbxxy', 'qNlGOBmo', 'izDpdZxF', 'dsUYhgai'], axis = 1, inplace = True)

# no seperation between classes
hhold_b_train.drop(['qrOrXLPM','NjDdhqIe', 'rCVqiShm', 'ldnyeZwD',
       'BEyCyEUG', 'VyHofjLM', 'GrLBZowF', 'oszSdLhD',
       'IOMvIGQS'], axis = 1, inplace = True)

# correlated features
hhold_b_train.drop(['ZvEApWrk'], axis = 1, inplace = True)

# drop columns with lots of NaNs
indiv_b_train.drop(['BoxViLPz', 'qlLzyqpP', 'unRAgFtX', 'TJGiunYp', 'WmKLEUcd', 'DYgxQeEi', 'jfsTwowc', 'MGfpfHam', 'esHWAAyG', 'DtcKwIEv', 'ETgxnJOM', 'TZDgOhYY', 'sWElQwuC', 'jzBRbsEG', 'CLTXEwmz', 'WqEZQuJP', 'DSttkpSI', 'sIiSADFG', 'uDmhgsaQ', 'hdDTwJhQ', 'AJgudnHB', 'iZhWxnWa', 'fyfDnyQk', 'nxAFXxLQ', 'mAeaImix', 'HZqPmvkr', 'tzYvQeOb', 'NfpXxGQk'], axis = 1, inplace = True)

indiv_b_train['wJthinfa_2'] = indiv_b_train['wJthinfa']
indiv_b_train.drop('wJthinfa', axis = 1, inplace = True)

# just drop all category columns
print('Dropping all categoricals')
cat_columns = list(hhold_b_train.select_dtypes(include = ['object']).columns)
cat_columns.remove('country') # keep country. It gets selected by line above
hhold_b_train.drop(cat_columns, axis = 1, inplace = True)
#### end drop columns


#### Begin CV
# prep training and test data
X = hhold_b_train.drop(['country'], axis = 1) # need to keep poor to resample inside the CV loop
y = hhold_b_train['poor'].values
indiv_X = indiv_b_train.drop(['poor','country'], axis = 1)

# num_columns = X.select_dtypes(include = ['int64', 'float64']).columns

skf = StratifiedKFold(n_splits = 3, random_state = 144)

avg_logloss = [] # final scores to analyze later
# grid = {'n_estimators':[100], 'max_depth':[5], 'reg_alpha':[0.5], 'reg_lambda': [0.5], 
       # 'min_child_weight': [1], 'gamma' : [0.1,0.2,0.3], 'subsample': [0.5,0.8]
       # }

grid = {'n_estimators':[400], 'max_depth':[3], 'reg_alpha':[0], 'reg_lambda': [1], 'min_child_weight': [5]} # xgb params
# , 'gamma' : [0.1, 0.2, 0.3], 'subsample': [0.5]
print('Starting grid search...')
for params in list(ParameterGrid(grid)):
    clf = xgb.XGBClassifier(**params, eval_metric = 'logloss', random_state = 144, verbose = 2)
    # clf = RandomForestClassifier(**params)
    logloss=[] # reset list
    f1 = []    
    for train_idx, val_idx in skf.split(X,y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # get the corresponding id rows from X_train and X_val from the indiv sets
        indiv_X_train = indiv_X[indiv_X.index.get_level_values('id').isin(X_train.index.values)]
        indiv_X_val = indiv_X[indiv_X.index.get_level_values('id').isin(X_val.index.values)]

        # double check that the index values in the hhold data appear at least once in the individual index (indiv index has many duplicates of id because it is multi index)
        assert any(i in indiv_X_train.index.get_level_values('id').values for i in X_train.index.values)

        # I think: don't need to resample indiv set because adding the new column will just merge to the ids in the resampled X_resampled
        # so it will match...
        X_resampled = resample_data(X_train, how = 'up') # balance data with upsampling. 
        y_train = X_resampled['poor'] # resampled targets. now balaned
        X_train = X_resampled.drop('poor', axis = 1)
        X_val.drop('poor', axis = 1, inplace=True)

        ##########
        ## Begin features: standardizing, log transforms, labelencoder, etc go here inside CV loop 

        # new features
        # number of individuals
        X_train = num_indiv(X_train, indiv_X_train) # add num_indiv column
        X_val = num_indiv(X_val, indiv_X_val)

        indiv_X_train, indiv_cat_columns = labelencode_cat(indiv_X_train) # label encode categoricals in indiv train
        indiv_X_val, indiv_cat_columns = labelencode_cat(indiv_X_val)
    #     X_train = n_unique_cat(X_train, indiv_X_train, indiv_cat_columns)
    #     X_val = n_unique_cat(X_val, indiv_X_train, indiv_cat_columns)

        ## standardizing remaining columns
        # standardize only the numerical columns
        # num_columns = ['num_indiv', 'cDhZjxaW', 'vuQrLzvK', 'NBWkerdL', 'wJthinfa']
        num_columns = ['num_indiv']
        X_train[num_columns] = standardize(X_train[num_columns])
        X_val[num_columns] = standardize(X_val[num_columns])

        # label encode remaining cat columns. don't want to redo what was label encoded in indiv already
        # X_train[cat_columns] = X_train[cat_columns].apply(LabelEncoder().fit_transform)        # new features 
        # X_val[cat_columns] = X_val[cat_columns].apply(LabelEncoder().fit_transform)        # new features 

        assert X_train.shape[0] == y_train.shape[0]
       
        clf.fit(X_train, y_train)
        
        preds = clf.predict_proba(X_val)

        logloss.append(log_loss(y_val, preds[:,1]))
        preds_01 = (preds[:,1] > 0.5)
        f1.append(f1_score(y_val, preds_01))

    print(params)
    print('average logloss: ', np.average(logloss))
    print('\n')
    print('average f1: ', np.average(f1))
    avg_logloss.append((params, np.average(logloss)))

