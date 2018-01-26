import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PoverTCV import *
from PoverTHelperTools import *
from NewFeatFuncs import *

from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, f1_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


# Load data
print('Loading data...')
hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()



######
print('Running CV...')


####--- Drop columns that we won't need at all ####
# columns with lots of NaNs
indiv_a_train.drop('OdXpbPGJ', axis = 1, inplace = True)
# indiv_a_test.drop('OdXpbPGJ', axis = 1, inplace = True)

# these features have overlapping distributions. improved CV just a little bit
hhold_a_train.drop(['YFMZwKrU',
    # 'nEsgxvAq', # added 1_17
    'OMtioXZZ'], axis = 1, inplace = True)
# hhold_a_test.drop(['YFMZwKrU', 
    # # 'nEsgxvAq', # added 1_17 . removed again 1_17. See if it helps to have it in there, while dropping all categoricals in B
    # 'OMtioXZZ'], axis = 1, inplace = True)

# cat_columns = hhold_a_train.select_dtypes(include = ['object']).columns
# cat_to_keep = ['QyBloWXZ', 'NRVuZwXK', 'JwtIxvKg', 'KjkrfGLD', 'bPOwgKnT', 'bMudmjzJ', 'glEjrMIg', 'LjvKYNON','HHAeIHna' ,'CrfscGZl', 'yeHQSlwg', 'ZnBLVaqz', 'AsEmHUzj', 'pCgBHqsR', 'wEbmsuJO', 'IZFarbPw', 'GhJKwVWC', 'EuJrVjyG', 'qgxmqJKa', 'DNAfxPzs', 'xkUFKUoW', 'AtGRGAYi','xZBEXWPR','ishdUooQ','ptEAnCSs', 'kLkPtNnh','PWShFLnY', 'uRFXnNKV','vRIvQXtC', 'UjuNwfjv','cDkXTaWP' ,'country']
# cat_to_drop = list(set(cat_to_keep)^set(cat_columns))

# hhold_a_train.drop(cat_to_drop, axis = 1, inplace = True)
# hhold_a_test.drop(cat_to_drop, axis = 1, inplace = True)
# print('train shape: ', hhold_a_train.shape)
# print('test shape: ', hhold_a_test.shape)


#### end drop columns #####
logloss = []
f1 = []
recall=[]
X = hhold_a_train.drop(['country'], axis = 1) # need to keep poor to resample inside the CV loop
y = hhold_a_train['poor'].values
indiv_X = indiv_a_train.drop(['country'], axis = 1)

# store numerical and categorical columns. Only want to standardize the numerical ones, not the categorical which gets labelencoded
# num_columns = hhold_a_train.select_dtypes(include = ['int64', 'float64']).columns

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
        X_train.drop('poor', axis = 1, inplace = True) 
        # resample data

        # X_resampled = resample_data(X_train, how = 'up') # balance data with upsampling. 
        # y_train = X_resampled['poor'] # resampled targets. now balaned
        # X_train = X_resampled.drop('poor', axis = 1)
        X_val.drop('poor', axis = 1, inplace = True)
        # I think: don't need to resample indiv set because adding the new column will just merge to the ids in the resampled X_resampled
        # so it will match...
        
        ##########
        ## Begin features: standardizing, log transforms, labelencoder, etc go here inside CV loop 
        
        # new features
        # X_train = num_indiv(X_train, indiv_X_train) # add num_indiv column
        # X_val = num_indiv(X_val, indiv_X_train)

        # indiv_X_train, indiv_cat_columns = labelencode_cat(indiv_X_train) # label encode categoricals in indiv train
        # X_train = n_unique_cat(X_train, indiv_X_train, indiv_cat_columns)
        # X_val = n_unique_cat(X_val, indiv_X_train, indiv_cat_columns)
    #     X_train = sum_cat(X_train, indiv_X_train, indiv_cat_columns)

        # standardize only the numerical columns
        # num_columns = ['TiwRslOh', 'num_indiv']
        num_columns = ['TiwRslOh']
        X_train[num_columns] = standardize(X_train[num_columns])
        X_val[num_columns] = standardize(X_val[num_columns])
        
        cat_columns = list(hhold_a_train.select_dtypes(include = ['object']).columns)
        cat_columns.remove('country')
        # label encode remaining cat columns. don't want to redo what was label encoded in indiv already
        X_train[cat_columns] = X_train[cat_columns].apply(LabelEncoder().fit_transform)        # new features 
        X_val[cat_columns] = X_val[cat_columns].apply(LabelEncoder().fit_transform)        # new features 
        assert X_train.shape[0] == y_train.shape[0]
    
        # params = {'n_estimators':100, 'max_depth':5, 'reg_alpha':0.5, 'reg_lambda': 0.5, 
	      # 'min_child_weight': 1, 'gamma' : 0.1, 'subsample': 0.5, 'random_state' : 144,'eval_metric' : 'logloss', 'verbose': 2}

        # model = xgb.XGBClassifier(**params)
        model = KNeighborsClassifier()
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

print('A logloss: ', logloss)
print('Avg logloss:', np.average(logloss))
print('average f1: ', np.average(f1))
print('average recall: ', np.average(recall))

