import sys
sys.path.append("/Users/Bing/Documents/DS/DrivenData/Pover-T/Scripts/") # need to add path to the parent folder where CV.py is

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from PoverTCV import *
from PoverTHelperTools import *


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

indiv_a_train, indiv_b_train, indiv_c_train = load_indiv_train()

## NEW FEATURES FROM INDIV. ADD TO HHOLD DATA
from scipy import stats

def labelencode_cat(indiv_df):
    cat_columns = indiv_df.select_dtypes(include=['object']).columns
    indiv_df[cat_columns] = indiv_df[cat_columns].apply(LabelEncoder().fit_transform).astype('str')
    return indiv_df, cat_columns

def num_indiv(df, indiv_df):
    """
    Add number of individuals in household to the hhold sets
    """
    df['num_indiv'] = indiv_df.reset_index().groupby('id')['iid'].nunique()
    return df

def n_unique_cat(df, indiv_df,cat_columns):
    """
    takes a label encoded dataframe of categorical columns, and returns number of 
    unique values for each household from indiv data set
    """
    df[cat_columns + '_nunique'] = indiv_df[cat_columns].groupby(indiv_df.index.get_level_values('id')).nunique()
#     df[cat_columns + '_nunique'] = df[cat_columns + '_nunique'].astype('category')
    return df

def sum_cat(df, indiv_df,cat_columns):
    """
    takes a label encoded dataframe of categorical columns, and adds the values up for each column
    """
    df[cat_columns + '_sum'] = indiv_df[cat_columns].groupby(indiv_df.index.get_level_values('id').astype(int)).sum()
    return df

def mode_cat(df, indiv_df, cat_columns):
    """
    gets the mode of the category
    """
    df[cat_columns + '_mode'] = indiv_df[cat_columns].groupby(indiv_df.index.get_level_values('id')).apply(lambda x: stats.mode(x)[0][0])
    return df

X = hhold_a_train.drop(['country'], axis = 1) # need to keep poor to resample inside the CV loop
y = hhold_a_train['poor'].values
indiv_X = indiv_a_train.drop(['poor','country'], axis = 1)

cat_columns = X.select_dtypes(include = ['object']).columns
num_columns = X.select_dtypes(include = ['int64', 'float64']).columns

skf = StratifiedKFold(n_splits = 3, random_state = 144)


avg_logloss = [] # final scores to analyze later

grid = {'max_depth':[3,4,5,6,7,8], 'reg_alpha':[0,0.5,1], 'reg_lambda': [1,0.5, 0]}

print('Starting grid search...')
for params in list(ParameterGrid(grid)):
    clf = xgb.XGBClassifier(**params, eval_metic = 'logloss', random_state = 144, verbose = 2)
    logloss=[] # reset list
    
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
        X_train[num_columns] = standardize(X_train[num_columns])
        X_val[num_columns] = standardize(X_val[num_columns])
        # label encode remaining cat columns. don't want to redo what was label encoded in indiv already
        X_train[cat_columns] = X_train[cat_columns].apply(LabelEncoder().fit_transform)        # new features 
        X_val[cat_columns] = X_val[cat_columns].apply(LabelEncoder().fit_transform)        # new features 

        assert X_train.shape[0] == y_train.shape[0]

        clf.fit(X_train, y_train)

        # enforce X_val for prediction
    #     X_val = enforce_cols(X_val, X_train)

        preds = clf.predict_proba(X_val)

        logloss.append(log_loss(y_val, preds[:,1]))

    print(clf)
    print('average logloss: ', np.average(logloss))
    print('\n')
    avg_logloss.append((params, np.average(logloss)))

