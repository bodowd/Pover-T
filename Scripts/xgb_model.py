import pandas as pd
import numpy as np
import scipy

import xgboost as xgb

import PoverTHelperTools


SUB_NAME = 'xgb_benchmark.csv'

# Load data
hhold_a_train, hhold_b_train, hhold_c_train = PoverTHelperTools.load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = PoverTHelperTools.load_hhold_test()

# pre-process the data -- train
aX_train = PoverTHelperTools.pre_process_data(hhold_a_train.drop(['poor', 'country'], axis = 1), fillmean = None)
bX_train = PoverTHelperTools.pre_process_data(hhold_b_train.drop(['poor', 'country'], axis = 1), fillmean = None)
cX_train = PoverTHelperTools.pre_process_data(hhold_c_train.drop(['poor', 'country'], axis = 1), fillmean = None)
aX_train_sparse = scipy.sparse.csc_matrix(aX_train)
bX_train_sparse = scipy.sparse.csc_matrix(bX_train)
cX_train_sparse = scipy.sparse.csc_matrix(cX_train)

ay_train = hhold_a_train['poor'].values.astype(int)
by_train = hhold_b_train['poor'].values.astype(int)
cy_train = hhold_c_train['poor'].values.astype(int)

# train
print('Training xgb models')
# country A
xgb_model = xgb.XGBClassifier(objective = 'binary:logistic', 
                        learning_rate = 0.1,
                       n_jobs = 4)

a_model = xgb_model.fit(aX_train_sparse, ay_train)
# country B
xgb_model = xgb.XGBClassifier(objective = 'binary:logistic', 
                        learning_rate = 0.1,
                       n_jobs = 4)
b_model = xgb_model.fit(bX_train_sparse, by_train)
# country C
xgb_model = xgb.XGBClassifier(objective = 'binary:logistic', 
                        learning_rate = 0.1,
                       n_jobs = 4)

c_model = xgb_model.fit(cX_train_sparse, cy_train)

# preprocess test
aX_test = PoverTHelperTools.pre_process_data(hhold_a_test, enforce_cols = aX_train.columns , fillmean = None)
bX_test = PoverTHelperTools.pre_process_data(hhold_b_test, enforce_cols = bX_train.columns , fillmean = None)
cX_test = PoverTHelperTools.pre_process_data(hhold_c_test, enforce_cols = cX_train.columns , fillmean = None)
# need to do the following for xgb
aX_test_sparse = scipy.sparse.csc_matrix(aX_test)
bX_test_sparse = scipy.sparse.csc_matrix(bX_test)
cX_test_sparse = scipy.sparse.csc_matrix(cX_test)
## predict
a_preds = a_model.predict_proba(aX_test_sparse)
b_preds = b_model.predict_proba(bX_test_sparse)
c_preds = c_model.predict_proba(cX_test_sparse)

# make submission
a_sub = PoverTHelperTools.make_country_sub(a_preds, hhold_a_test, 'A')
b_sub = PoverTHelperTools.make_country_sub(b_preds, hhold_b_test, 'B')
c_sub = PoverTHelperTools.make_country_sub(c_preds, hhold_c_test, 'C')
sub = pd.concat([a_sub, b_sub, c_sub])
print('Submission shape: ',sub.shape)
sub.to_csv('../Submissions/{}'.format(SUB_NAME))
