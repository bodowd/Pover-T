import pandas as pd
import numpy as np

import lightgbm as lgb

import PoverTHelperTools


SUB_NAME = 'lgb_50est_dropNaN.csv'

# Load data
hhold_a_train, hhold_b_train, hhold_c_train = PoverTHelperTools.load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = PoverTHelperTools.load_hhold_test()

## Try keep only important features. Got idea from forum. Didn't work well. Need to revisit the plot_importance. Mightve only looked at the top features from the LAST CV KFold
# hhold_a_train = hhold_a_train[['TiwRslOh', 'GIMIxlmv', 'nEsgxvAq', 'IZFarbPw', 'qgxmqJKa', 'wEbmsuJO', 'QyBloWXZ', 'ZnBLVaqz', 'poor', 'country']]
# hhold_b_train=hhold_b_train[['wJthinfa', 'lCKzGQow', 'toNGbjGF', 'xjTIGPgB', 'BjWMmVMX', 'RcpCILQM','poor', 'country' ]]
# hhold_c_train=hhold_c_train[['xFKmUXhu', 'DBjxSUvf','kiAJBGqv','GIwNbAsH', 'HNRJQbcm','phbxKGlB','GRGAYimk','EQtGHLFz', 'aFKPYcDt','poor', 'country']] 

# Try dropping columns with NaN. Only hhold_b_train had these. LB gave me: 0.22469!!!
hhold_b_train.drop(['FGWqGkmD',
 'BXOWgPgL',
 'umkFMfvA',
 'McFBIGsm',
 'IrxBnWxE',
 'BRzuVmyf',
 'dnlnKrAg',
 'aAufyreG',
 'OSmfjCbE'], axis = 1, inplace=True)

####
# pre-process the data -- train
aX_train = PoverTHelperTools.pre_process_data(hhold_a_train.drop(['poor', 'country'], axis = 1), fillmean = None)
bX_train = PoverTHelperTools.pre_process_data(hhold_b_train.drop(['poor', 'country'], axis = 1), fillmean = None)
cX_train = PoverTHelperTools.pre_process_data(hhold_c_train.drop(['poor', 'country'], axis = 1), fillmean = None)

ay_train = hhold_a_train['poor'].values.astype(int)
by_train = hhold_b_train['poor'].values.astype(int)
cy_train = hhold_c_train['poor'].values.astype(int)

# train
model_lgb = lgb.LGBMClassifier(n_estimators = 50,
                              objective = 'binary', 
                              num_threads = 4,
                              learning_rate = 0.05)

a_model = model_lgb.fit(aX_train, ay_train)

model_lgb = lgb.LGBMClassifier(n_estimators = 50,
                              objective = 'binary', 
                              num_threads = 4,
                              learning_rate = 0.05)
b_model = model_lgb.fit(bX_train, by_train)

model_lgb = lgb.LGBMClassifier(n_estimators = 50,
                              objective = 'binary', 
                              num_threads = 4,
                              learning_rate = 0.05)
c_model = model_lgb.fit(cX_train, cy_train)

# preprocess test
aX_test = PoverTHelperTools.pre_process_data(hhold_a_test,enforce_cols =aX_train.columns , fillmean = None)
bX_test = PoverTHelperTools.pre_process_data(hhold_b_test,enforce_cols =bX_train.columns , fillmean = None)
cX_test = PoverTHelperTools.pre_process_data(hhold_c_test,enforce_cols = cX_train.columns , fillmean = None)
## predict
a_preds = a_model.predict_proba(aX_test)
b_preds = b_model.predict_proba(bX_test)
c_preds = c_model.predict_proba(cX_test)

# make submission
a_sub = PoverTHelperTools.make_country_sub(a_preds, hhold_a_test, 'A')
b_sub = PoverTHelperTools.make_country_sub(b_preds, hhold_b_test, 'B')
c_sub = PoverTHelperTools.make_country_sub(c_preds, hhold_c_test, 'C')
sub = pd.concat([a_sub, b_sub, c_sub])
print('Submission shape: ',sub.shape)
sub.to_csv('../Submissions/{}'.format(SUB_NAME))
