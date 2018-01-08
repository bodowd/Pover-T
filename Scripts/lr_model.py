import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

import PoverTHelperTools


SUB_NAME = 'LR_dropNaN_drpCorrFeats_drplowdiversity_trsfmHighSkew.csv'

# Load data
hhold_a_train, hhold_b_train, hhold_c_train = PoverTHelperTools.load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = PoverTHelperTools.load_hhold_test()
## Try keep only important features. Got idea from forum. Didn't work well. Need to revisit the plot_importance. Mightve only looked at the top features from the LAST CV KFold
# hhold_a_train = hhold_a_train[['TiwRslOh', 'GIMIxlmv', 'nEsgxvAq', 'IZFarbPw', 'qgxmqJKa', 'wEbmsuJO', 'QyBloWXZ', 'ZnBLVaqz', 'poor', 'country']]
# hhold_b_train=hhold_b_train[['wJthinfa', 'lCKzGQow', 'toNGbjGF', 'xjTIGPgB', 'BjWMmVMX', 'RcpCILQM','poor', 'country' ]]
# hhold_c_train=hhold_c_train[['xFKmUXhu', 'DBjxSUvf','kiAJBGqv','GIwNbAsH', 'HNRJQbcm','phbxKGlB','GRGAYimk','EQtGHLFz', 'aFKPYcDt','poor', 'country']] 

# NOTE: pre_process_data will enforce columns to be the same as train so it will match the columns to train later
# Try dropping columns with NaN. Only hhold_b_train had these. LB gave me: 0.22469!!!
print('Before dropping NaN: ',hhold_b_train.shape)
hhold_b_train.drop(['FGWqGkmD',
 'BXOWgPgL',
 'umkFMfvA',
 'McFBIGsm',
 'IrxBnWxE',
 'BRzuVmyf',
 'dnlnKrAg',
 'aAufyreG',
 'OSmfjCbE'], axis = 1, inplace=True)
print('Dropped NaN columns in hhold_b. Shape is now: ', hhold_b_train.shape)

# Drop correlated features: 0.22487 on LB. Didn't improve from dropping NaN, slightly worse than just dropping NaN. Removed features that correlate with others > 0.5
# Try removing less features. Removed features that are >0.7 correlated. Did about the same. Saved GIwNbAsH even though it's correlated to other things because it's distribution looks pretty good.
print('Before dropping correlated features in B: ', hhold_b_train.shape)
print('Before dropping correlated features in C: ', hhold_c_train.shape)
hhold_b_train.drop(['ZvEApWrk'], axis = 1, inplace = True)
hhold_c_train.drop(['jmsRIiqp', 'WWuPOkor', 'CtFxPQPT', 'gAZloxqF', 'PNAiwXUz', 'izNLFWMH', 'EQSmcscG'], axis = 1, inplace = True)
print('Dropped correlated features in c and b')
print('B: ', hhold_b_train.shape)
print('C: ', hhold_c_train.shape)

# Try drop features that have low diversity, or have no seperation between poor and not poor in the distribution. Found these by looking at features with really high skew (like ~23 & ~41). 
hhold_a_train.drop(['nEsgxvAq', 'OMtioXZZ'], axis = 1, inplace = True) # OMtioXZZ might want to keep
hhold_b_train.drop(['cDhZjxaW', 'oszSdLhD', 'GrLBZowF', 'NBWkerdL', 'rCVqiShm', 'NjDdhqIe'], axis = 1, inplace = True)
hhold_c_train.drop(['detlNNFh', 'xFKmUXhu', 'snkiwkvf', 'POJXrpmn' ], axis = 1, inplace = True)

# Log transform features with high skew
hhold_c_train['DBjxSUvf'] = np.log(hhold_c_train['DBjxSUvf'])
hhold_c_train['nTaJkLaJ'] = np.log(hhold_c_train['nTaJkLaJ']+10) # might not be that good of feature. Poor 0/1 overlaps quite a bit

####
# pre-process the data -- train
aX_train = PoverTHelperTools.pre_process_data(hhold_a_train.drop(['poor', 'country'], axis = 1), fillmean = None)
bX_train = PoverTHelperTools.pre_process_data(hhold_b_train.drop(['poor', 'country'], axis = 1), fillmean = None)
cX_train = PoverTHelperTools.pre_process_data(hhold_c_train.drop(['poor', 'country'], axis = 1), fillmean = None)

ay_train = hhold_a_train['poor'].values.astype(int)
by_train = hhold_b_train['poor'].values.astype(int)
cy_train = hhold_c_train['poor'].values.astype(int)

# train
model_LR = LogisticRegression()
a_model = model_LR.fit(aX_train, ay_train)

model_LR = LogisticRegression()
b_model = model_LR.fit(bX_train, by_train)

model_LR = LogisticRegression()
c_model = model_LR.fit(cX_train, cy_train)

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
