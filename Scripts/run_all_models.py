# V1 - xgbA rfB rfC removed categorial features with nunique >10 and outliers in C
# V2 - xgbA xgbB rfC removed categorical featues with nunique >10 and outliers in C  remove categorical features with only 1 unique feature LB 0.26999
# V3 - xgbA xgbB rfC leave the features with nunique >10. remove categorical features with only 1 unique feature 
# 1_16. Removed overlapping distributions in all countries. LB: 0.19532. CV: ~0.29689
# 1_17. Replaced categorical with frequency in country C. Made things worse for the other countries. LB was worse. 0.25669. CV: 0.28336382637574314
# 1_17_v2. Added a few more features to drop in countryB, and one in countryA of overlapping distributions. Slightly worse than 1_16: 0.19588
# 1_17_v3. Removed a bunch of categorical features that did not seem predictive in countryA. CV got worse. But CV seems unreliable again...LB: 0.22991
# 1_17_v4. Dropped all categoricals in countryB. Returned country A model back to v2. LB: 0.20818...categoricals seem pretty useless in B? log loss only got slightly worse than v2. NExt, check out C.
# 1_21_v1. Drop all categoricals in C still had cats in B. LB : 0.18572. CV is still hard to figure out. I tried looking at F1 scores too. With categoricals and without categoricals in C is about the same F1 score...
# 1_21_v2. drop all cat in C and B. still have some cats in A LB: 0.18512
# 1_22_v1. upsample B. LB: 0.33852... check the probabilities. are they just getting conservative, but they're actually right?
# 1_23_v1. All RF models   dropping all the cats in B and C. LB : 0.30588
# 1_23_v2 . All xgb models. dropped all cats in B and C LB: 0.18425
# 1_23_v3. convert everything > 0.5 to 0/1 . LB: 4.05918
# 1_23_v4. dropped all cats in A. LB: 0.26196
# XGB_A_BC_all_zeros got LB: ~2.
# lgb_all LB: 0.22413
# keep some cats in A, drop all cats B and C. All XGB. LB: ~0.21 
# 1_25  if proba is >0.9, make it 1.0, and if it is < 0.05 make it 0. Double down when the model is confident LB: 0.37160....dang
# 1_26_v1 SVC_A, XGB B AND C... LB:0.18975
# 1_26_v2 SVC_A, SVC_B XGB C LB: 0.19123
# 1_27_v1. Stack A  LB:0.21840
# 1_27_v2. XGBA stack BC LB:0.21653
# 1_29_v1. Mean of Stack on A. BC as before (XGB) LB:0.21656
# 1_29_v2. Mean of 3 XGBs on A. BC as single XGB LB: 0.22141
# 2_3_v1. Try get dummies on A, BC as before. LB: 0.1863
# 2_3_v2. Use get dummies on all countries. Just noticed after dropping all the cats and stuff in B, there is basically one feature left, wJthinfa and then the one I created num_indiv. LB: 0.18400. new lowest score, but only barely better than previous best. people on LB have far lower scores, I wonder if that is due to model tuning or some more feature engineering can be pushed. Num_indiv seems to be a useful feature. See notebook 'num_indiv'

from A_model import *
from B_model import *
from C_model import *

# from A_stacker import *
# from B_stacker import *
# from C_stacker import *

from PoverTHelperTools import *

import pandas as pd

SUB_NAME = '2_3_v2.csv'

hhold_a_train, hhold_b_train, hhold_c_train = load_hhold_train()
hhold_a_test, hhold_b_test, hhold_c_test = load_hhold_test()

a_preds = run_a_model()
b_preds = run_b_model()
c_preds = run_c_model()


a_sub = make_country_sub(a_preds, hhold_a_test, 'A')
b_sub = make_country_sub(b_preds, hhold_b_test, 'B')
c_sub = make_country_sub(c_preds, hhold_c_test, 'C')

sub = pd.concat([a_sub, b_sub, c_sub])
print('Submission shape: ', sub.shape)
sub.to_csv('../Submissions/{}'.format(SUB_NAME))
