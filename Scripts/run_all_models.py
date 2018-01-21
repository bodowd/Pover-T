# V1 - xgbA rfB rfC removed categorial features with nunique >10 and outliers in C
# V2 - xgbA xgbB rfC removed categorical featues with nunique >10 and outliers in C  remove categorical features with only 1 unique feature LB 0.26999
# V3 - xgbA xgbB rfC leave the features with nunique >10. remove categorical features with only 1 unique feature 
# 1_16. Removed overlapping distributions in all countries. LB: 0.19532. CV: ~0.29689
# 1_17. Replaced categorical with frequency in country C. Made things worse for the other countries. LB was worse. 0.25669. CV: 0.28336382637574314
# 1_17_v2. Added a few more features to drop in countryB, and one in countryA of overlapping distributions. Slightly worse than 1_16: 0.19588
# 1_17_v3. Removed a bunch of categorical features that did not seem predictive in countryA. CV got worse. But CV seems unreliable again...LB: 0.22991
# 1_17_v4. Dropped all categoricals in countryB. Returned country A model back to v2. LB: 0.20818...categoricals seem pretty useless in B? log loss only got slightly worse than v2. NExt, check out C.
# 1_22_v1. Drop all categoricals in C. LB : 0.18572. CV is still hard to figure out. I tried looking at F1 scores too. With categoricals and without categoricals in C is about the same F1 score...

from A_model import *
from B_model import *
from C_model import *

from PoverTHelperTools import *

import pandas as pd

SUB_NAME = 'xgbA_xgbB_rfC_1_22_v1.csv'

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
