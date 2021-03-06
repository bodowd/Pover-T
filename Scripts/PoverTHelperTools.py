import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

"""Helper functions to load data"""
def load_hhold_train():
    hhold_a_train = pd.read_csv('../input/A_hhold_train.csv', index_col = 'id')
    hhold_b_train = pd.read_csv('../input/B_hhold_train.csv', index_col = 'id')
    hhold_c_train = pd.read_csv('../input/C_hhold_train.csv', index_col = 'id')
    
    return hhold_a_train, hhold_b_train, hhold_c_train

def load_hhold_test():    
    hhold_a_test = pd.read_csv('../input/A_hhold_test.csv', index_col = 'id')
    hhold_b_test = pd.read_csv('../input/B_hhold_test.csv', index_col = 'id')
    hhold_c_test = pd.read_csv('../input/C_hhold_test.csv', index_col = 'id')
    
    return hhold_a_test, hhold_b_test, hhold_c_test

def load_indiv_train():
    indiv_a_train = pd.read_csv('../input/A_indiv_train.csv', index_col = ['id', 'iid'])
    indiv_b_train = pd.read_csv('../input/B_indiv_train.csv', index_col = ['id', 'iid'])
    indiv_c_train = pd.read_csv('../input/C_indiv_train.csv', index_col = ['id', 'iid'])

    return indiv_a_train, indiv_b_train, indiv_c_train

def load_indiv_test():
    indiv_a_test = pd.read_csv('../input/A_indiv_test.csv', index_col = ['id', 'iid'])
    indiv_b_test = pd.read_csv('../input/B_indiv_test.csv', index_col = ['id', 'iid'])
    indiv_c_test = pd.read_csv('../input/C_indiv_test.csv', index_col = ['id', 'iid'])

    return indiv_a_test, indiv_b_test, indiv_c_test

"""Helper functions to standardize data and preprocess data"""
def standardize(df):
    # print('Standardizing (normalizing)')
    numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # subtract mean and divide by std
    df[numeric.columns] = (numeric- numeric.mean()) / numeric.std() 
    return df

def min_max_scaler(df):
    print('Min Max scaling...')
    numeric = df.select_dtypes(include=['int64', 'float64'])
    df[numeric.columns] =  MinMaxScaler().fit_transform(df[numeric.columns])
    return df


def enforce_cols(test, train):
    """Takes the training dataframe in to enforce the columns on the test set
    test : the test set
    train : the dataframe to base the enforcement off of. Should be the train set
    """ 
    to_drop = np.setdiff1d(test.columns, train)
    to_add = np.setdiff1d(train, test.columns)

    test.drop(to_drop, axis=1, inplace=True)
    test = test.assign(**{c: 0 for c in to_add})
    
    return test

def pre_process_data(df, normalize_num = 'standardize', enforce_cols = None, fillmean=None, categorical = 'label_encoder', target = None):
    """
    Standardize the numeric columns and one hot encode the categorical columns
    """

    # First enforce columns
    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    # print('Input shape:\t{}'.format(df.shape))
    if normalize_num == 'standardize': 
        df = standardize(df)
    elif normalize_num == 'min_max':
        df = min_max_scaler(df)
    elif normalize_num is None:
        pass
    # print('After standardization {}'.format(df.shape))

    # get categorical columns
    cat_columns = df.select_dtypes(['object']).columns
    # label binarizer
    if categorical == 'label_encoder':
        print('Label encoding categoricals')
        df[cat_columns] = df[cat_columns].apply(LabelEncoder().fit_transform)
        # for col in cat_columns:
            # le = LabelEncoder()
            # df[col] = le.fit_transform(df[col])
        print('After Label encoding:\t{}'.format(df.shape))
    elif categorical == 'get_dummies':
        print('One hot encoding categoricals')
        # get dummies for categoricals--one hot encoder
        df = pd.get_dummies(df)
        print('After converting categoricals:\t{}'.format(df.shape))
        # df.info()
    
    # elif categorical == 'target':
        # for col in cat_columns:
            # df[col] = target_encode(df[col],
                           # target = target,
                           # min_samples_leaf=100,
                           # smoothing = 10,
                           # noise_level = 0.01)
    
    if fillmean == 'mean':
        print('Filling NaN with mean')
        for col in df.columns:
            df.fillna(df[col].mean(), inplace = True)
    elif fillmean == 'median':
        print('Filling NaN with median')
        for col in df.columns:
            df.fillna(df[col].median(), inplace = True)
    elif fillmean == None:
        print('Filling NaN with 0')
        # just fill NaN with 0 for now. Later try fill it with mean or explore more to figure out how to best impute
        df.fillna(0, inplace = True) 

    return df    

def make_country_sub(preds, test_feat, country):

    # just get probabilities for p = 1 with preds[:,1]
    country_sub = pd.DataFrame(data = preds[:,1],
                              columns = ['poor'],
                              index = test_feat.index)
    
    # add country code to join later on
    country_sub['country'] = country
    return country_sub[['country', 'poor']]
