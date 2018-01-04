import pandas as pd
import numpy as np


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
    numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # subtract mean and divide by std
    df[numeric.columns] = (numeric- numeric.mean()) / numeric.std()
    
    return df

def pre_process_data(df, enforce_cols = None):
    """
    Standardize the numeric columns and one hot encode the categorical columns
    """
    
    print('Input shape:\t{}'.format(df.shape))
    
    df = standardize(df)
    print('After standardization {}'.format(df.shape))
    
    # get dummies for categoricals
    df = pd.get_dummies(df)
    print('After converting categoricals:\t{}'.format(df.shape))
    df.info()
    
    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    
    # just fill NaN with 0 for now. Later try fill it with mean or explore more to figure out how to best impute
    df.fillna(0, inplace = True) 
    
    
    return df    
