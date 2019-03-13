# Helpful functions for building and assessing models and shaping data

import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import load

def expand_grid(data_dict):
    """Create a dataframe from every combination of given values."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def one_hot(df, r_col_list):
    '''One-hot encodes some variables and removes the originals.'''
    col_list = [c for c in r_col_list if c in df.columns]
    dummies = pd.concat([pd.get_dummies(df[c], 
                                        prefix = c, 
                                        drop_first = True) for c in col_list], axis = 1)
    return(pd.concat([df.drop(col_list, axis = 1), dummies], axis = 1))

def brier_score(y, y_hat):
    '''Calculates the Brier Score for some set of observations
    y and the predictions y_hat. Assumes these two are np arrays.'''
    bs = np.mean((y - y_hat)**2)
    return(bs)
    
def scaled_brier_score(y, y_hat):
    '''Calculate the scaled Brier Score for some set of observations
    y and the predictions y_hat. Assumes these two are np arrays.'''
    bs = brier_score(y, y_hat)
    bs_uninformative = brier_score(y, np.mean(y))
    scaled_bs = 1 - (bs / bs_uninformative)
    return(scaled_bs)

def cv_wrapper(split, imp, day, target, model_name):
    '''Wrapper function to load pre-processed training data files
    and generate CV performance for each model.'''
    # Load model
    model_fname = 'models/final_' + model_name + '_' + split + '_' + imp + '_' + str(day) + '_' + target + '.joblib'
    model = load(model_fname)
    data_fname = 'processed_data/dat_'+ model_name + '_' + split + '_' + imp + '_' + str(day) + '_' + target + '_train.csv'
    df = pd.read_csv(data_fname)
    # sort first so the CV splits will be the same in all cases
    df.sort_values(by='id')
    df.drop('id', axis = 1)
    y = df.target
    X = df.drop('target', axis = 1)
    return(get_cv_score(model, X, y))

def get_cv_score(final_model, X, y):
    '''Generate cross-validated performance on the training set using the final model.
    NB. Use the same splits as all other models for consistency'''
    # NB sign of score is flipped: https://github.com/scikit-learn/scikit-learn/issues/2439
    cv_scores = -1 * cross_val_score(final_model, X, y, scoring = 'brier_score_loss',
                               cv = StratifiedKFold(n_splits = 5, random_state = 2019))
    # Convert BS to scaled BS for comparison
    sbs = 1 - (cv_scores / (np.mean((y - np.mean(y))**2)))
    #print("CV Scaled Brier Scores:", sbs)
    #print("Mean CV Scaled Brier Score (+/- 95CI) on Training Data: %0.2f (+/- %0.2f)" % (sbs.mean(), sbs.std() * 1.96))
    return(sbs.mean(), sbs.mean() - 1.96 * sbs.std(), sbs.mean() + 1.96 * sbs.std())