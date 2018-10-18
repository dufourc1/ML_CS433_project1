# -*- coding: utf-8 -*-
'''
Utility functions for regression.
Contains:
    split_data
    build_k_indices
    build_poly

'''

import numpy as np

#*************************************************
# GENERAL FUNCTIONS
#-------------------------------------------------

#Data splitting
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)

    set_size = len(y)
    n = int(np.floor(set_size*ratio))
    shuffled_i = np.random.permutation(set_size)
    train_i = shuffled_i[:n]
    test_i = shuffled_i[n:]
    x_train = x[train_i]
    x_test = x[test_i]
    y_train = y[train_i]
    y_test = y[test_i]

    return x_train, x_test, y_train, y_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Return the augmented basis matrix tx."""
    tx = np.zeros((x.shape[0],degree+1))
    for i in range(degree+1):
        tx[:,i]=x**i

    return tx

#*************************************************
# PLOTS FUNCTIONS
#-------------------------------------------------

def scatter(x, which, other_f =False, against=None):
    '''Scatter plot of the features of x.
    Arguments:
        other_f: bool, if True give an argument in against
        against: an np.array of same size of x features against which you will have the scatterplot.'''
        
    for i in which:
        feature = x[:,i]
        if not other_f:
            if len(feature[feature==-999]) > 0: #If there is some misplaced value we do not include them in the scatterplot
                print("ATTENTION: missing values in {i}th feature removed!".format(i=i))
            feature = feature[feature>-999]
            against = range(len(feature))
        plt.scatter(feature, against)
        print("Scatter plot for {i}th feature :".format(i=i))
        plt.show()
