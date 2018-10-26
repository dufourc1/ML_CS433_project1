# -*- coding: utf-8 -*-
'''
Utility functions for data analysis and treatment.
Contains:
    split_data
    build_k_indices
    build_poly

'''
from proj1_helpers import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
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
    num_row = len(y)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


#*************************************************
# FEATURES TRANSFORMATIONS
#-------------------------------------------------
def id(x,*args):
    return x

def log_plus(feature, n=1, *args):
    return np.log(feature + n)

def feature_multitransform(x, functions, *args):
    tx = np.copy(x)
    for func, features, *arg in functions:
        for feature in features:
            tx[:,feature] = func(tx[:,feature], *arg)
    return tx

def feature_transform(x, func, features, *args):
    tx = np.copy(x)
    for feature in features:
        tx[:,feature] = func(tx[:,feature], *args)
    return tx

def build_poly(x, degree, *args):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Return the augmented basis matrix tx."""

    n,p = x.shape
    tx = np.zeros((n,(degree*p)+1))
    tx[:,0] = 1
    for feature in range(p):
        for i in range(1,degree+1):
            tx[:,feature*degree+i]=(x[:,feature])**i
    return tx

def interactions(x,i,j):
    '''
    add to the data the vector of pointwise product between x[indices[0]] and x[indidces[1]]

    data = interactions(data,4,5)
    '''

    x_new = x[:,i]*x[:,j]

    #reshape so that the hstack gets the proper dimensions
    x_new = x_new.reshape(len(x_new),1)

    x_augmented = np.hstack((x,x_new))

    return x_augmented


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

def cross_validation_visualization(steps, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.plot(steps, mse_tr, marker=".", color='b', label='train error')
    plt.plot(steps, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("CV step")
    plt.ylabel("loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    #plt.savefig("cross_validation")

#**********************************************
# DATA INPUTATION
#----------------------------------------------


def imputation(data, method = "mean",features_treated = "all" ):
    '''
    Impute the missing values with the different methods: mean,median

    example of use: data_inputed = inputation(data,method = "median")

    data is the data you wish to inpute
    features_treated is the array containing the columns of the data you want to inpute,
    the default configuration is for the all dataset (where there is no features with more than
    70 % of NA's and we inpute the missing values where there are some)

    for the moment only treat the features number 1,19,20,21 but could be easily generalized if felt necessary
    '''

    if features_treated == "all":
        features_treated = range(data.shape[1])

    data_imputed = data.copy()

    if method == "mean":
        for i in features_treated:
            t = data_imputed[:,i]
            mean = np.mean(t[t != -999])
            t[t == -999] = mean
            data_imputed[:,i] = t

        return data_imputed

    if method == "median":
        for i in features_treated:
            t = data_imputed[:,i]
            median = np.median(t[t != -999])
            t[t == -999] = median
            data_imputed[:,i] = t

        return data_imputed

#**********************************************
# PRELIMINARY TREATMENT
#----------------------------------------------

def preliminary_treatment_X(X, keepers=np.ones(30,dtype=bool), imp_method="mean") :
    '''
    imputation, normalization of the data, features engineering are done here
    '''


    keepers[22]= 0

    jet_num = X[:,22]
    n = len(jet_num)
    null_v = np.zeros((n,1))
    jet_0 = np.copy(null_v)
    jet_1 = np.copy(null_v)
    jet_2 = np.copy(null_v)
    jet_3 = np.copy(null_v)

    jet_0[jet_num==0] = 1
    jet_1[jet_num==1] = 1
    jet_2[jet_num==2] = 1
    jet_3[jet_num==3] = 1
    intercept = np.ones((n,1))


    x = standardize_data(imputation(X[:,keepers], imp_method))
    return np.hstack((intercept, x, jet_0,jet_1,jet_2,jet_3))


#************************************************
#Splitting data depending on the value of jet_num
#------------------------------------------------

def split_num_jet(data,y):
    '''

    num_jet0,y0, num_jet1,y1, num_jet2,y2 = split_num_jet(data)

    takes raw data (i.e all the covariates), then return three different dataset depending on the value of num_jet, after
    normalizing and doing inputation on the data

    return an empty array if there is no num_jet with the corresponding value

    the values 2 and 3 are merged since they do not seem to differ a lot and splitting between these two would make the
    data_set for the regression even smaller
    '''

    data_cat = data.copy()
    y_cat = y.copy()

    try :
        num_jet = data_cat[:,22]
    except :
        num_jet = data_cat[22]

    #split the data depending on the value of num_jet
    data_n2 = np.vstack((data_cat[num_jet == 2], data_cat[num_jet == 3]))
    #have to change the dimension otherwise it won't stack them properly
    y2a = y_cat[num_jet == 2].reshape(len(np.where(num_jet== 2)[0]),1)
    y2b = y_cat[num_jet == 3].reshape(len(np.where(num_jet== 3)[0]),1)
    y2 = np.vstack((y2a,y2b))

    data_n0 = data_cat[num_jet == 0]
    y0 = y_cat[num_jet == 0]
    data_n1 = data_cat[num_jet == 1]
    y1 = y_cat[num_jet == 1]


    #the only feature where there will be NA left after the next step
    data_n0 = imputation(data_n0,features_treated = [0])
    data_n1 = imputation(data_n1,features_treated = [0])
    data_n2 = imputation(data_n2,features_treated = [0])

    #delete the features with 100% of NA depending on the value if num_jet, plud deletion of the feature num_jet
    data_n0_modified = np.delete(data_n0,[4,6,12,23,24,25,26,27,28,22],1)
    data_n1_modified = np.delete(data_n1,[4,5,6,12,26,27,28,22],1)
    data_n2_modified = np.delete(data_n2,[22],1)


    data_n1_modified = standardize_data(data_n1_modified)
    data_n2_modified = standardize_data(data_n2_modified)
    data_n0_modified = standardize_data(data_n0_modified)

    return data_n0_modified,y0,data_n1_modified,y1,data_n2_modified,y2


#*************************************************
# CATEGORISATION
#-------------------------------------------------

def categories(y_hat):
    y_cat = np.copy(y_hat)
    y_cat[y_hat >= 0.5] = 1
    y_cat[y_hat < 0.5] = 0
    return y_cat
