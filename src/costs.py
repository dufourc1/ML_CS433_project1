# -*- coding: utf-8 -*-
"""Functions to compute loss. | || || |_
Contains:
    error
    mse, mae, rmse
    mse gradient"""

import math
import numpy as np
from data_utility import *
from Regressions import *



def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))

def error(y, tx, pred_func, w):
    '''Compute estimation error'''
    e = y - pred_func(tx, w)
    return e

def category_error(y, tx, pred_func, w):
    '''Gives error for binary categorization (Cat. coded as 0-1)
    Input true values as y, features as tx and estimated weights as w'''
    y_hat = categories(pred_func(tx, w))
    e = np.zeros((len(y_hat),1))
    e[y != y_hat] = 1
    return e

#**************************************
# GLOBAL ERROR FUNCTION
err_f = category_error

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_rmse(e):
    """ Calculate the mse for vector e."""
    return math.sqrt(2.0*calculate_mse(e))

#****************************************
# GLOBAL LOSS FUNCTION
loss_f = calculate_mae

def calculate_loss(y, tx, w, loss, kind):
    """Calculate the loss.
    Arguments:
    y       Target;
    tx      Observed features;
    w       Estimated weights;
    loss    Desired loss function (possible: 'mae', 'mse', 'rmse');
    kind    Kind of target (possible: 'cont', 'cat')
    """
    loss_switcher = {
        'mae': calculate_mae,
        'mse': calculate_mse,
        'rmse': calculate_rmse
    }
    err_switcher = {
        'cont': error,
        'cat': category_error
    }
    loss_func = loss_switcher.get(loss, calculate_mse)
    err_func = err_switcher.get(kind, error)

    return loss_func(err_func(y, tx, linear_predictor, w)) ####is this a mistake, should it be err_f ? I'm a bit lost


def compute_gradient(y, tx, w, which_loss='mse', kind='cont', **kwargs):

    if which_loss == "mse":
        """Compute the gradient of mse."""
        err = y - tx.dot(w)
        grad = -tx.T.dot(err) / len(err)
        return grad, err

    elif which_loss == "logistic":
        '''compute the gradient for the logistic regression'''
        # w = w.reshape(len(w),1)
        # y = y.reshape(len(y),1)
        grad = tx.T.dot(sigmoid(tx.dot(w))-y)
        err = calculate_mse(sigmoid(tx.dot(w))-y)
        return grad, err

    elif which_loss == "lasso":
        '''Compute the gradient for Lasso'''
        # w = w.reshape(len(w),1)
        # y = y.reshape(len(y),1)
        lambda_ = kwargs.get('lambda_',0)
        N = len(y)
        err = y - tx.dot(w)
        grad = -tx.T.dot(err)/N + lambda_*np.sign(w)
        return grad, err


    else:
        raise(NotImplementedError)
