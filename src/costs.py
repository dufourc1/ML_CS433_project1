# -*- coding: utf-8 -*-
"""Functions to compute loss. | || || |_
Contains:
    error
    mse, mae, rmse
    mse gradient"""

import math
import numpy as np
from data_utility import *

def error(y, tx, w):
    '''Compute estimation error'''
    e = y - tx.dot(w)
    return e

def category_error(y, tx, w):
    '''Gives error for categorization (2 coategories coded as 0-1)
    Input true values as y, features as tx and estimated weights as w'''
    y_hat = categories(tx.dot(w))
    e = np.zeros(len(y_hat))
    e[y != y_hat] = 1
    return e

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_rmse(e):
    """ Calculate the mse for vector e."""
    return math.sqrt(2.0*calculate_mse(e))

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

    return loss_func(err_func(y, tx, w))


def compute_gradient(y, tx, w, loss='mse', kind='cont'):
    """Compute the gradient of mse."""
    err = calculate_loss(y, tx, w, loss, kind)
    grad = -tx.T.dot(err) / len(err)
    return grad, err
