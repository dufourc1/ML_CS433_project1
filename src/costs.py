# -*- coding: utf-8 -*-
"""Functions to compute loss. | || || |_
Contains:
    error
    mse, mae, rmse
    mse gradient"""

import math
import numpy as np


def error(y, tx, w):
    '''Compute estimation error'''
    e = y - tx.dot(w)
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

# TODO: define a better general function.
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using different methods.
    """
    e = y - tx.dot(w)
    raise notImplementedError
    # return calculate_mae(e)

def compute_gradient(y, tx, w):
    """Compute the gradient of mse."""
    err = error(y, tx, w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err
