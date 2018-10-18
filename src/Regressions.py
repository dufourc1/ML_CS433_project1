'''
functions to implement least squares using gradient descent, stochastic gradient descent or normal equation

All functions return w and loss , which is the last weight vector of the method and the corresponding loss
'''

import numpy as np

from costs import *
from gradient_descent import *
from stochastic_gradient_descent import *
from data_utility import *

#*****************************************
# GRADIENT DESCENT METHODS
#-----------------------------------------

def gradient_descent(y, tx, initial_w, max_iters, gamma, all_step=False, printing=False):
    """Gradient descent algorithm.
    Return: w, loss
    ******************
    all_step    If 'True' gives all the computed parameters and respective losses. False by default.
    printing    If 'True' print the loss and first 2 parameters estimate at each step. False by defalt.
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if printing:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    if all_step:
        return ws, losses
    else :
        return ws[-1], losses[-1]

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

#************************************************
#LEAST SQUARES
#------------------------------------------------

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return (w,loss)

def least_squares_SGD(y, tx, initial_w,max_iters, gamma):

    return (w,loss)

def least_squares(y, tx):
    """
    Calculate the least squares solution.
    Returns w and mse loss.
    """
    w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))

    return w, compute_mse(error(y,tx,w))

#**************************************************
# RIDGE REGRESSION
#--------------------------------------------------

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    Returns w and rmse loss"""
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    w = np.linalg.solve(a,tx.T.dot(y))
    return w, compute_RMSE(error(y,tx,w))

#**************************************************
# CROSS VALIDATION
#--------------------------------------------------

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train: DONE
    x_te = x[k_indices[k]]
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    k_complement = np.ravel(np.vstack([k_indices[:k],k_indices[k+1:]]))
    x_tr = x[k_complement]
    y_tr = y[k_complement]

    # form data with polynomial degree: DONE
    x_te = build_poly(x_te, degree)
    x_tr = build_poly(x_tr, degree)

    # ridge regression: DONE
    w = ridge_regression(y_tr, x_tr, lambda_)

    # calculate the loss for train and test data: DONE
    loss_tr = compute_RMSE(y_tr, x_tr, w)
    loss_te = compute_RMSE(y_te, x_te, w)

    return loss_tr, loss_te
