'''
functions to implement least squares using gradient descent, stochastic gradient descent or normal equation
All functions return w and loss , which is the last weight vector of the method and the corresponding loss
'''

import numpy as np

# from proj1_helpers import *
# from data_utility import *
from costs import *

#*****************************************
# PREDICTORS
#-----------------------------------------

def linear_predictor(x_te, w):
    '''Compute predictions for x_te in a linear model with weights w.'''
    return x_te.dot(w)

#*****************************************
# GRADIENT DESCENT
#-----------------------------------------

def gradient_descent(y, tx, initial_w, gamma, which_loss, max_iters=500, all_step=False, printing=False, **kwargs):
    """Gradient descent algorithm.
    Return: [predictor,] w, loss
    ******************
    all_step    If 'True' gives all the computed parameters and respective losses. False by default.
    printing    If 'True' print the loss and first 2 parameters estimate at each step. False by defalt.
    """

    w = initial_w
    err = np.zeros(len(y))
    if all_step:
        ws = [initial_w]
        errors = []

    for n_iter in range(max_iters):
        w_old = np.copy(w)
        # compute gradient, err
        grad, err = compute_gradient(y, tx, w, which_loss=which_loss, **kwargs)
        # gradient w by descent update
        w = w - gamma * grad

        if all_step:
            # store w and err
            ws.append(w)
            errors.append(err)

        if printing:
            print("Gradient Descent({bi}/{ti}): mse={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=calculate_mse(err), w0=w[0], w1=w[1]))

        #convergence criterion
        if max(abs(w_old-w))/(1+max(abs(w_old))) < gamma**10:
            if printing:
                print('Converged at step {bi}'.format(bi=n_iter))
            break

    if all_step:
        w_out = ws
        err_out = errors
    else:
        w_out = w
        err_out = err

    return w_out, err_out


#*****************************************
# GRADIENT DESCENT METHODS
#-----------------------------------------

def least_squares_GD(y, tx, initial_w, gamma, max_iters=500, *args, pred=False, all_step=False, printing=False):
    """Least squares computed though radient descent algorithm.
    Return: [predictor,] w, loss
    ******************
    all_step    If 'True' gives all the computed parameters and respective losses. False by default.
    printing    If 'True' print the loss and first 2 parameters estimate at each step. False by defalt.
    """
    # Define parameters to store w and loss
    w, err = gradient_descent(y, tx, initial_w, which_loss="mse", gamma=gamma, max_iters=max_iters, all_step=all_step, printing=printing)
    out = []
    if pred:
        out.append(linear_predictor)
    out.append(w)
    out.append(calculate_mse(err))

    return out


def least_squares_SGD(y, tx, initial_w, batch_size, gamma, max_iters=500, *args, pred=False, all_step=False, printing=False):
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
            loss = loss_f(err_f(y, tx, linear_predictor, w))
            # store w and loss
            ws.append(w)
            losses.append(loss)
        if printing:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    out = []
    if pred:
        out.append(linear_predictor)
    if all_step:
        out.append([ws, losses])
    else:
        out.append([ws[-1], losses[-1]])
    return out


#************************************************
#LEAST SQUARES
#------------------------------------------------

def least_squares(y, tx, *args, pred=False,):
    """
    Calculate the least squares solution.
    Returns [predictor,] w, loss.
    """
    w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    if pred:
        return linear_predictor, w, loss_f(err_f(y, tx, linear_predictor, w))
    return w, loss_f(err_f(y, tx, linear_predictor, w))

#**************************************************
# RIDGE REGRESSION
#--------------------------------------------------

def ridge_regression(y, tx, lambda_, *args, pred=False):
    """implement ridge regression.
    Returns [predictor,] w, loss.
    """
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    w = np.linalg.solve(a,tx.T.dot(y))
    if pred:
        return linear_predictor, w, loss_f(err_f(y, tx, linear_predictor, w))
    return w, loss_f(err_f(y, tx, linear_predictor, w))

#**************************************************
# LASSO REGRESSION
#--------------------------------------------------

def lasso_regression(y, tx, lambda_, initial_w=None , max_iters = 500, gamma = 0.005, printing = False, *args, pred=False):
    """implement ridge regression.
    Returns [predictor,] w, loss.
    """
    w = initial_w
    if initial_w is None:
        w = np.zeros(tx.shape[1])

    w, _ = gradient_descent(y, tx, w, which_loss="lasso", lambda_=lambda_, gamma=gamma, max_iters=max_iters, all_step=False, printing=printing)

    if pred:
        return linear_predictor, w, loss_f(err_f(y, tx, linear_predictor, w))
    return w, loss_f(err_f(y, tx, linear_predictor, w))

#*************************************************
# Logistic regression
#-------------------------------------------------


def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))


def logistic_regression(y, x, w=None, max_iters = 100, gamma = 0.000005, printing = False, pred = False):

    '''
    compute the logistic regression on the data x,y, return the probability to be 1 in the classification problem (0,1), using gradient descent
    y_proba1 = Logistic_regression(...)
    Have to add intercept to the data !
    '''

    if w is None:
        w = np.zeros(tx.shape[1])

    w, _ = gradient_descent(y, x, w, which_loss="logistic", gamma=gamma, max_iters=max_iters, all_step=False, printing=printing)

    loss = np.mean(abs(y - categories(pred_logistic(x,w))))
    # w = w.reshape(len(w))
    if pred:
        return pred_logistic, w, loss
    else :
        return w, loss


def reg_logistic_regression(y, x, lambda_, initial_w=None, max_iters = 100, gamma =0.000005 , printing = False, pred = False):
    '''
    compute the reguralized logistic regression using gradient descent
    w,loss = reg_logistic_regression(..)
    '''
    w = initial_w
    if initial_w is None:
        w = np.zeros(x.shape[1])

    for n_iter in range(max_iters):
        grad,loss = compute_gradient(y, x, w, loss = "logistic")
        #add the constrained part
        grad += lambda_*w/2.
        w_old = w
        # update w with gradient update
        w = w - gamma * grad
        # calculate loss
        # y = y.reshape(len(y),1)
        loss = np.mean(abs(y - categories(pred_logistic(x,w))))
        if printing:
            print("Gradient Descent({bi}/{ti}):loss = {l}".format(
              bi=n_iter, ti=max_iters - 1, l = loss))
        #convergence criterion
        if max(abs(w_old-w))/(1+max(abs(w_old))) < 10**-3:
            break
    # w = w.reshape(len(w))
    if pred:
        return pred_logistic, w, loss
    else :
        return w, loss




def pred_logistic(x,w):
    #w = w.reshape(len(w),1)
    y_hat = sigmoid(x.dot(w))
    #y_hat = y_hat.reshape(len(y_hat))
    return y_hat



#**************************************************
# GENERAL REGRESSION FUNCTION
#--------------------------------------------------

def regression(y, tx, method, parameters=None, loss='mse', kind='cont'):
    '''General regression function.
    Specify response y, features matrix tx, desired regression method and a parameters list formatted as follows.
    METHOD              PARAMETERS
    least squares       None
    ridge               [lambda_]
    '''
    reg_switcher = {
        'least squares': least_squares,
        'ridge': ridge_regression
    }

    reg = reg_switcher.get(method, least_squares)
    return reg(y, tx, parameters, loss, kind)


#**************************************************
# CROSS VALIDATION
#--------------------------------------------------

def single_validation(y, x, k_indices, k, method, *args_method):
    """
    Train given model on all subset of k_indices but k-th and compute prediction error over k-th subset of k_indices.
    Returns predictor, w, single_loss_tr, single_loss_te
    """

    # get k'th subgroup in test, others in train
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    k_complement = np.ravel(np.vstack([k_indices[:k],k_indices[k+1:]]))
    x_tr = x[k_complement]
    y_tr = y[k_complement]

    # regression using the method given
    predictor, w, single_loss_tr = method(y_tr, x_tr, *args_method, pred=True)

    # calculate the loss for test data
    single_loss_te = loss_f(err_f(y_te, x_te, predictor, w))
    return predictor, w, single_loss_tr, single_loss_te

def cross_validation(y, tx, k_fold, method, *args_method, k_indices=None, seed=1):
    '''
    return an estimate of the expected predicted error outside of the train set for the model, using k fold cross validation
    *args_model are the parameter needed for the model to train (for example lambda for ridge,..,)
    estimate = CV(y, tx, k_fold[, k_indices, loss_f, err_f,], model, *args_model)
    prediction = model(x_test,y_train,x_train,*args_model): model is a function that return the prediction classification for a specific model
    '''

    if k_indices is None:
        k_indices = build_k_indices(y, k_fold, seed = seed)

    single_loss_tr = np.zeros(k_fold)
    single_loss_te = np.zeros(k_fold)
    w_l = np.zeros((tx.shape[1],k_fold))
    predictor = None
    for k in range(k_fold):
        predictor, w_l[:,k], single_loss_tr[k], single_loss_te[k] = single_validation(y, tx, k_indices, k, method, *args_method)
    loss_tr = np.mean(single_loss_tr)
    loss_te = np.mean(single_loss_te)
    w = np.mean(w_l, axis=1)

    return predictor, w, loss_tr, loss_te


def multi_cross_validation(y, x, k_fold, transformations=[[id, []]], methods=[[least_squares, []]], seed=1, only_best=True):
    '''
        Run cross validation for whatever you can think of.
        Return predictors, ws, losses_tr, losses_te, t_list, m_list. (Only best value if only_best=True)
    '''
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    # ATTENTION: not sure if good idea to store everything.
    predictors = []
    ws = []
    losses_tr = []
    losses_te = []
    t_list = []
    m_list = []

    for t, t_arg in transformations:
        tx = t(x, *t_arg)
        for method, parameters in methods:
            print('Testing for method {name} with transf. {transf}({list})... Be patient! ;)'.format(name=method.__name__,
            transf=t.__name__, list=t_arg))
            for m_arg in parameters:
                predictor, w, loss_tr, loss_te = cross_validation(y, tx, k_fold, method, m_arg, k_indices=k_indices)
                predictors.append(predictor)
                ws.append(w)
                losses_tr.append(loss_tr)
                losses_te.append(loss_te)
                t_list.append([t, t_arg])
                m_list.append([method, m_arg])


    cross_validation_visualization(range(len(predictors)), losses_tr, losses_te)

    #Routine to gest just best hyper_parameter
    if only_best:
        best_i = np.argmin(losses_te)
        return predictors[best_i], ws[best_i], losses_tr[best_i], losses_te[best_i], t_list[best_i], m_list[best_i]
    return predictors, ws, losses_tr, losses_te, t_list, m_list
