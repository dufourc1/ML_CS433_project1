# CDM Group Project

The purpose of this project is to generate an ML model able to infer the presence, or absence, of the Higgs Boson starting from CERN measurements. The data were obtained on the Kaggle competition platform ( https://www.kaggle.com/c/epfml18-higgs ).

In this README you will find general information about the methods, and a more detailed documentation is given within the functions.

## Getting Started

The provided code was tested with Python 3.6.5.
The following libraries are used within the script:

Computational:

    numpy (as np)

Graphical:

    seaborn (as sns)
    matplotlib (as plt)


### Prerequisites

The folder structure is the following:

    .
    ├── Data                    # Data files, in .csv
    ├── src                     # Source files
    └── README.md

All the scripts are in src, where in `run.py` you can find the code that generates our prediction.


## Implementation details

### run.py

<!-- @Marie put here the description of how you obtained the best model! (general and schematic should be fine) -->

### Implementation of class methods.

For the sake of automation we added a `pred` keyword argument (kwarg) to all our model functions. It is `False` by default, and if set to `True` the function returns as first output a pointer on the function to use in order to get the predictions for that model.

All functions using the `gradient_descent` algorithm have, in addition, the two kwargs `printing, all_step`, which are `False` by default. If `printing=True`, then at all GD steps you will see in the shell the actual mse value and the value of the first two parameters of `w`. If `all_step=True`, then the function returns all the computed w-s and errors (by default they are not stored and only the last value is given).

The following functions were implemented:

| Function            | Arguments |
|-------------------- |-----------|
| `least_squares_GD`  | `y, tx, initial_w[, max_iters, gamma, *args, **kwargs]`  |
| `least_squares_SGD` | `y, tx, initial_w[, batch_size, max_iters, gamma, *args, **kwargs]`  |
| `least_squares`     | `y, tx[, **kwargs]` |
| `ridge_regression`  | `y, tx, lambda_[, **kwargs]` |
| `logistic_regression`| `y, x, [w, max_iters, gamma, **kwars]` |
| `reg_logistic_regression` | `y, x, lambda_, [initial_w, max_iters, gamma, **kwargs]` |

The default values were chosen in order to get convergence on the GD algorithm.

### Notes on `cross_validation` and `multi_cross_validation`

These are the two main functions inplemented in order to chose our model, and in particular to get an estimation of the prediction error.

* `cross_validation(y, tx, k_fold, method, *args_method[, k_indices, seed])` compute the k-fold cross validation for the estimation of `y` using a the method-function stored (as pointer) in the argument `method`. The arguments necessary for the `method` are to be passed freely after method. It returns `predictor, w, loss_tr, loss_te`, which are, in order, the predicting function, the mean of the trained weights, the mean of the train error and the estimate test error.

* `multi_cross_validation(y, x, k_fold[, transformations=[[id, []]], methods=[[least_squares, []]], seed=1, only_best=True])` Perform automatically the cross validation on all the combinations of transformations in the `transformations` list (their parameters have to be passed as a list coupled with the transformation) and methods with changing parameters in the `methods` list (the coupled list have in this case to be a list of the tuples of parameters combinations to test.) It then plot the estimated losses (both on train and test) ans outputs `predictor, weight, losses_tr, losses_te, transformations_list, methods_list`. If `only_best=True`, those are the variables corresponding to the lowest test-error estimate, otherwise they contain the variables computed at each step. An implementation example can be found in the documentation.

## Authors

* *William Cappelletti*
* *Charles Dufour*
* *Marie Sadler*
