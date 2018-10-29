'''
Function which produces exactly the same .csv predictions which we used in our
best submission in the Kaggle competition

'''

from Regressions import *




###### logarithmic transformation on selected features

#first replace -999 by nan

def X_processing(X):

    # add log transformations

    x_nan = X.copy()
    x_nan[x_nan==-999]=np.nan

    selected_features = [3,8,13,16,19,23,29]
    selected_features_plus = [3,1,1,1,2,2,1]

    for i, feature in enumerate(selected_features):
        x_nan[:,feature] = log_plus(x_nan[:,feature], selected_features_plus[i])

    x_nan[np.isnan(x_nan)]=-999

    ###### Feature selection and data preprocessing
    ## take primitive phi and eta features out
    features = np.zeros(30,dtype=bool)
    keepers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,19,21,22,23,26,29]
    features[keepers] = True
    ## pretreat the test data (imputation, normalization, categorical splitting)
    X_treated = preliminary_treatment_X(x_nan, features, imp_method="median")

    ###### Polynomial feature extension of degree 4
    X_poly = build_poly(X_treated, 4)

    ###### Interaction with the binary categorical variables and linear terms of
    ###### continuous variables
    X_poly_inter = X_poly.copy()
    for i in range(4):
        for j in range(4,X_poly.shape[1],4):
            X_poly_inter = interactions(X_poly_inter,i,j)

    return X_poly_inter


#load the data
y_tr, X_tr, ids_tr = load_csv_data('../Data/train.csv')
y_te, X_te, ids_te = load_csv_data('../Data/test.csv')

#change -1 in y to 0
y_tr[y_tr==-1] = 0
y_te[y_te==-1] = 0

######## generate the model

#find the optimized lambda of the ridge regression
#set the training data to be used
tx = X_processing(X_tr)
w = np.zeros(tx.shape[1])
# define the lambdas range
lambdas = np.logspace(-8,0,15)
k_fold = 5
transformations = [[id,[]]]
methods = [[ridge_regression, lambdas]]
predictor, w, loss_tr, loss_te, transformation, method = multi_cross_validation(y_tr, tx, k_fold, transformations=transformations, methods=methods, seed=1, only_best=True)

#get the optimized w with the optimal determined lambda
predictor, w, loss = func(y_tr, tx, par, pred = True)

###### predict label with the optimized w vector

tx_te = X_processing(X_te)
y_pred = predict_labels(w,tx_te)
# get the prediction in the binary -1, 1 format
y_predict = categories(y_pred)
y_predict[y_predict==0]=-1


create_csv_submission(ids_te, y_predict, "Kaggle_CDM_submission.csv")
