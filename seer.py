## Prepare workspace
from scipy.io import loadmat
from sklearn import linear_model
from sklearn import svm
from sklearn import impute
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def decision(vec):
    return np.asarray([[0] if i<0.5 else [1] for i in vec])

# Returns errors approximating d using A,w
def errors(d, d_pred):
    err = np.where(d_pred!=d, 1, 0)
    return err

def residual_norm(A, w, d):
    resid = A @ w - d
    resid_norm = np.linalg.norm(resid, 2)
    return resid_norm

# Helper functions for Question 2
def residual_norms(A, W, d):
    resid_norms = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        w = W[:,i]
        resid = residual_norm(A, w, d)
        resid_norms[i] = resid
    return resid_norms

# Returns the sparsity of w (number of nonzero entries in w)
def sparsity(w):
    sparsity = [1 if wi > 10**(-6) else 0 for wi in w]
    return sum(sparsity)

# The closed form solution to ridge regression - using SVD
def fit_ridge_models(A, d, la_array):
    n = A.shape[1]
    num_lam = len(la_array)
    X = np.zeros((n,num_lam))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    for i, each_lambda in enumerate(la_array):
        #w = A.transpose() @ np.linalg.inv(A@A.transpose() + each_lambda * np.identity(A.shape[0]))@d
        w = Vt.transpose() @ np.linalg.inv(np.diag(s**2) + each_lambda*np.identity(Vt.shape[0])) @ np.diag(s) @ U.transpose() @ d
        X[:, i] = np.asmatrix(w).transpose()
    return X

# Trains lasso on a grid of lambda values
def fit_lasso_models(A, d, la_array):
    n = A.shape[1]
    num_lam = len(la_array)
    _,coefs,_ = linear_model.lasso_path(A, d, alphas=la_array)
    return coefs

# Trains SVM on a grid of lambdas, gammas
def fit_svm_models(A, d, la_array, ga_array):
    n_lambdas = len(la_array)
    n_gammas = len(ga_array)
    models = {l:{} for l in range(n_lambdas)}
    for i in range(n_lambdas):
        print("lambda {idx}: {l}".format(idx=i, l=la_array[i]))
        for j in range(n_gammas):
            print("gamma {idx}: {g}".format(idx=j, g=ga_array[j]))
            mdl = svm.SVC(C=la_array[i], gamma=ga_array[j])
            models[i][j] = mdl.fit(A, d)
    return models

seer = loadmat("/Volumes/HSIAO USB/SEER/seer_data.mat")
print([key for key in seer])
X = seer['seer_input']
y = seer['seer_ten_year_mortality']

# Impute missing values in data using median
print("Imputing missing training values using median imputation...")
imputer = impute.SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

## Set up the 10-fold CV partitions

print(np.shape(X)[0])
n = np.shape(X)[0]
k = 10
fold_size = int(n/k)
print("10-fold CV: n, k, fold size")
print(n, k, fold_size)

# each row of setindices denotes the starting an ending index for one
# partition of the data: 5 sets of 30 samples and 5 sets of 29 samples
setindices = [[start,start+fold_size] for start in np.arange(k)*fold_size]

# each row of holdoutindices denotes the partitions that are held out from
# the training set
holdoutindices = [[1,2],[2,3],[3,4],[4,5],[5,6],[7,8],[9,10],[10,1]]

cases = len(holdoutindices)

##  10-fold CV

# Initiate the quantities you want to measure before looping
# through the various training, validation, and test partitions
pred_errors = {}
final_aucs = {}
best_lambdas = {'ridge':[], 'lasso':[]}
pred_errors['ridge'] = np.zeros(cases)
pred_errors['lasso'] = np.zeros(cases)
final_aucs['ridge'] = np.zeros(cases)
final_aucs['lasso'] = np.zeros(cases)
final_aucs['svm'] = np.zeros(cases)

# Cross validation framework for regression models
for j in range(cases):
    print("Case:{c}".format(c=j))
    # row indices of first validation set
    v1_ind = np.arange(setindices[holdoutindices[j][0] - 1][0] - 1, setindices[holdoutindices[j][0] - 1][1])

    # row indices of second validation set
    v2_ind = np.arange(setindices[holdoutindices[j][1] - 1][0] - 1, setindices[holdoutindices[j][1] - 1][1])

    # row indices of training set
    trn_ind = list(set(range(n)) - set(v1_ind) - set(v2_ind))

    # define matrix of features and labels corresponding to first
    # validation set
    Av1 = X[v1_ind, :]
    bv1 = y[v1_ind]

    # define matrix of features and labels corresponding to second
    # validation set
    Av2 = X[v2_ind, :]
    bv2 = y[v2_ind]

    # define matrix of features and labels corresponding to the
    # training set
    At = X[trn_ind, :]
    bt = y[trn_ind]

    print("Validation 1: {v1_i} | Validation 2: {v2_i} | training_indices: {trn_i}".format(v1_i=v1_ind, v2_i=v2_ind, trn_i=trn_ind))

    # Regression Models
    ## Grid of parameters to try
    lam_vals = np.logspace(-2, 10, num=5)

    print("Training lasso models...")
    W_lasso = fit_lasso_models(At, bt, lam_vals)[0]

    print("Training ridge models...")
    W_ridge = fit_ridge_models(At, bt, lam_vals)

    # Find best lambda value using first validation set, then evaluate
    # performance on second validation set, and accumulate performance metrics
    # over all cases partitions

    regression_mdls = {'ridge': W_ridge, 'lasso': W_lasso}
    for alg in regression_mdls:
        print("Calculating performance for model {m}".format(m=alg))
        error_rates = np.zeros(len(lam_vals))
        aucs = np.zeros(len(lam_vals))
        for i in range(len(lam_vals)):
            w = regression_mdls[alg][:, i]
            scores = Av1 @ w
            bv1_pred = decision(scores)
            err = errors(bv1, bv1_pred)
            error_rates[i] = np.average(err)
            aucs[i] = metrics.roc_auc_score(bv1, scores)

        #best_lambda_idx = [i for i in range(len(lam_vals)) if error_rates[i] == min(error_rates)][0]
        best_lambda_idx = [i for i in range(len(lam_vals)) if aucs[i] == max(aucs)][0]
        final_scores = Av2 @ regression_mdls[alg][:, best_lambda_idx]
        final_errors = errors(bv2, decision(final_scores))
        final_error_rate = np.average(final_errors)
        pred_errors[alg][j] = final_error_rate
        final_aucs[alg][j] = metrics.roc_auc_score(bv2, final_scores)
        best_lambdas[alg].append(best_lambda_idx)
        print("Error rates {r}: {er}".format(r=alg, er=error_rates.round(3)))
        print("Final error rate {r}: {er}".format(r=alg, er=final_error_rate.round(3)))
        print("AUCs {r}: {auc}".format(r=alg, auc=aucs.round(3)))
        print("Best Lambda Index: {l}".format(l=best_lambda_idx))

    # SVM
    print("Fitting model SVM")
    svm_lambdas = [1, 0.1]
    svm_gammas = [0.003, 1]
    SVM_models = fit_svm_models(At, np.ravel(bt), svm_lambdas, svm_gammas)
    print("Calculating performance for SVM")
    best_auc = 0
    best_lambda_idx = 0
    best_gamma_idx = 0
    for i in range(len(svm_lambdas)):
        for j in range(len(svm_gammas)):
            scores = SVM_models[i][j].decision_function(Av1)
            auc = metrics.roc_auc_score(bv1, scores)
            if auc > best_auc:
                best_auc = auc
                best_lambda_idx = i
                best_gamma_idx = j
    final_mdl = SVM_models[best_lambda_idx][best_gamma_idx]
    final_errors = errors(bv2, final_mdl.predict(Av2))
    final_scores = final_mdl.decision_function(Av2)
    final_error_rate = np.average(final_errors)
    pred_errors['svm'][j] = final_error_rate
    final_aucs['svm'][j] = metrics.roc_auc_score(bv2, final_scores)
    print("Final error rate {r}: {er}".format(r='svm', er=final_error_rate.round(3)))
    print("AUCs {r}: {auc}".format(r='svm', auc=aucs.round(3)))
    print("Best Lambda Index: {l}".format(l=best_lambda_idx))
    print("Best Gamma Index: {g}".format(g=best_gamma_idx))

    print("-----------------------------------------")

for alg in ['lasso', 'ridge']:
    print("Average {r} prediction error rate: {er}".format(r=alg, er=np.average(pred_errors[alg]).round(3)))
    print("Average {r} AUC: {auc}".format(r=alg, auc=np.average(final_aucs[alg]).round(3)))


