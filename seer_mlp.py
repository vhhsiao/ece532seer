## Prepare workspace
from scipy.io import loadmat
from sklearn import linear_model
from sklearn import impute
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Returns errors approximating d using A,w
def errors(d, d_pred):
    err = np.where(d_pred!=d, 1, 0)
    return err

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
pred_errors = np.zeros(cases)
final_aucs = np.zeros(cases)

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
    hidden_neurons = [50,100]

    # Find best lambda value using first validation set, then evaluate
    # performance on second validation set, and accumulate performance metrics
    # over all cases partitions

    print("Calculating performance for model MLP")
    error_rates = np.zeros(len(hidden_neurons))
    aucs = np.zeros(len(hidden_neurons))
    mdls = {}
    for i in range(len(hidden_neurons)):
        nh = hidden_neurons[i]
        clf = MLPClassifier(hidden_layer_sizes=nh).fit(At, np.ravel(bt))
        mdls[i] = clf
        scores = clf.predict_proba(Av1)
        bv1_pred = clf.predict(Av1)
        err = errors(bv1, bv1_pred)
        error_rates[i] = np.average(err)
        aucs[i] = metrics.roc_auc_score(bv1, scores[:,1])

    #best_lambda_idx = [i for i in range(len(lam_vals)) if error_rates[i] == min(error_rates)][0]
    best_nneuron_idx = [i for i in range(len(hidden_neurons)) if aucs[i] == max(aucs)][0]
    print("Best neuron index: i".format(i=best_nneuron_idx))
    final_mdl = mdls[best_nneuron_idx]
    final_scores = final_mdl.predict_proba(Av2)
    final_pred = final_mdl.predict(Av2)
    final_errors = errors(bv2, final_pred)
    final_error_rate = np.average(final_errors)
    pred_errors[j] = final_error_rate
    final_aucs[j] = metrics.roc_auc_score(bv2, final_scores[:,1])
    print("Error rates: {er}".format(er=error_rates.round(3)))
    print("Final error rate: {er}".format(er=final_error_rate.round(3)))
    print("AUCs: {auc}".format(auc=aucs.round(3)))
    print("Best Lambda Index: {l}".format(l=best_nneuron_idx))

print("Average MLP prediction error rate: {er}".format(er=np.average(pred_errors).round(3)))
print("Average MLP AUC: {auc}".format(auc=np.average(final_aucs).round(3)))

