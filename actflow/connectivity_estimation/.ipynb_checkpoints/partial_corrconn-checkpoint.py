
from sklearn.covariance import EmpiricalCovariance
import numpy as np

def partial_corrconn(activity_matrix, target_ts=None):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    Output: connectivity_mat, formatted targets X sources
    """

    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]
    if nnodes > timepoints:
        print('activity_matrix shape: ',np.shape(activity_matrix))
        raise Exception('More nodes (regressors) than timepoints! Use regularized regression')

    if target_ts is None:
        connectivity_mat = np.zeros((nnodes,nnodes))
        cov = EmpiricalCovariance().fit(activity_matrix.T)
        #Calculate the inverse covariance matrix (equivalent to partial correlation)
        connectivity_mat=cov.get_precision()
    else:
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,1))
        X = activity_matrix.T
        y = target_ts
        #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(X, y)
        connectivity_mat=reg.coef_

    return connectivity_mat