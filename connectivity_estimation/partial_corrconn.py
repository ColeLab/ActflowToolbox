from scipy import linalg
from sklearn.covariance import EmpiricalCovariance,LedoitWolf
import numpy as np

def partial_corrconn(activity_matrix,estimator='EmpiricalCovariance', target_ts=None):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    estimator:      can be either 'Empirical covariance' the default, or 'LedoitWolf' partial correlation with Ledoit-Wolf shrinkage

    Output: connectivity_mat, formatted targets X sources
    Credit goes to nilearn connectivity_matrices.py which contains code that was simplified for this use.
    """

    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]
    if nnodes > timepoints:
        print('activity_matrix shape: ',np.shape(activity_matrix))
        raise Exception('More nodes (regressors) than timepoints! Use regularized regression')
    if 2*nnodes > timepoints:
        print('activity_matrix shape: ',np.shape(activity_matrix))
        print('Consider using a shrinkage method')
    
    if target_ts is None:
        connectivity_mat = np.zeros((nnodes,nnodes))
        # calculate covariance
        if estimator is 'LedoitWolf':
            cov_estimator = LedoitWolf(store_precision=False)
        elif estimator is 'EmpiricalCovariance':
            cov_estimator = EmpiricalCovariance(store_precision=False)
        covariance = cov_estimator.fit(activity_matrix.T).covariance_

        # calculate precision
        precision = linalg.inv(covariance)

        # precision to partial corr
        diagonal = np.atleast_2d(1. / np.sqrt(np.diag(precision)))
        correlation = precision * diagonal * diagonal.T

        # Force exact 0. on diagonal
        np.fill_diagonal(correlation, 0.)
        connectivity_mat = -correlation
    else:
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,1))
        X = activity_matrix.T
        y = target_ts
        #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(X, y)
        connectivity_mat=reg.coef_

    return connectivity_mat
