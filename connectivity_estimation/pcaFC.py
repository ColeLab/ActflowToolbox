# Compute principle component regression functional connectivity 

import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = str(1)
from sklearn.decomposition import PCA


def pcaFC(stim,resp,n_components=500,nproc=10):
    """
    stim    - time x feature/region matrix of regressors
    resp    - time x feature/region matrix of targets (y-values)
    """
    print '\tRunning PCA'
    os.environ['OMP_NUM_THREADS'] = str(nproc)
    if stim.shape[1]<n_components:
        n_components = stim.shape[1]
    pca = PCA(n_components)
    reduced_mat = pca.fit_transform(stim) # Time X Features
    components = pca.components_

    print '\tRunning regression'
    betas, resid = regression(resp,reduced_mat,alpha=0,constant=False) # betas are components x targets

    ## Remove coliders
    # Identify pair-wise covariance matrix
    cov_mat = np.dot(reduced_mat.T, resp)
    # Identify postive weights with also postive cov
    pos_mat = np.multiply(cov_mat>0,betas>0)
    # Identify negative weights with also negative cov
    neg_mat = np.multiply(cov_mat<0,betas<0)
    # Now identify both positive and negative weights
    pos_weights = np.multiply(pos_mat,betas)
    neg_weights = np.multiply(neg_mat,betas)
    fc_mat = pos_weights + neg_weights

    # Now map back into physical vertex space
    # Dimensions: Source X Target vertices
    #fc_mat = np.dot(fc_mat.T,components).T

    return fc_mat,components


def multregressionconnectivity(activityMatrix):
    """
    Activity matrix should be region/voxel X time
    Assumes all time series are de-meaned
    """

    nregions = activityMatrix.shape[0]
    timepoints = activityMatrix.shape[1]
    if nregions > timepoints:
         raise Exception('More regions (regressors) than timepoints! Use regularized regression')

    interaction_mat = np.zeros((nregions,nregions))

    for targetregion in range(nregions):
        otherregions = range(nregions)
        otherregions = np.delete(otherregions, targetregion) # Delete target region from 'other regions'
        X = activityMatrix[otherregions,:].T
        # Add 'constant' regressor
        y = activityMatrix[targetregion,:]
        betas, resid = regression(y,X,constant=True)
        interaction_mat[otherregions, targetregion] = betas[1:] # all betas except for constant betas

    return interaction_mat

def regression(data,regressors,alpha=0,constant=True):
    """
    Taku Ito
    2/21/2019

    Hand coded OLS regression using closed form equation: betas = (X'X + alpha*I)^(-1) X'y
    Set alpha = 0 for regular OLS.
    Set alpha > 0 for ridge penalty

    PARAMETERS:
        data = observation x feature matrix (e.g., time x regions)
        regressors = observation x feature matrix
        alpha = regularization term. 0 for regular multiple regression. >0 for ridge penalty
        constant = True/False - pad regressors with 1s?

    OUTPUT
        betas = coefficients X n target variables
        resid = observations X n target variables
    """
    # Add 'constant' regressor
    if constant:
        ones = np.ones((regressors.shape[0],1))
        regressors = np.hstack((ones,regressors))
    X = regressors.copy()

    # construct regularization term
    LAMBDA = np.identity(X.shape[1])*alpha

    # Least squares minimization
    try:
        C_ss_inv = np.linalg.pinv(np.dot(X.T,X) + LAMBDA)
    except np.linalg.LinAlgError as err:
        C_ss_inv = np.linalg.pinv(np.cov(X.T) + LAMBDA)

    betas = np.dot(C_ss_inv,np.dot(X.T,data))
    # Calculate residuals
    resid = data - (betas[0] + np.dot(X[:,1:],betas[1:]))

    return betas, resid
