# Taku Ito
# 2/21/2019

# Multiple linear regression (with L2 regularization option)


import numpy as np

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
    C_ss_inv = np.linalg.pinv(np.matmul(X.T,X) + LAMBDA)
    
    betas = np.dot(C_ss_inv,np.matmul(X.T,data))
    # Calculate residuals
    resid = data - (betas[0] + np.matmul(X[:,1:],betas[1:]))

    # Remove imaginary portion (will be all 0s anyway)
    betas = betas.real
    resid = resid.real

    return betas, resid
