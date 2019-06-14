
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np

def pc_multregconn(activity_matrix, target_ts=None, n_components=None):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    n_components:  Optional. Number of PCA components to use. If None, the smaller of number of nodes or number of time points (minus 1) will be selected.
    Output: connectivity_mat, formatted targets X sources
    """

    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]
    
    if n_components == None:
        n_components = np.min([nnodes-1, timepoints-1])
    else:
        if nnodes<n_components or timepoints<n_components:
            print('activity_matrix shape: ',np.shape(activity_matrix))
            raise Exception('More components than nodes and/or timepoints! Use fewer components')
            
    pca = PCA(n_components)

    if target_ts is None:
        connectivity_mat = np.zeros((nnodes,nnodes))
        for targetnode in range(nnodes):
            othernodes = list(range(nnodes))
            othernodes.remove(targetnode) # Remove target node from 'other nodes'
            X = activity_matrix[othernodes,:].T
            y = activity_matrix[targetnode,:]
            #Run PCA on source time series
            reduced_mat = pca.fit_transform(X) # Time X Features
            components = pca.components_
            #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
            reg = LinearRegression().fit(reduced_mat, y)
            #Convert regression betas from component space to node space
            betasPCR = pca.inverse_transform(reg.coef_)
            connectivity_mat[targetnode,othernodes]=betasPCR
    else:
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,1))
        X = activity_matrix.T
        y = target_ts
        #Run PCA on source time series
        reduced_mat = pca.fit_transform(X) # Time X Features
        components = pca.components_
        #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(reduced_mat, y)
        #Convert regression betas from component space to node space
        betasPCR = pca.inverse_transform(reg.coef_)
        connectivity_mat=betasPCR

    return connectivity_mat
