
from sklearn.linear_model import LinearRegression
#from ..tools import regression
import numpy as np

def multregconn(activity_matrix, target_ts=None, parcelstoexclude_bytarget=None, conn_mask=None):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    parcelstoexclude_bytarget: Optional. A dictionary of lists, each listing parcels to exclude for each target parcel (e.g., to reduce potential circularity by removing parcels near the target parcel). Note: This is not used if target_ts is set.
    conn_mask: Optional. Specifies a mask to exclude some connections from being fit (setting them to 0). Consists of a matrix of 1s and 0s, where 1s indicate a connection and 0s indicate no connection. If target_ts=None then it is a N X N matrix (where N=number of nodes), otherwise it is N X 1.
    Output: connectivity_mat, formatted targets X sources
    """

    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]
    if nnodes > timepoints:
        print('activity_matrix shape: ',np.shape(activity_matrix))
        raise Exception('More nodes (regressors) than timepoints! Use regularized regression')

    if target_ts is None:
        connectivity_mat = np.zeros((nnodes,nnodes))
        for targetnode in range(nnodes):
            othernodes = list(range(nnodes))
            #Remove parcelstoexclude_bytarget parcels (if flagged); parcelstoexclude_bytarget is by index (not parcel value)
            if parcelstoexclude_bytarget is not None:
                parcelstoexclude_thisnode=parcelstoexclude_bytarget[targetnode].tolist()
                parcelstoexclude_thisnode.append(targetnode) # Remove target node from 'other nodes'
                othernodes = list(set(othernodes).difference(set(parcelstoexclude_thisnode)))
            elif conn_mask is not None:
                othernodes = list(set(othernodes).difference(set((conn_mask[targetnode,:]<1).nonzero()[0])))
            else:
                othernodes.remove(targetnode) # Remove target node from 'other nodes'
            X = activity_matrix[othernodes,:].T
            y = activity_matrix[targetnode,:]
            #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
            if len(othernodes)>0:
                reg = LinearRegression().fit(X, y)
                connectivity_mat[targetnode,othernodes]=reg.coef_
            # run multiple regression, and add constant
            #beta_fc,resids = regression.regression(y,X,alpha=0, constant=True) # increase alpha if want to apply a ridge penalty
            #connectivity_mat[targetnode,othernodes] = beta_fc[1:] # exclude 1st coef; first coef is beta_0 (or mean)
    else:
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,1))
        X = activity_matrix.T
        y = target_ts
        #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(X, y)
        connectivity_mat=reg.coef_
        # run multiple regression, and add constant
        #beta_fc,resids = regression.regression(y,X,alpha=0, constant=True) # increase alpha if want to apply a ridge penalty
        #connectivity_mat = beta_fc[1:] # exclude 1st coef; first coef is beta_0 (or mean)

    return connectivity_mat



def logit(x,a=1):
    return (1/a)*np.log(x/(1-x))
