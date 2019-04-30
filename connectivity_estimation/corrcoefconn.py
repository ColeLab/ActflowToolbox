
from sklearn.linear_model import LinearRegression
#from ..tools import regression
import numpy as np

def corrcoefconn(activity_matrix, target_ts=None):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    Output: connectivity_mat, formatted targets X sources, with diagonal set to 0
    """

    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]

    if target_ts is None:
        connectivity_mat = np.corrcoef(activity_matrix,rowvar=1)
        #Set FC matrix diagonal to 0s
        np.fill_diagonal(connectivity_mat,0)
    else:
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,))

        #Compute correlation for each pair
        for source_node in range(nnodes):
            connectivity_mat[source_node] = np.corrcoef(activity_matrix[source_node,:],target_ts)[0,1]

    return connectivity_mat
