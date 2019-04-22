
from sklearn.linear_model import LinearRegression

def multregressionconnectivity(activityMatrix):
    """
    Activity matrix should be region/voxel X time
	Output: connectivity_mat, formatted targets X sources
    """

    nregions = activityMatrix.shape[0]
    timepoints = activityMatrix.shape[1]
    if nregions > timepoints:
         raise Exception('More regions (regressors) than timepoints! Use regularized regression')

    connectivity_mat = np.zeros((nregions,nregions))

    for targetregion in range(nregions):
        otherregions = range(nregions)
        otherregions.remove(targetregion) # Remove target region from 'other regions'
        X = activityMatrix[otherregions,:].T
        y = activityMatrix[targetregion,:]
		#Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
		reg = LinearRegression().fit(X, y)
		connectivity_mat[nodeNum,otherNodes]=reg.coef_
        #betas, resid = regression(y,X,constant=True)
        #interaction_mat[otherregions, targetregion] = betas[1:] # all betas except for constant betas

    return connectivity_mat
