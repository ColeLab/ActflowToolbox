
#from sklearn.linear_model import LinearRegression
from ..tools import regression
import numpy as np

def multregconn(activity_matrix, target_ts):
	"""
	activity_matrix:	Activity matrix should be nodes X time
	target_ts: 			Optional, used when only a single target time series (returns 1 X nnodes matrix)
	Output: connectivity_mat, formatted targets X sources
	"""

	nnodes = activity_matrix.shape[0]
	timepoints = activity_matrix.shape[1]
	if nnodes > timepoints:
		 raise Exception('More nodes (regressors) than timepoints! Use regularized regression')

	if target_ts is None:
		connectivity_mat = np.zeros((nnodes,nnodes))
		for targetnode in range(nnodes):
			othernodes = range(nnodes)
			othernodes.remove(targetnode) # Remove target node from 'other nodes'
			X = activity_matrix[othernodes,:].T
			y = activity_matrix[targetnode,:]
			#Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
			#reg = LinearRegression().fit(X, y)
			#connectivity_mat[nodeNum,otherNodes]=reg.coef_
			# run multiple regression, and add constant
			beta_fc = regression.regression(y,X,alpha=0, constant=True) # increase alpha if want to apply a ridge penalty
			connectivity_mat[nodeNum,otherNodes] = beta_fc[1:] # exclude 1st coef; first coef is beta_0 (or mean)
	else:
		#Computing values for a single target node
		connectivity_mat = np.zeros((nnodes,1))
		X = activity_matrix.T
		y = target_ts.T
		#Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
		#reg = LinearRegression().fit(X, y)
		#connectivity_mat=reg.coef_
		# run multiple regression, and add constant
		beta_fc = regression.regression(y,X,alpha=0, constant=True) # increase alpha if want to apply a ridge penalty
		connectivity_mat = beta_fc[1:] # exclude 1st coef; first coef is beta_0 (or mean)

	return connectivity_mat
