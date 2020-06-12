
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def pc_multregconn(activity_matrix, target_ts=None, n_components=None, n_comp_search=False, n_components_min=1, n_components_max=None, n_comp_search_cvs=5, svd_solver='auto'):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    n_components:  Optional. Number of PCA components to use. If None, the smaller of number of nodes or number of time points (minus 1) will be selected.
    n_comp_search: Optional. Boolean indicating whether to search for the best number of components based on cross-validation generalization (to reduce overfitting).
    n_components_min: Optional. The smallest number to test in the n_comp_search.
    n_components_max: Optional. The largest number to test in the n_comp_search.
    n_comp_search_cvs: Optional. The number of cross-validation folds to use for the n_comp_search. More folds will take longer, but likely improve generalization accuracy. Note that the time series are also split into the number of cross-validation folds, such that nearby time points are unlikely to be included in both training and testing sets. This is an attempt to reduce the influence of temporal autocorrelation. Thus, a very large number of folds may reduce generalization accuracy (due to the data being split into smaller splits in the time series). Ideally for fMRI data each split would be about 100 seconds or longer (given known autocorrelation as long as 0.01 Hz).
    
    If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
    
    Output: connectivity_mat (formatted targets X sources), n_components
    """

    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]
    
    if n_components == None:
        n_components = np.min([nnodes-1, timepoints-1])
    else:
        if nnodes<n_components or timepoints<n_components:
            print('activity_matrix shape: ',np.shape(activity_matrix))
            raise Exception('More components than nodes and/or timepoints! Use fewer components')
    
    if n_comp_search:
        if n_components_max == None:
            n_components_max = np.min([nnodes-1, timepoints-1])
        else:
            if nnodes<n_components_max or timepoints<n_components_max:
                print('activity_matrix shape: ',np.shape(activity_matrix))
                raise Exception('More components than nodes and/or timepoints! Use fewer components')
    
    #De-mean time series
    activity_matrix_mean = np.mean(activity_matrix,axis=1)
    activity_matrix = activity_matrix - activity_matrix_mean[:, np.newaxis]
    
    if target_ts is None:
        
        #Cross-validation to find optimal number of components (based on mean MSE across all nodes)
        if n_comp_search:
            if n_components_max is None:
                n_components_max = np.min([nnodes-1, timepoints-1])
            componentnum_set=np.arange(n_components_min,n_components_max+1)
            mse_regionbycomp = np.zeros([np.shape(componentnum_set)[0],nnodes])
            for targetnode in range(nnodes):
                othernodes = list(range(nnodes))
                othernodes.remove(targetnode) # Remove target node from 'other nodes'
                X = activity_matrix[othernodes,:].T
                y = activity_matrix[targetnode,:]
                #Run PCA
                pca = PCA(svd_solver=svd_solver)
                Xreg_allPCs = pca.fit_transform(X)
                mscv_vals=np.zeros(np.shape(componentnum_set)[0])
                comp_count=0
                for comp_num in componentnum_set:
                    regr = LinearRegression()
                    Xreg = Xreg_allPCs[:,:comp_num]
                    regr.fit(Xreg, y)
                    #Split into num_cvs equal groups (helping account for temporal autocorrelation)
                    num_cvs=n_comp_search_cvs
                    num_values_per_group=np.floor(np.shape(y)[0]/num_cvs)
                    group_labels=np.zeros((np.shape(y)[0]))
                    for group_num in np.arange(num_cvs):
                        start=group_num*num_values_per_group
                        end=(group_num+1)*num_values_per_group-1
                        group_labels[start:end]=group_num
                    group_kfold = GroupKFold(n_splits=num_cvs)
                    # Cross-validation                  
                    y_cv = cross_val_predict(regr, Xreg, y, groups=group_labels, cv=group_kfold)
                    mscv_vals[comp_count] = mean_squared_error(y, y_cv)
                    comp_count=comp_count+1
                mse_regionbycomp[:,targetnode] = mscv_vals
            min_comps_means = np.mean(mse_regionbycomp, axis=1)
            n_components=componentnum_set[np.where(min_comps_means==np.min(min_comps_means))[0][0]]
            print('n_components = ' + str(n_components))
        
        connectivity_mat = np.zeros((nnodes,nnodes))
        for targetnode in range(nnodes):
            othernodes = list(range(nnodes))
            othernodes.remove(targetnode) # Remove target node from 'other nodes'
            X = activity_matrix[othernodes,:].T
            y = activity_matrix[targetnode,:]
            #Run PCA on source time series
            pca = PCA(n_components, svd_solver=svd_solver)
            reduced_mat = pca.fit_transform(X) # Time X Features
            components = pca.components_
            #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
            regrmodel = LinearRegression()
            reg = regrmodel.fit(reduced_mat, y)
            #Convert regression betas from component space to node space
            betasPCR = pca.inverse_transform(reg.coef_)
            connectivity_mat[targetnode,othernodes]=betasPCR
    else:
        #Remove time series mean
        target_ts = target_ts - np.mean(target_ts)
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,1))
        X = activity_matrix.T
        y = target_ts
        #Cross-validation to find optimal number of components
        if n_comp_search:
            componentnum_set=np.arange(n_components_min,n_components_max+1)
            mscv_vals=np.zeros(np.shape(componentnum_set)[0])
            comp_count=0
            for comp_num in componentnum_set:
                mscv_vals[comp_count] = pcr_cvtest(X,y, pc=comp_num, svd_solver=svd_solver)
                comp_count=comp_count+1
            n_components=componentnum_set[np.where(mscv_vals==np.min(mscv_vals))[0][0]]
            print('n_components = ' + str(n_components))
        #Run PCA on source time series
        pca = PCA(n_components, svd_solver=svd_solver)
        reduced_mat = pca.fit_transform(X) # Time X Features
        components = pca.components_
        #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(reduced_mat, y)
        #Convert regression betas from component space to node space
        betasPCR = pca.inverse_transform(reg.coef_)
        connectivity_mat=betasPCR

    return connectivity_mat


def pcr_cvtest(X,y,pc,num_cvs=5,svd_solver='auto'):
    ''' Principal Component Regression in Python'''
    ''' Based on code from here: https://nirpyresearch.com/principal-component-regression-python/'''
    
    ''' Step 1: PCA on input data'''
    # Define the PCA object
    pca = PCA(svd_solver=svd_solver)
    # Run PCA producing the reduced variable Xred and select the first pc components
    Xreg = pca.fit_transform(X)[:,:pc]
    ''' Step 2: regression on selected principal components'''
    # Create linear regression object
    regr = LinearRegression()
    # Fit
    regr.fit(Xreg, y)
    # Calibration
    #y_c = regr.predict(Xreg)
    #Split into num_cvs equal groups (helping account for temporal autocorrelation)
    num_values_per_group=np.floor(np.shape(y)[0]/num_cvs)
    group_labels=np.zeros((np.shape(y)[0]))
    for group_num in np.arange(num_cvs):
        start=int(group_num*num_values_per_group)
        end=int((group_num+1)*num_values_per_group-1)
        group_labels[start:end]=group_num
    group_kfold = GroupKFold(n_splits=num_cvs)
    # Cross-validation
    y_cv = cross_val_predict(regr, Xreg, y, groups=group_labels, cv=group_kfold)
    #y_cv = cross_val_predict(regr, Xreg, y, cv=num_cvs)
    # Calculate mean square error for calibration and cross validation
    mse_cv = mean_squared_error(y, y_cv)
    
    #return(y_cv, mse_c, mse_cv)
    return(mse_cv)