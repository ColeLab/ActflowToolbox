import os
import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso,lasso_path

# Kirsten Peterson, Jan. 2024

# Peterson, K. L., Sanchez-Romero, R., Mill, R. D., & Cole, M. W. (2023). Regularized partial correlation provides reliable functional connectivity estimates while correcting for widespread confounding. In bioRxiv (p. 2023.09.16.558065). https://doi.org/10.1101/2023.09.16.558065

def lassoCV(data, L1s=None, kFolds=10, optMethod='R2', saveFiles=0, outDir='', foldsToRun='all', targetNodesToRun='all',  nodewiseHyperparams=True): 
    '''
    Runs node-wise lasso regression to compute the L1-regularized multiple regression FC matrix of a dataset, using cross validation to select the optimal L1 hyperparameters. Currectly, model performance for each hyperparameter value is scored as the R^2 between held-out activity predicted for each node using the training data connectivity matrix and the actual held-out activity. 
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1s : list of L1 (lambda1) hyperparameter values to test (all values must be >0); scales the lasso penalty term; the default L1s may not be suitable for all datasets and are only intended as a starting point
        kFolds : number of cross-validation folds to use during hyperparameter selection; an FC model is fit k times, each time using all folds but one; we recommend using a higher number (e.g., at least 10) so that the number of timepoints being fit is close to that of the full dataset (e.g., at least 90% as many timepoints), since the optimal hyperparameter value depends on data quantity
        optMethod : method for choosing the optimal hyperparameter (only 'R2' currently available), or None to skip
        saveFiles : whether to save intermediate and output variables to .npy files (0 = no; 1 = savefinal optimal FC matrix, optimal hyperparameter value, and model fit metrics (i.e., R^2) for each hyperparameter and fold; 2 = save CV fold connectivity coefficients too)
        outDir : if saveFiles>=1, the directory in which to save the files
        foldsToRun : list of CV folds (0 to kFolds-1) for which to compute model coefficients during hyperparameter selection; default ('all') is to run for all kFolds folds, but running only a subset of folds can reduce computation
        targetNodesToRun : list of target nodes (0 to nNodes-1) for which to fit an FC model (i.e., rows in the FC matrix), with all variables in the provided dataset used as source nodes (i.e., columns in the FC matrix); default ('all') is to use all nodes in the dataset as both targets and sources, to construct an [nNodes x nNodes] FC matrix
        nodewiseHyperparams : whether to select a different optimal hyperparameter for each target node's regression model
        
    OUTPUT:
        FC : lasso connectivity matrix, using optimal L1 value from list of L1s
        cvResults : dictionary with 'bestParam' for the optimal L1 value, 'L1s' for the list of tested L1 values, and 'R2' containing scores for every L1 value and cross-validation fold
    '''
    
    if L1s is None:
        # (log scale from 0.316 to 0.001)
        L1s = np.arange(-.5,-3.1,-.1)
        L1s = np.round(10**L1s,6)
        # We'd recommend checking the optimal hyperparameters for a few subjects and then narrowing down the range
    else:
        L1s = np.array(L1s)
    
    if not ((optMethod == 'R2') or (optMethod == None)):
        raise ValueError(f'optMethod "{optMethod}" does not match available methods. Available options are "R2" or None (compute CV matrices but skip model selection).')
    
    if isinstance(foldsToRun,str) and (foldsToRun == 'all'):
        foldsToRun = np.arange(kFolds)
    else:
        foldsToRun = np.array(foldsToRun)
        
    # Divide timepoints into folds
    nTRs = data.shape[1]
    kFoldsTRs = np.full((kFolds,nTRs),False)     
    k = 0
    for t in range(nTRs): # (interleaving folds)
        kFoldsTRs[k,t] = True
        k += 1
        if k >= kFolds:
            k = 0
    
    # Define arrays to hold performance metrics
    nNodes = data.shape[0]
    if nodewiseHyperparams:
        scores = np.full((len(L1s),kFolds,nNodes),np.nan)
    else:
        scores = np.full((len(L1s),kFolds,1),np.nan)
    
    # If saving intermediate files
    if saveFiles >= 1:
        
        # Where to save files
        if outDir == '':
            outDir = os.getcwd()
            print(f'Directory for output files not provided, saving to current directory:\n{outDir}')
        if not os.path.exists(outDir):
            os.mkdir(outDir)
            
        outfileCVModel = {}
        outfileCVScores = {}
        
        # Loop through folds
        for k in foldsToRun:
            outfileCVModel[k] = {}
            outfileCVScores[k] = {}
            
            # Loop through L1s - find which need to be run
            L1sToRun = np.full(L1s.shape,True)
            for l,L1 in enumerate(L1s):
                outfileCVModel[k][L1] = f'{outDir}/L1-{L1}_kFolds-{k+1}of{kFolds}.npy'
                if nodewiseHyperparams:
                    outfileCVScores[k][L1] = f'{outDir}/L1-{L1}_kFolds-{k+1}of{kFolds}_{optMethod}_nodewise.npy'
                else:
                    outfileCVScores[k][L1] = f'{outDir}/L1-{L1}_kFolds-{k+1}of{kFolds}_{optMethod}.npy'
            
                # If performance metrics were already saved for this fold and L1, load and move on
                if os.path.exists(outfileCVScores[k][L1]):
                    scores[l,k] = np.load(outfileCVScores[k][L1])
                    L1sToRun[l] = False
                    
                # If FC matrix was saved for this fold and L1, calculate performance metric and move on
                elif os.path.exists(outfileCVModel[k][L1]):
                    lassoCV = np.load(outfileCVModel[k][L1])
                    if optMethod == 'R2':
                        r2,r = activityPrediction(data[:,kFoldsTRs[k]],lassoCV,nodewise=nodewiseHyperparams)
                        scores[l,k] = r2.squeeze()
                        np.save(outfileCVScores[k][L1],scores[l,k])
                    L1sToRun[l] = False
                    
            # Estimate the lasso FC matrices for missing L1s
            if np.sum(L1sToRun) > 0:
                lassoCV = lasso(data[:,~kFoldsTRs[k]],L1s[L1sToRun],targetNodesToRun=targetNodesToRun)
                    
                # Save CV matrices if saveFiles = 2
                if saveFiles == 2:
                    for l,L1 in enumerate(L1s[L1sToRun]):
                        np.save(outfileCVModel[k][L1],lassoCV[:,:,l].squeeze())
                      
                if optMethod == 'R2':
                    # Calculate R^2
                    r2,r = activityPrediction(np.repeat(data[:,kFoldsTRs[k],np.newaxis],lassoCV.shape[2],axis=2),lassoCV,nodewise=nodewiseHyperparams)
                    print(r2.shape,r2.T.shape,scores[L1sToRun,k].shape,scores[:,k].shape,scores[0,k].shape)
                    scores[L1sToRun,k] = r2.T
                
                # Save performance metrics
                for l,L1 in enumerate(L1s[L1sToRun]):
                    np.save(outfileCVScores[k][L1],r2[:,l].squeeze())
    
    # If not saving intermediate files
    else:
        # Loop through folds
        for k in range(kFolds):
            # Estimate the lasso FC matrices
            lassoCV = lasso(data[:,~kFoldsTRs[k]],L1s,targetNodesToRun=targetNodesToRun)
                
            if optMethod == 'R2':
                # Calculate R^2
                r2,r = activityPrediction(np.repeat(data[:,kFoldsTRs[k],np.newaxis],lassoCV.shape[2],axis=2),lassoCV,nodewise=nodewiseHyperparams)
                scores[:,k] = r2.T 
    
    # Find the best param according to each performance metric 
    scoresKFoldMean = np.nanmean(scores,axis=1)
    if optMethod == 'R2':
        bestParam = L1s.T[np.argmax(scoresKFoldMean,axis=0)]
    
    # Estimate the regularized partial correlation using all data and the optimal hyperparameters
    lassoOpt = lasso(data,bestParam,targetNodesToRun=targetNodesToRun,nodewiseHyperparams=nodewiseHyperparams).squeeze()
    
    if saveFiles >= .5:
        if nodewiseHyperparams:
            np.save(f'{outDir}/bestL1_opt-{optMethod}_nodewise.npy',bestParam)
            np.save(f'{outDir}/lasso_opt-{optMethod}_nodewise.npy',lassoOpt)
        else:
            np.save(f'{outDir}/bestL1_opt-{optMethod}.npy',bestParam)
            np.save(f'{outDir}/lasso_opt-{optMethod}.npy',lassoOpt)
    
    cvResults = {'bestParam': bestParam, optMethod: scores, 'L1s': L1s}
    
    return lassoOpt,cvResults

#-----
def lasso(data,L1s,zscore=True,targetNodesToRun='all',nodewiseHyperparams=False):
    '''
    Calculates the L1-regularized regression coefficients of a dataset. Runs sklearn's Lasso function and several other necessary steps.
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1 : L1 (lambda1) hyperparameter value; single value for all target nodes, or list with a value for each node
        zscore : whether to zscore the input data, default True
        targetNodesToRun : 'all', or subset to use as target nodes 
        nodewiseHyperparams : whether to use different hyperparameter for each target node's regression model
    
    OUTPUT:
        conn : regularized multiple regression coefficients (i.e., the FC matrix)
    '''
    
    nNodes = data.shape[0]
    if isinstance(targetNodesToRun,str) and (targetNodesToRun == 'all'):
        targetNodesToRun = np.arange(nNodes)
    else:
        targetNodesToRun = np.array(targetNodesToRun)
    
    # Check L1s
    if nodewiseHyperparams:
        if not (isinstance(L1s,list) or isinstance(L1s,np.ndarray)):
            raise ValueError(f'If nodewiseHyperparams=True, then L1 must be a list or np.ndarray. L1 is {type(L1)}.')
        elif not (len(L1s)==nNodes):
            raise ValueError(f'If nodewiseHyperparams=True, then L1 must contain as many elements as there are nodes (data dimension 0). L1 has length {len(L1)} while data has shape {data.shape}.')
        L1sByNode = np.array(L1s).copy()
        L1s = np.array([L1sByNode[0]])
    else:
        if (isinstance(L1s,float) or isinstance(L1s,int)):
            L1s = [L1s]
        L1s = np.array(L1s)
            
    # Pre-allocate FC matrix        
    conn = np.full((nNodes,nNodes,len(L1s)),np.nan)
    
    # Z-score the data
    # (recommended when applying regularization)
    if zscore:
        data = stats.zscore(data,axis=1)
    
    # Loop through target nodes (only nodes to keep, i.e. cortex only)
    allNodes = np.arange(nNodes)
    for target in targetNodesToRun:
        sources = allNodes[allNodes!=target] # include all nodes in model (i.e. cortex and subcortex)
        
        X = data[sources,:].copy()
        y = data[target,:].copy()
            
        # Get node-specific hyperparameter
        if nodewiseHyperparams: 
            L1s = np.array([L1sByNode[target]])
            
        # Run the regression for target node
        _,coefs,_ = lasso_path(X.T,y,alphas=L1s)
        
        # Fill in coefficients
        conn[target,sources] = coefs
    
    return conn

#-----
def activityPrediction(activity,conn,zscore=True,nodewise=False):
    '''
    Uses a functional connectivity matrix to predict the (held-out) activity of each node from the activities of all other nodes. Returns R^2 and Pearson's r as measures of the similarity between predicted and actual timeseries, with higher similarity indicating a more accurate connectivity model.
    INPUT:
        activity : held-out timeseries ([nNodes x nDatapoints], or [nNodes x Datapoints x nSubjects])
        conn : connectivity matrix being tested
        zscore : whether to zscore the input activity data, default True
        nodewise : whether to calculate separate scores for each node
    OUTPUT:
        R2 : the Coefficient of Determination (R^2) between predicted and actual activity
        pearson : the Pearson correlation (r) between predicted and actual activity
    '''
    if zscore:
        activity = stats.zscore(activity,axis=1)
    
    if activity.ndim == 2:
        activity = activity[:,:,np.newaxis]
    if conn.ndim == 2:
        conn = conn[:,:,np.newaxis]
    
    nSubjs = activity.shape[2]
    nNodes = activity.shape[0]
    
    # Predict the activities of each node (j) from the activities of all other nodes (i)
    # prediction_j = sum(activity_i * connectivity_ij)
    prediction = np.full((activity.shape),np.nan)
    nodesList = np.arange(nNodes)
    for n in nodesList:
        otherNodes = nodesList!=n
        for s in range(nSubjs):
            X = activity[otherNodes,:,s]
            betas = conn[n,otherNodes,s]
            yPred = np.sum(X*betas[:,np.newaxis],axis=0)
            prediction[n,:,s] = yPred
        
    # Calculate R^2 and Pearson's r between the actual and predicted timeseries
    if nodewise:
        sumSqrReg = np.sum((activity-prediction)**2,axis=1)
        sumSqrTot = np.sum((activity-np.mean(np.mean(activity,axis=1),axis=0))**2,axis=1)
        R2 = 1 - (sumSqrReg/sumSqrTot)
          
        pearson = np.full((nNodes,nSubjs),np.nan)
        for s in range(nSubjs):    
            for n in range(nNodes):
                pearson[n,s] = np.corrcoef(activity[n,:,s],prediction[n,:,s])[0,1]

    else:
        sumSqrReg = np.sum(np.sum((activity-prediction)**2,axis=1),axis=0)
        sumSqrTot = np.sum(np.sum((activity-np.mean(np.mean(activity,axis=1),axis=0))**2,axis=1),axis=0)
        R2 = 1 - (sumSqrReg/sumSqrTot)
        R2 = np.reshape(R2,(1,nSubjs))
        
        pearson = np.full((1,nSubjs),np.nan)
        for s in range(nSubjs):
            pearson[0,s] = np.corrcoef(activity[:,:,s].flatten(),prediction[:,:,s].flatten())[0,1]
    
    return R2,pearson
