import numpy as np

def actflow_sourcetarget(actvect_sources, nodelist_sources, nodelist_targets, fc_mat, separate_activations_bytarget=False, transfer_func=None):
    """

    Function to run activity flow mapping algorithm
    
    actvect_sources: node vector with activation values for source regions
    nodelist_sources: list of source regions
    nodelist_targets: list of target regions
    fcMat: target nodes x source node matrix with connectivity values
    separate_activations_bytarget: indicates if the input actVect matrix has a separate activation vector for each target (to-be-predicted) node (e.g., for the locally-non-circular approach)
    transfer_func: The transfer function to apply to the outputs of all source regions. Assumes observed time series are primarily driven by inputs (e.g., local field potentials), such that the source time series need to be converted from inputs to outputs via a transfer function. Default is 'None', which specifies a linear transfer function wherein the output is the same as the input.
    """
    
    numSourceNodes = len(nodelist_sources)
    numTargetNodes = len(nodelist_targets)
    actPredVector  = np.zeros((numTargetNodes,))
    sourceRegions  = list(range(numSourceNodes))

    if transfer_func is None:
        
        if separate_activations_bytarget:
            for targetRegion in range(numTargetNodes):
                actPredVector[targetRegion]=np.sum(actvect_sources[targetRegion,sourceRegions]*fc_mat[targetRegion,sourceRegions])
        else:
            for targetRegion in range(numTargetNodes):
                actPredVector[targetRegion]=np.sum(actvect_sources[sourceRegions]*fc_mat[targetRegion,sourceRegions])
        return actPredVector
    
    else:
        
        if separate_activations_bytarget:

            for targetRegion in range(numTargetNodes):
                
                inputActVect=transfer_function(actvect_sources[targetRegion,sourceRegions],transfer_func=transfer_func)
                actPredVector[targetRegion]=np.sum(inputActVect*fc_mat[targetRegion,sourceRegions])
        else:
            for targetRegion in range(numTargetNodes):
                
                inputActVect=transfer_function(actvect_sources[sourceRegions],transfer_func=transfer_func)
                actPredVector[targetRegion]=np.sum(inputActVect*fc_mat[targetRegion,sourceRegions])
        return actPredVector


#Define input transfer function
def transfer_function(activity, transfer_func='linear', threshold=0, a=1):
    if transfer_func == 'linear':
        return activity
    elif transfer_func == 'relu':
        return activity*(activity>threshold)
    elif transfer_func == 'sigmoid':
        return 1 / (1 + np.exp(-activity))
    elif transfer_func == 'logit':
        return (1/a)*np.log(activity/(1-activity))
    
