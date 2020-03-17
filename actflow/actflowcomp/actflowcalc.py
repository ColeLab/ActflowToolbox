#Define activity flow mapping function

import numpy as np

def actflowcalc(actVect, fcMat, separate_activations_bytarget=False, transfer_func=None):
    """
    Function to run activity flow mapping algorithm
    
    actVect: node vector with activation values
    fcMat: node x node matrix with connectiivty values
    separate_activations_bytarget: indicates if the input actVect matrix has a separate activation vector for each target (to-be-predicted) node (e.g., for the locally-non-circular approach)
    transfer_func: The transfer function to apply to the outputs of all source regions. Assumes observed time series are primarily driven by inputs (e.g., local field potentials), such that the source time series need to be converted from inputs to outputs via a transfer function. Default is 'None', which specifies a linear transfer function wherein the output is the same as the input.
    """
    
    numRegions=np.shape(actVect)[0]
    actPredVector=np.zeros((numRegions,))
    
    if transfer_func is None:
        
        if separate_activations_bytarget:
            for heldOutRegion in range(numRegions):
                otherRegions=list(range(numRegions))
                otherRegions.remove(heldOutRegion)
                actPredVector[heldOutRegion]=np.sum(actVect[heldOutRegion,otherRegions]*fcMat[heldOutRegion,otherRegions])
        else:
            for heldOutRegion in range(numRegions):
                otherRegions=list(range(numRegions))
                otherRegions.remove(heldOutRegion)
                actPredVector[heldOutRegion]=np.sum(actVect[otherRegions]*fcMat[heldOutRegion,otherRegions])
        return actPredVector
    
    else:
        
        if separate_activations_bytarget:
            for heldOutRegion in range(numRegions):
                otherRegions=list(range(numRegions))
                otherRegions.remove(heldOutRegion)
                inputActVect=transfer_function(actVect[heldOutRegion,otherRegions],transfer_func=transfer_func)
                actPredVector[heldOutRegion]=np.sum(inputActVect*fcMat[heldOutRegion,otherRegions])
        else:
            for heldOutRegion in range(numRegions):
                otherRegions=list(range(numRegions))
                otherRegions.remove(heldOutRegion)
                inputActVect=transfer_function(actVect[otherRegions],transfer_func=transfer_func)
                actPredVector[heldOutRegion]=np.sum(inputActVect*fcMat[heldOutRegion,otherRegions])
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