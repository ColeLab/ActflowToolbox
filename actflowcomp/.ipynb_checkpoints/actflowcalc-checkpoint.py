#Define activity flow mapping function

import numpy as np

def actflowcalc(actVect, fcMat, separate_activations_bytarget=False):
    """
    Function to run activity flow mapping algorithm
    
    actVect: node vector with activation values
    fcMat: node x node matrix with connectiivty values
    separate_activations_bytarget: indicates if the input actVect matrix has a separate activation vector for each target (to-be-predicted) node (e.g., for the locally-non-circular approach)
    """
    
    numRegions=len(actVect)
    actPredVector=np.zeros((numRegions,))
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
