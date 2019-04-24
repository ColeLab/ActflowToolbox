#Define activity flow mapping function

import numpy as np

def actflowcalc(actVect, fcMat):
    numRegions=len(actVect)
    actPredVector=np.zeros((numRegions,))
    for heldOutRegion in range(numRegions):
        otherRegions=list(range(numRegions))
        otherRegions.remove(heldOutRegion)
        actPredVector[heldOutRegion]=np.sum(actVect[otherRegions]*fcMat[heldOutRegion,otherRegions])
    return actPredVector
