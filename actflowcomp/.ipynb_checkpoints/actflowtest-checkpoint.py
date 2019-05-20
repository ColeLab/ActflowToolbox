
import numpy as np
from scipy import stats
from .actflowcalc import *

def actflowtest(actVect_group, fcMat_group, separate_activations_bytarget=False):
    """
    Function to run activity flow mapping with spatial correlation predicted-to-actual testing across multiple tasks and subjects, either with a single (e.g., rest) connectivity matrix or with a separate connectivity matrix for each task. Returns stats at the group level.
    
    actVect_group: node x condition x subject matrix with activation values
    fcMat_group: node x node x condition x subject matrix (or node x node x subject matrix) with connectiivty values
    separate_activations_bytarget: indicates if the input actVect_group matrix has a separate activation vector for each target (to-be-predicted) node (e.g., for the locally-non-circular approach)
    """
    
    #Add empty task dimension if only 1 task
    if separate_activations_bytarget:
        if len(np.shape(actVect_group)) < 4:
            actVect_group=np.reshape(actVect_group,(np.shape(actVect_group)[0],np.shape(actVect_group)[1],1,np.shape(actVect_group)[2]))
        nTasks=np.shape(actVect_group)[2]
        nSubjs=np.shape(actVect_group)[3]
    else:
        if len(np.shape(actVect_group)) < 3:
            actVect_group=np.reshape(actVect_group,(np.shape(actVect_group)[0],1,np.shape(actVect_group)[1]))
        nTasks=np.shape(actVect_group)[1]
        nSubjs=np.shape(actVect_group)[2]
    
    nNodes=np.shape(actVect_group)[0]
    
    if len(np.shape(fcMat_group)) > 3:
        #If a separate FC state for each task
        if separate_activations_bytarget:
            actPredVector_bytask_bysubj=[[actflowcalc(actVect_group[:,:,taskNum,subjNum],fcMat_group[:,:,taskNum,subjNum], separate_activations_bytarget=True) for taskNum in range(nTasks)] for subjNum in range(nSubjs)]
        else:
            actPredVector_bytask_bysubj=[[actflowcalc(actVect_group[:,taskNum,subjNum],fcMat_group[:,:,taskNum,subjNum]) for taskNum in range(nTasks)] for subjNum in range(nSubjs)]
    else:
        if separate_activations_bytarget:
            actPredVector_bytask_bysubj=[[actflowcalc(actVect_group[:,:,taskNum,subjNum],fcMat_group[:,:,subjNum], separate_activations_bytarget=True) for taskNum in range(nTasks)] for subjNum in range(nSubjs)]
        else:
            actPredVector_bytask_bysubj=[[actflowcalc(actVect_group[:,taskNum,subjNum],fcMat_group[:,:,subjNum]) for taskNum in range(nTasks)] for subjNum in range(nSubjs)]
    actPredVector_bytask_bysubj=np.transpose(actPredVector_bytask_bysubj)
    
    #Calculate actual activation values
    actVect_actual_group = np.zeros((nNodes,nTasks,nSubjs))
    if separate_activations_bytarget:
        for taskNum in range(nTasks):
            for subjNum in range(nSubjs):
                actVect_actual_group[:,taskNum,subjNum] = actVect_group[:,:,taskNum,subjNum].diagonal()
    else:
        actVect_actual_group = actVect_group

    #Test for accuracy of actflow prediction, separately for each subject ("compare-then-average")
    predAcc_bytask_bysubj=[[np.corrcoef(actVect_actual_group[:,taskNum,subjNum],actPredVector_bytask_bysubj[:,taskNum,subjNum])[0,1] for subjNum in range(nSubjs)] for taskNum in range(nTasks)]
    
    #Run t-tests to quantify cross-subject consistency, by task
    tval_ActflowPredAcc_bytask=np.zeros(nTasks)
    pval_ActflowPredAcc_bytask=np.zeros(nTasks)
    for taskNum in range(nTasks):
        [tval_ActflowPredAcc_bytask[taskNum],pval_ActflowPredAcc_bytask[taskNum]]=stats.ttest_1samp(np.arctanh(predAcc_bytask_bysubj[taskNum]),0.0)

    #Grand mean (across task) t-test
    [tval_ActflowPredAcc_taskmean,pval_ActflowPredAcc_taskmean]=stats.ttest_1samp(np.mean(np.arctanh(predAcc_bytask_bysubj),axis=0),0.0)

    #Test for accuracy of actflow prediction, averaging across subjects before comparing ("average-then-compare")
    predAcc_bytask_avgfirst=[np.corrcoef(np.mean(actVect_actual_group[:,taskNum,:],axis=1),np.mean(actPredVector_bytask_bysubj[:,taskNum,:],axis=1))[0,1] for taskNum in range(nTasks)]
    
    print("==Correlation between predicted and actual activation patterns (mean across all tasks & subjects):")
    print("--Average-then-compare (calculating prediction accuracies after cross-subject averaging):")
    print("r=" + str("%.2f" % np.tanh(np.mean(np.arctanh(predAcc_bytask_avgfirst)))))
    print("By task:")
    for taskNum in range(nTasks):
        print("Task " + str(taskNum+1) + ": r=" + str("%.2f" % predAcc_bytask_avgfirst[taskNum]))
    print("--Compare-then-average (calculating prediction accuracies before cross-subject averaging):")
    print("r=" + str("%.2f" % np.tanh(np.mean(np.mean(np.arctanh(predAcc_bytask_bysubj))))) + ", t-value vs. 0: " + str("%.2f" % tval_ActflowPredAcc_taskmean) + ", p-value vs. 0: " + str(pval_ActflowPredAcc_taskmean))
    print("By task:")
    for taskNum in range(nTasks):
        print("Task " + str(taskNum+1) + ": r=" + str("%.2f" % np.tanh(np.mean(np.arctanh(predAcc_bytask_bysubj[taskNum])))) + ", t-value vs. 0: " + str("%.2f" % tval_ActflowPredAcc_bytask[taskNum]) + ", p-value vs. 0: " + str(pval_ActflowPredAcc_bytask[taskNum]))
    
    output = {'actPredVector_bytask_bysubj':actPredVector_bytask_bysubj,
              'predAcc_bytask_bysubj':predAcc_bytask_bysubj,
              'tval_ActflowPredAcc_bytask':tval_ActflowPredAcc_bytask,
              'pval_ActflowPredAcc_bytask':pval_ActflowPredAcc_bytask,
              'tval_ActflowPredAcc_taskmean':tval_ActflowPredAcc_taskmean,
              'pval_ActflowPredAcc_taskmean':pval_ActflowPredAcc_taskmean,
              'predAcc_bytask_avgfirst':predAcc_bytask_avgfirst,
              'actVect_actual_group':actVect_actual_group}
    
    return output
