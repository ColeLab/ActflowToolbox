#Function to run activity flow mapping with spatial correlation predicted-to-actual testing across multiple tasks and subjects, either with a single (e.g., rest) connectivity matrix or with a separate connectivity matrix for each task. Returns stats at the group level.

import numpy as np
from scipy import stats
from .actflowcalc import *

def actflowtest(actVect_group, fcMat_group):
    if len(np.shape(actVect_group)) < 3:
        #Add empty task dimension if only 1 task
        actVect_group=np.reshape(actVect_group,(np.shape(actVect_group)[0],1,np.shape(actVect_group)[1]))
    nTasks=np.shape(actVect_group)[1]
    nSubjs=np.shape(actVect_group)[2]
    if len(np.shape(fcMat_group)) > 3:
        #If a separate FC state for each task
        actPredVector_bytask_bysubj=[[actflowcalc(actVect_group[:,taskNum,subjNum],fcMat_group[:,:,taskNum,subjNum]) for taskNum in range(nTasks)] for subjNum in range(nSubjs)]
    else:
        actPredVector_bytask_bysubj=[[actflowcalc(actVect_group[:,taskNum,subjNum],fcMat_group[:,:,subjNum]) for taskNum in range(nTasks)] for subjNum in range(nSubjs)]
    actPredVector_bytask_bysubj=np.transpose(actPredVector_bytask_bysubj)

    #Test for accuracy of actflow prediction, separately for each subject ("compare-then-average")
    predAcc_bytask_bysubj=[[np.corrcoef(actVect_group[:,taskNum,subjNum],actPredVector_bytask_bysubj[:,taskNum,subjNum])[0,1] for subjNum in range(nSubjs)] for taskNum in range(nTasks)]
    
    #Run t-tests to quantify cross-subject consistency, by task
    tval_ActflowPredAcc_bytask=np.zeros(nTasks)
    pval_ActflowPredAcc_bytask=np.zeros(nTasks)
    for taskNum in range(nTasks):
        [tval_ActflowPredAcc_bytask[taskNum],pval_ActflowPredAcc_bytask[taskNum]]=stats.ttest_1samp(np.arctanh(predAcc_bytask_bysubj[taskNum]),0.0)

    #Grand mean (across task) t-test
    [tval_ActflowPredAcc_taskmean,pval_ActflowPredAcc_taskmean]=stats.ttest_1samp(np.mean(np.arctanh(predAcc_bytask_bysubj),axis=0),0.0)

    #Test for accuracy of actflow prediction, averaging across subjects before comparing ("average-then-compare")
    predAcc_bytask_avgfirst=[np.corrcoef(np.mean(actVect_group[:,taskNum,:],axis=1),np.mean(actPredVector_bytask_bysubj[:,taskNum,:],axis=1))[0,1] for taskNum in range(nTasks)]
    
    print("==Correlation between predicted and actual activation patterns (mean across all tasks & subjects):")
    print("--Average-then-compare (calculating prediction accuracies after cross-subject averaging):")
    print("r=" + str(np.tanh(np.mean(np.arctanh(predAcc_bytask_avgfirst)))))
    print("By task:")
    for taskNum in range(nTasks):
        print("Task " + str(taskNum+1) + ": r=" + str(predAcc_bytask_avgfirst[taskNum]))
    print("--Compare-then-average (calculating prediction accuracies before cross-subject averaging):")
    print("r=" + str(np.tanh(np.mean(np.mean(np.arctanh(predAcc_bytask_bysubj))))) + ", t-value vs. 0: " + str(tval_ActflowPredAcc_taskmean) + ", p-value vs. 0: " + str(pval_ActflowPredAcc_taskmean))
    print("By task:")
    for taskNum in range(nTasks):
        print("Task " + str(taskNum+1) + ": r=" + str(np.tanh(np.mean(np.arctanh(predAcc_bytask_bysubj[taskNum])))) + ", t-value vs. 0: " + str(tval_ActflowPredAcc_bytask[taskNum]) + ", p-value vs. 0: " + str(pval_ActflowPredAcc_bytask[taskNum]))
    
    output = {'actPredVector_bytask_bysubj':actPredVector_bytask_bysubj,
              'predAcc_bytask_bysubj':predAcc_bytask_bysubj,
              'tval_ActflowPredAcc_bytask':tval_ActflowPredAcc_bytask,
              'pval_ActflowPredAcc_bytask':pval_ActflowPredAcc_bytask,
              'tval_ActflowPredAcc_taskmean':tval_ActflowPredAcc_taskmean,
              'pval_ActflowPredAcc_taskmean':pval_ActflowPredAcc_taskmean,
              'predAcc_bytask_avgfirst':predAcc_bytask_avgfirst}
    
    return output
