
import numpy as np
import scipy.stats
from .actflowcalc import *

def actflowtest(actVect_group, fcMat_group, actVect_group_test=None, print_by_condition=True, separate_activations_bytarget=False, mean_absolute_error=False):
    """
    Function to run activity flow mapping with spatial correlation predicted-to-actual testing across multiple tasks and subjects, either with a single (e.g., rest) connectivity matrix or with a separate connectivity matrix for each task. Returns statistics at the group level.
    
    actVect_group: node x condition x subject matrix with activation values
    fcMat_group: node x node x condition x subject matrix (or node x node x subject matrix) with connectiivty values
    actVect_group_test: independent data (e.g., a separate run) than actVect_group, used as separate "test" data for testing prediction accuracies. Node x condition x subject matrix with activation values. Note: In separate_activations_bytarget=True case, actVect_group_test should not have separate activations by target (just use original node x condition x subject version of data).
    separate_activations_bytarget: indicates if the input actVect_group matrix has a separate activation vector for each target (to-be-predicted) node (e.g., for the locally-non-circular approach)
    mean_absolute_error: if True, compute the absolute mean error: mean(abs(a-p)), where a are the actual activations
    and p the predicted activations across all the nodes.
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
    
    #Calculate activity flow predictions
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
    
    #Calculate actual activation values (e.g., for when separate_activations_bytarget is True)
    actVect_actual_group = np.zeros((nNodes,nTasks,nSubjs))
    if actVect_group_test is not None:
        #Setting target activations to be from the separate test data
        #Note: In separate_activations_bytarget=True case, actVect_group_test should not have separate activations by target (just use original version of data)
        actVect_actual_group = actVect_group_test
    elif separate_activations_bytarget:
        for taskNum in range(nTasks):
            for subjNum in range(nSubjs):
                actVect_actual_group[:,taskNum,subjNum] = actVect_group[:,:,taskNum,subjNum].diagonal()
    else:
        actVect_actual_group = actVect_group

        
    print("==Similarity between predicted and actual brain activation patterns==")
    
    
    ## Condition-wise (rather than node-wise) tests for accuracy of actflow predictions (only if more than one task condition)
    if nTasks > 1:
    
        #Test for accuracy of actflow prediction, separately for each subject ("compare-then-average")
        predAcc_bynode_bysubj=[[np.corrcoef(actVect_actual_group[nodeNum,:,subjNum],actPredVector_bytask_bysubj[nodeNum,:,subjNum])[0,1] for subjNum in range(nSubjs)] for nodeNum in range(nNodes)]
        
        predAccRankCorr_bynode_bysubj=[[scipy.stats.spearmanr(actVect_actual_group[nodeNum,:,subjNum],actPredVector_bytask_bysubj[nodeNum,:,subjNum])[0] for subjNum in range(nSubjs)] for nodeNum in range(nNodes)]

        #Run t-tests to quantify cross-subject consistency, by node
        #tval_ActflowPredAcc_bynode=np.zeros(nNodes)
        #pval_ActflowPredAcc_bynode=np.zeros(nNodes)
        #tval_ActflowPredAccRankCorr_bynode=np.zeros(nNodes)
        #pval_ActflowPredAccRankCorr_bynode=np.zeros(nNodes)
        #for nodeNum in range(nNodes):
        #    [tval_ActflowPredAcc_bynode[nodeNum],pval_ActflowPredAcc_bynode[nodeNum]] = scipy.stats.ttest_1samp(np.arctanh(predAcc_bynode_bysubj[nodeNum]),0.0)
        #    [tval_ActflowPredAccRankCorr_bynode[nodeNum],pval_ActflowPredAccRankCorr_bynode[nodeNum]] = scipy.stats.ttest_1samp(np.nan_to_num(np.arctanh(predAccRankCorr_bynode_bysubj[nodeNum])),0.0)

        #Grand mean (across task) t-test
        [tval_ActflowPredAcc_nodemean,pval_ActflowPredAcc_nodemean] = scipy.stats.ttest_1samp(np.nanmean(np.ma.arctanh(predAcc_bynode_bysubj),axis=0),0.0)
        #Set perfect rank correlations (rho=1) to be slightly lower (to avoid warnings) (rho=0.9999)
        predAccRankCorr_bynode_bysubj_mod=predAccRankCorr_bynode_bysubj.copy()
        predAccRankCorr_bynode_bysubj_mod=np.asarray(predAccRankCorr_bynode_bysubj_mod)
        predAccRankCorr_bynode_bysubj_mod[np.abs(predAccRankCorr_bynode_bysubj_mod)==1] = 0.9999
        [tval_ActflowPredAccRankCorr_nodemean,pval_ActflowPredAccRankCorr_nodemean] = scipy.stats.ttest_1samp(np.nanmean(np.ma.arctanh(predAccRankCorr_bynode_bysubj_mod),axis=0),0.0)
        #import pdb; pdb.set_trace()
        
        #Test for accuracy of actflow prediction, separately for each subject ("average-then-compare")
        predAcc_bynode_avgthencomp=[np.corrcoef(np.nanmean(actVect_actual_group[nodeNum,:,:],axis=1),np.nanmean(actPredVector_bytask_bysubj[nodeNum,:,:],axis=1))[0,1] for nodeNum in range(nNodes)]
        
        predAccRankCorr_bynode_avgthencomp=[scipy.stats.spearmanr(np.nanmean(actVect_actual_group[nodeNum,:,:],axis=1),np.nanmean(actPredVector_bytask_bysubj[nodeNum,:,:],axis=1))[0] for nodeNum in range(nNodes)]
    
        print(" ")
        print("==Condition-wise correlations between predicted and actual activation patterns (calculated for each node separetely):==")
        
        print("--Compare-then-average (calculating prediction accuracies before cross-subject averaging):")
        print("Each correlation based on N conditions: " + str(nTasks) + ", p-values based on N subjects (cross-subject variance in correlations): " + str(nSubjs))
        print("Mean Pearson r=" + str("%.2f" % np.tanh(np.nanmean(np.nanmean(np.ma.arctanh(predAcc_bynode_bysubj))))) + ", t-value vs. 0: " + str("%.2f" % tval_ActflowPredAcc_nodemean) + ", p-value vs. 0: " + str(pval_ActflowPredAcc_nodemean))
        print("Mean rank-correlation rho=" + str("%.2f" % np.nanmean(np.nanmean(predAccRankCorr_bynode_bysubj))) + ", t-value vs. 0: " + str("%.2f" % tval_ActflowPredAccRankCorr_nodemean) + ", p-value vs. 0: " + str(pval_ActflowPredAccRankCorr_nodemean))
        
        print("--Average-then-compare (calculating prediction accuracies after cross-subject averaging):")
        print("Each correlation based on N conditions: " + str(nTasks))
        print("Mean Pearson r=" + str("%.2f" % np.tanh(np.nanmean(np.ma.arctanh(predAcc_bynode_avgthencomp)))))
        print("Mean rank-correlation rho=" + str("%.2f" % np.nanmean(predAccRankCorr_bynode_avgthencomp)))
        
        if mean_absolute_error == True:
            print(" ")
            print("==Condition-wise Mean Absolute Error (MAE) between predicted and actual activation patterns (calculated for each node separateley):==")
            maeAcc_bynode_bysubj=[[np.nanmean(np.abs(np.subtract(actVect_actual_group[nodeNum,:,subjNum],actPredVector_bytask_bysubj[nodeNum,:,subjNum]))) for subjNum in range(nSubjs)] for nodeNum in range(nNodes)]
            maeAcc_bynode_avgthencomp=[np.nanmean(np.abs(np.subtract(np.nanmean(actVect_actual_group[nodeNum,:,:],axis=1),np.nanmean(actPredVector_bytask_bysubj[nodeNum,:,:],axis=1)))) for nodeNum in range(nNodes)]
            print("--Compare-then-average (calculating MAE accuracies before cross-subject averaging):")
            print("Each MAE based on N conditions: " + str(nTasks))
            print("Mean MAE r=" + str("%.2f" % np.nanmean(np.nanmean(maeAcc_bynode_bysubj))))
            
            print("--Average-then-compare (calculating MAE accuracies after cross-subject averaging):")
            print("Each MAE based on N conditions: " + str(nTasks))
            print("Mean MAE=" + str("%.2f" % np.nanmean(maeAcc_bynode_avgthencomp)))

        
        
    ## Parcel-wise analyses (as opposed to condition-wise)
    
    print(" ")
    print("==Parcel-wise (spatial) correlations between predicted and actual activation patterns (calculated for each condition separetely):==")
    
    #Test for accuracy of actflow prediction, separately for each subject ("compare-then-average")
    predAcc_bytask_bysubj=[[np.corrcoef(actVect_actual_group[:,taskNum,subjNum],actPredVector_bytask_bysubj[:,taskNum,subjNum])[0,1] for subjNum in range(nSubjs)] for taskNum in range(nTasks)]
    
    #Run t-tests to quantify cross-subject consistency, by task
    tval_ActflowPredAcc_bytask=np.zeros(nTasks)
    pval_ActflowPredAcc_bytask=np.zeros(nTasks)
    for taskNum in range(nTasks):
        [tval_ActflowPredAcc_bytask[taskNum],pval_ActflowPredAcc_bytask[taskNum]]=scipy.stats.ttest_1samp(np.ma.arctanh(predAcc_bytask_bysubj[taskNum]),0.0)

    #Grand mean (across task) t-test
    [tval_ActflowPredAcc_taskmean,pval_ActflowPredAcc_taskmean]=scipy.stats.ttest_1samp(np.nanmean(np.ma.arctanh(predAcc_bytask_bysubj),axis=0),0.0)

    #Test for accuracy of actflow prediction, averaging across subjects before comparing ("average-then-compare")
    predAcc_bytask_avgthencomp=[np.corrcoef(np.nanmean(actVect_actual_group[:,taskNum,:],axis=1),np.nanmean(actPredVector_bytask_bysubj[:,taskNum,:],axis=1))[0,1] for taskNum in range(nTasks)]

    print("--Compare-then-average (calculating prediction accuracies before cross-subject averaging):")
    print("r=" + str("%.2f" % np.tanh(np.nanmean(np.nanmean(np.ma.arctanh(predAcc_bytask_bysubj))))) + ", t-value vs. 0: " + str("%.2f" % tval_ActflowPredAcc_taskmean) + ", p-value vs. 0: " + str(pval_ActflowPredAcc_taskmean))
    if print_by_condition:
        print("By task condition:")
        for taskNum in range(nTasks):
            print("Condition " + str(taskNum+1) + ": r=" + str("%.2f" % np.tanh(np.nanmean(np.ma.arctanh(predAcc_bytask_bysubj[taskNum])))) + ", t-value vs. 0: " + str("%.2f" % tval_ActflowPredAcc_bytask[taskNum]) + ", p-value vs. 0: " + str(pval_ActflowPredAcc_bytask[taskNum]))
            
    print("--Average-then-compare (calculating prediction accuracies after cross-subject averaging):")
    print("r=" + str("%.2f" % np.tanh(np.nanmean(np.ma.arctanh(predAcc_bytask_avgthencomp)))))
    if print_by_condition:
        print("By task condition:")
        for taskNum in range(nTasks):
            print("Condition " + str(taskNum+1) + ": r=" + str("%.2f" % predAcc_bytask_avgthencomp[taskNum]))
            
  
    if mean_absolute_error == True:
        ##Accuracy of prediction using mean absolute error, separately for each subject ("compare-then average")
        maeAcc_bytask_bysubj = [[np.nanmean(np.abs(np.subtract(actVect_actual_group[:,taskNum,subjNum],actPredVector_bytask_bysubj[:,taskNum,subjNum]))) for subjNum in range(nSubjs)] for taskNum in range(nTasks)]            
        #averaging across subjects before comparing ("average-then-compare")
        maeAcc_bytask_avgthencomp=[np.nanmean(np.abs(np.subtract(np.nanmean(actVect_actual_group[:,taskNum,:],axis=1),np.nanmean(actPredVector_bytask_bysubj[:,taskNum,:],axis=1)))) for taskNum in range(nTasks)]
        
        print(" ")
        print("==Parcel-wise (spatial) Mean Absolute Error (MAE) between predicted and actual activation patterns (calculated for each condition separateley):==")
        print("--Compare-then-average (calculating MAE accuracies before cross-subjects averaging:)")
        print("mae=" + str("%.2f" % np.nanmean(np.nanmean(maeAcc_bytask_bysubj))))
        if print_by_condition:
            print("By task condition:")
            for taskNum in range(nTasks):
                print("Condition " + str(taskNum+1) + ": mae=" + str("%.2f" % np.nanmean(maeAcc_bytask_bysubj[taskNum])))
        
        print("--Average-then-compare (calculating MAE accuracies after cross-subject averaging):")
        print("mae=" + str("%.2f" % np.nanmean(maeAcc_bytask_avgthencomp)))
        if print_by_condition:
            print("By task condition:")
            for taskNum in range(nTasks):
                print("Condition " + str(taskNum+1) + ": mae=" + str("%.2f" % maeAcc_bytask_avgthencomp[taskNum]))



    output = {'actPredVector_bytask_bysubj':actPredVector_bytask_bysubj,
              'predAcc_bytask_bysubj':predAcc_bytask_bysubj,
              'tval_ActflowPredAcc_bytask':tval_ActflowPredAcc_bytask,
              'pval_ActflowPredAcc_bytask':pval_ActflowPredAcc_bytask,
              'tval_ActflowPredAcc_taskmean':tval_ActflowPredAcc_taskmean,
              'pval_ActflowPredAcc_taskmean':pval_ActflowPredAcc_taskmean,
              'predAcc_bytask_avgthencomp':predAcc_bytask_avgthencomp,
              'actVect_actual_group':actVect_actual_group,
              'predAcc_bynode_bysubj':predAcc_bynode_bysubj,
              'predAccRankCorr_bynode_bysubj':predAccRankCorr_bynode_bysubj,
              'tval_ActflowPredAcc_nodemean':tval_ActflowPredAcc_nodemean,
              'pval_ActflowPredAcc_nodemean':pval_ActflowPredAcc_nodemean,
              'tval_ActflowPredAccRankCorr_nodemean':tval_ActflowPredAccRankCorr_nodemean,
              'pval_ActflowPredAccRankCorr_nodemean':pval_ActflowPredAccRankCorr_nodemean,
              'predAcc_bynode_avgthencomp':predAcc_bynode_avgthencomp,
              'predAccRankCorr_bynode_avgthencomp':predAccRankCorr_bynode_avgthencomp
             }
    
    if mean_absolute_error == True:
        output.update({'maeAcc_bytask_bysubj':maeAcc_bytask_bysubj,
                      'maeAcc_bytask_avgthencomp':maeAcc_bytask_avgthencomp,
                      'maeAcc_bynode_bysubj':maeAcc_bynode_bysubj,
                      'maeAcc_bynode_avgthencomp':maeAcc_bynode_avgthencomp})
    
    return output
