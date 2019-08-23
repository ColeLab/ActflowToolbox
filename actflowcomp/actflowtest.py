
import numpy as np
import scipy.stats
from .actflowcalc import *
from ..model_compare import *

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
    if separate_activations_bytarget:
        for taskNum in range(nTasks):
            for subjNum in range(nSubjs):
                actVect_actual_group[:,taskNum,subjNum] = actVect_group[:,:,taskNum,subjNum].diagonal()
    else:
        actVect_actual_group = actVect_group

    ## Compare predicted to actual activations
    if actVect_group_test is not None:
        model_compare_output = model_compare(target_actvect=actVect_group_test, model1_actvect=actPredVector_bytask_bysubj, model2_actvect=actVect_actual_group, full_report=True, print_report=print_by_condition, mean_absolute_error=mean_absolute_error)
    else:
        model_compare_output = model_compare(target_actvect=actVect_actual_group, model1_actvect=actPredVector_bytask_bysubj, model2_actvect=None, full_report=True, print_report=print_by_condition, mean_absolute_error=mean_absolute_error)


    #     if mean_absolute_error == True:
    #         ##Accuracy of prediction using mean absolute error, separately for each subject ("compare-then average")
    #         maeAcc_bytask_bysubj = [[np.nanmean(np.abs(np.subtract(actVect_actual_group[:,taskNum,subjNum],actPredVector_bytask_bysubj[:,taskNum,subjNum]))) for subjNum in range(nSubjs)] for taskNum in range(nTasks)]            
    #         #averaging across subjects before comparing ("average-then-compare")
    #         maeAcc_bytask_avgthencomp=[np.nanmean(np.abs(np.subtract(np.nanmean(actVect_actual_group[:,taskNum,:],axis=1),np.nanmean(actPredVector_bytask_bysubj[:,taskNum,:],axis=1)))) for taskNum in range(nTasks)]

    #         print(" ")
    #         print("==Parcel-wise (spatial) Mean Absolute Error (MAE) between predicted and actual activation patterns (calculated for each condition separateley):==")
    #         print("--Compare-then-average (calculating MAE accuracies before cross-subjects averaging:)")
    #         print("mae=" + str("%.2f" % np.nanmean(np.nanmean(maeAcc_bytask_bysubj))))
    #         if print_by_condition:
    #             print("By task condition:")
    #             for taskNum in range(nTasks):
    #                 print("Condition " + str(taskNum+1) + ": mae=" + str("%.2f" % np.nanmean(maeAcc_bytask_bysubj[taskNum])))

    #         print("--Average-then-compare (calculating MAE accuracies after cross-subject averaging):")
    #         print("mae=" + str("%.2f" % np.nanmean(maeAcc_bytask_avgthencomp)))
    #         if print_by_condition:
    #             print("By task condition:")
    #             for taskNum in range(nTasks):
    #                 print("Condition " + str(taskNum+1) + ": mae=" + str("%.2f" % maeAcc_bytask_avgthencomp[taskNum]))



    output = {'actPredVector_bytask_bysubj':actPredVector_bytask_bysubj,
              'predAcc_bytask_bysubj':model_compare_output['corr_conditionwise_compthenavg_bynode'],
              'predAcc_bytask_avgthencomp':model_compare_output['corr_conditionwise_avgthencomp_bynode'],
              'actVect_actual_group':actVect_actual_group,
              'predAcc_bynode_bysubj':model_compare_output['corr_nodewise_compthenavg_bycond'],
              'predAccRankCorr_bynode_bysubj':model_compare_output['rankcorr_conditionwise_compthenavg_bynode'],
              'predAcc_bynode_avgthencomp':model_compare_output['corr_conditionwise_avgthencomp_bynode'],
              'predAccRankCorr_bynode_avgthencomp':model_compare_output['rankcorr_conditionwise_avgthencomp_bynode'],
              'model_compare_output':model_compare_output
             }
    
    if mean_absolute_error:
        #output.update({'maeAcc_bytask_bysubj':maeAcc_bytask_bysubj,
        #              'maeAcc_bytask_avgthencomp':maeAcc_bytask_avgthencomp,
        output.update({'maeAcc_bynode_bysubj':model_compare_output['maeAcc_bynode_compthenavg'],
                      'maeAcc_bynode_avgthencomp':model_compare_output['maeAcc_bynode_avgthencomp']})
    
    return output
