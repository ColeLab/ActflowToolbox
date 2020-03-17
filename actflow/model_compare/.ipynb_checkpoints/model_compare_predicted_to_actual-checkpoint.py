import numpy as np
import scipy.stats
import sklearn as sklearn

def model_compare_predicted_to_actual(target_actvect, pred_actvect, comparison_type='conditionwise_compthenavg'):
    
    nNodes=np.shape(target_actvect)[0]
    nConds=np.shape(target_actvect)[1]
    nSubjs=np.shape(target_actvect)[2]
    
    
    ## fullcompare_compthenavg - Compare-then-average across all conditions and all nodes between predicted and actual activations
    if comparison_type=='fullcompare_compthenavg':
        
        #Test for accuracy of actflow prediction, separately for each subject ("compare-then-average")
        corr_fullcomp_compthenavg = [np.corrcoef(target_actvect[:,:,subjNum].flatten(),pred_actvect[:,:,subjNum].flatten())[0,1] for subjNum in range(nSubjs)]
        #R2 coefficient of determination, https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
        R2_fullcomp_compthenavg = [sklearn.metrics.r2_score(target_actvect[:,:,subjNum].flatten(),pred_actvect[:,:,subjNum].flatten()) for subjNum in range(nSubjs)]
        #mean_absolute_error: compute the absolute mean error: mean(abs(a-p)), where a are the actual activations and p the predicted activations across all the nodes.
        maeAcc_fullcomp_compthenavg = [np.nanmean(np.abs(np.subtract(target_actvect[:,:,subjNum].flatten(),pred_actvect[:,:,subjNum].flatten()))) for subjNum in range(nSubjs)]

        output = {'corr_vals':corr_fullcomp_compthenavg,'R2_vals':R2_fullcomp_compthenavg,'mae_vals':maeAcc_fullcomp_compthenavg}
    
    ## conditionwise_compthenavg - Compare-then-average condition-wise correlation between predicted and actual activations
    if comparison_type=='conditionwise_compthenavg':

        #Test for accuracy of actflow prediction, separately for each subject ("compare-then-average")
        corr_conditionwise_compthenavg_bynode = [[np.corrcoef(target_actvect[nodeNum,:,subjNum],pred_actvect[nodeNum,:,subjNum])[0,1] for subjNum in range(nSubjs)] for nodeNum in range(nNodes)]
        #R2 coefficient of determination, https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
        R2_conditionwise_compthenavg_bynode = [[sklearn.metrics.r2_score(target_actvect[nodeNum,:,subjNum],pred_actvect[nodeNum,:,subjNum]) for subjNum in range(nSubjs)] for nodeNum in range(nNodes)]
        
        ## mean_absolute_error: compute the absolute mean error: mean(abs(a-p)), where a are the actual activations and p the predicted activations across all the nodes.
        maeAcc_bynode_compthenavg = [[np.nanmean(np.abs(np.subtract(target_actvect[nodeNum,:,subjNum],pred_actvect[nodeNum,:,subjNum]))) for subjNum in range(nSubjs)] for nodeNum in range(nNodes)]

        output = {'corr_vals':corr_conditionwise_compthenavg_bynode,'R2_vals':R2_conditionwise_compthenavg_bynode,'mae_vals':maeAcc_bynode_compthenavg}
        
        

    ## conditionwise_avgthencomp - Average-then-compare condition-wise correlation between predicted and actual activations
    if comparison_type=='conditionwise_avgthencomp':
        
        corr_conditionwise_avgthencomp_bynode=[np.corrcoef(np.nanmean(target_actvect[nodeNum,:,:],axis=1),np.nanmean(pred_actvect[nodeNum,:,:],axis=1))[0,1] for nodeNum in range(nNodes)]

        #R2 coefficient of determination, https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
        R2_conditionwise_avgthencomp_bynode = [sklearn.metrics.r2_score(np.nanmean(target_actvect[nodeNum,:,:],axis=1),np.nanmean(pred_actvect[nodeNum,:,:],axis=1)) for nodeNum in range(nNodes)]
        
        ## mean_absolute_error: compute the absolute mean error: mean(abs(a-p)), where a are the actual activations and p the predicted activations across all the nodes.
        maeAcc_bynode_avgthencomp =[np.nanmean(np.abs(np.subtract(np.nanmean(target_actvect[nodeNum,:,:],axis=1),np.nanmean(pred_actvect[nodeNum,:,:],axis=1)))) for nodeNum in range(nNodes)]

        output = {'corr_conditionwise_avgthencomp_bynode':corr_conditionwise_avgthencomp_bynode,'R2_conditionwise_avgthencomp_bynode':R2_conditionwise_avgthencomp_bynode,'maeAcc_bynode_avgthencomp':maeAcc_bynode_avgthencomp}
    
    
    
    ## nodewise_compthenavg - Compare-then-average cross-node correlation between predicted and actual activations (whole-brain activation patterns)
    if comparison_type=='nodewise_compthenavg':
    
        #Test for accuracy of actflow prediction, separately for each subject ("compare-then-average")
        corr_nodewise_compthenavg_bycond=[[np.corrcoef(target_actvect[:,taskNum,subjNum], pred_actvect[:,taskNum,subjNum])[0,1] for subjNum in range(nSubjs)] for taskNum in range(nConds)]
                                               
        #R2 coefficient of determination, https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
        R2_nodewise_compthenavg_bycond = [[sklearn.metrics.r2_score(target_actvect[:,taskNum,subjNum], pred_actvect[:,taskNum,subjNum]) for subjNum in range(nSubjs)] for taskNum in range(nConds)]
        
        ## mean_absolute_error: compute the absolute mean error: mean(abs(a-p)), where a are the actual activations and p the predicted activations across all the nodes.
        maeAcc_nodewise_compthenavg_bycond = [[np.nanmean(np.abs(np.subtract(target_actvect[:,taskNum,subjNum], pred_actvect[:,taskNum,subjNum]))) for subjNum in range(nSubjs)] for taskNum in range(nConds)]

        output = {'corr_vals':corr_nodewise_compthenavg_bycond,'R2_vals':R2_nodewise_compthenavg_bycond,'mae_vals':maeAcc_nodewise_compthenavg_bycond}
    
    
    ## nodewise_avgthencomp - Average-then-compare cross-node correlation between predicted and actual activations (whole-brain activation patterns)
    if comparison_type == 'nodewise_avgthencomp':
        
        #Test for accuracy of actflow prediction, averaging across subjects before comparing ("average-then-compare")
        corr_nodewise_avgthencomp_bycond=[np.corrcoef(np.nanmean(target_actvect[:,taskNum,:],axis=1),np.nanmean(pred_actvect[:,taskNum,:],axis=1))[0,1] for taskNum in range(nConds)]
        
        output = {'corr_nodewise_avgthencomp_bycond':corr_nodewise_avgthencomp_bycond}
                                               
        #R2 coefficient of determination, https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
        R2_nodewise_avgthencomp_bycond = [sklearn.metrics.r2_score(np.nanmean(target_actvect[:,taskNum,:],axis=1),np.nanmean(pred_actvect[:,taskNum,:],axis=1)) for taskNum in range(nConds)]
        
        ## mean_absolute_error: compute the absolute mean error: mean(abs(a-p)), where a are the actual activations and p the predicted activations across all the nodes.
        maeAcc_nodewise_avgthencomp_bycond = [np.nanmean(np.abs(np.subtract(np.nanmean(target_actvect[:,taskNum,:],axis=1),np.nanmean(pred_actvect[:,taskNum,:],axis=1)))) for taskNum in range(nConds)]

        output = {'corr_nodewise_avgthencomp_bycond':corr_nodewise_avgthencomp_bycond,'R2_nodewise_avgthencomp_bycond':R2_nodewise_avgthencomp_bycond,'maeAcc_nodewise_avgthencomp_bycond':maeAcc_nodewise_avgthencomp_bycond}
        
        
    return output