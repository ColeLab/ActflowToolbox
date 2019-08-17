
import numpy as np
import scipy.stats

def noiseceilingcalc(actvect_group, full_report=False, print_report=True, reliability_type='conditionwise_compthenavgthen'):
    """
    Function to calculate the repeat reliability of the data in various ways. This is equivalent to calculating the "noise ceiling" for predictive models (such as encoding models like activity flow models), which identifies theoretical limits on the highest prediction accuracy (based on the assumption that the data predicting itself is the highest possible prediction accuracy).
    
    Note that incorporation of spontaneous activity to predict task-evoked activity might allow for predictions above the noise ceiling (since spontaneous activity is considered "noise" with the noise ceiling approach).
    
    INPUTS
    actvect_group: node x condition x repetition x subject matrix with activation values. Note that only the first 2 values in the 'repetition' dimension are used.
    
    full_report: Calculate full report with all reliability types
    
    print_report: Print the reliability report to screen
    
    reliability_type: The kind of reliability to calculate (when full_report=False). Options are:
    
        conditionwise_compthenavgthen - Compare-then-average condition-wise correlation between repetitions. This is run separately for each node, computing the correlation between the activations across conditions (which characterizes each node's response profile). The comparisons are computed separately for each subject, then results are summarized via averaging.
        conditionwise_avgthencomp - Average-then-compare condition-wise correlation between repetitions. This is run separately for each node, computing the correlation between the cross-condition activation patterns (which characterizes each node's response profile). Activations are averaged across subjects prior to comparison (sometimes called a "fixed effects" analysis), boosting signal-to-noise ratio but likely reducing dimensionality (through inter-subject averaging) and reducing the ability to assess the consistency of the result across subjects relative to compare-then-average.
        nodewise_compthenavg - Compare-then-average cross-node correlation between repetitions (whole-brain activation patterns). This is run separately for each condition, computing the correlation between the cross-node activation patterns (which characterizes each condition's response profile). The comparisons are computed separately for each subject, then results are summarized via averaging (sometimes called a "random effects" analysis).
        nodewise_avgthencomp - Average-then-compare cross-node correlation between repetitions (whole-brain activation patterns). This is run separately for each condition, computing the correlation between the cross-node activation patterns (which characterizes each condition's response profile). The comparisons are computed after averaging across subjects (sometimes called a "fixed effects" analysis).
        [TODO: subjwise_compthenavg (each node & condition based on individual differences)]
    
    OUTPUT
    output: a dictionary containing 1-7 variables, depending on user input for full_report & reliability_type
        when full_report=True: output contains variables for all reliability_type runs (7 total). It is recommended to set full_report=False if setting a specific reliability_type. (see examples below)
        when full_report=False,reliability_type='conditionwise_compthenavgthen': output contains 2 variables, 1 node x subject matrix of r values and 1 grand mean r
        when full_report=False,reliability_type='conditionwise_avgthencomp': output contains 2 variables, 1 node x 1 vector of r values and 1 mean r
        when full_report=False,reliability_type='conditionwise_compthenavgthen': output contains 2 variables, 1 subject x 1 vector of r values and 1 mean r
        when full_report=False,reliability_type='conditionwise_compthenavgthen': output contains 1 variable, 1 r value
    
    USAGE EXAMPLES
    import noiseceilingcalc as nc
    output = nc.noiseceilingcalc(actvect_group,full_report=False,print_report=True,reliability_type='conditionwise_compthenavgthen')
    print('Noise ceiling variables available: ',list(output)) # will show all variables available from this usage of nc; in this case it will contain 2 variables corresponding to conditionwise_compthenavgthen (1 matrix of r values, 1 grand mean r value)
    noiseCeilVal = output['repeat_corr_conditionwise_compthenavg_bynode_meanR'] # access the variable containing the grand mean r and assign it to be the noise ceiling metric for this model
    
    import noiseceilingcalc as nc
    output = nc.noiseceilingcalc(actvect_group,full_report=True,print_report=True)
    print('Noise ceiling variables available: ',list(output)) # will show all variables available from this usage of nc; in this case it will contain all 7 results (because full_report=True)
    noiseCeilVal = output['repeat_corr_nodewise_avgthencomp'] # an example of accessing the r value associated with 'nodewise_avgthencomp'
    """
    
    nNodes=np.shape(actvect_group)[0]
    nConds=np.shape(actvect_group)[1]
    nSubjs=np.shape(actvect_group)[3]
    
    #conditionwise_compthenavgthen - Compare-then-average, condition-wise repeat reliability
    if full_report or reliability_type=='conditionwise_compthenavgthen':

        repeat_corr_conditionwise_compthenavg_bynode=np.zeros((nNodes,nSubjs))
        for scount in list(range(nSubjs)):
            for node_num in list(range(nNodes)):
                run1_mean_activations=actvect_group[node_num,:,0,scount]
                run2_mean_activations=actvect_group[node_num,:,1,scount]
                repeat_corr_conditionwise_compthenavg_bynode[node_num,scount] = np.corrcoef(run1_mean_activations,run2_mean_activations)[0,1]
    
        repeat_corr_conditionwise_compthenavg_bynode_meanR = np.mean(np.mean(repeat_corr_conditionwise_compthenavg_bynode))

        if print_report:
            print('Compare-then-average condition-wise correlation between repetitions (cross-node & cross-subj mean):')
            print('r = ', str("%.2f" % np.mean(np.mean(repeat_corr_conditionwise_compthenavg_bynode))))

        if reliability_type=='conditionwise_compthenavgthen':
            output = {'repeat_corr_conditionwise_compthenavg_bynode':repeat_corr_conditionwise_compthenavg_bynode,'repeat_corr_conditionwise_compthenavg_bynode_meanR':repeat_corr_conditionwise_compthenavg_bynode_meanR}
        
        
    #conditionwise_avgthencomp - Average-then-compare condition-wise correlation between repetitions
    if full_report or reliability_type=='conditionwise_avgthencomp':
        
        repeat_corr_conditionwise_avgthencomp_bynode=np.zeros((nNodes))
        for node_num in list(range(nNodes)):
            run1_mean_activations=np.mean(actvect_group[node_num,:,0,:],axis=1)
            run2_mean_activations=np.mean(actvect_group[node_num,:,1,:],axis=1)
            repeat_corr_conditionwise_avgthencomp_bynode[node_num] = np.corrcoef(run1_mean_activations,run2_mean_activations)[0,1]
        
        repeat_corr_conditionwise_avgthencomp_bynode_meanR = np.mean(repeat_corr_conditionwise_avgthencomp_bynode)

        if print_report:
            print('Average-then-compare condition-wise correlation between repetitions (cross-node & cross-subj mean):')
            print('r = ', str("%.2f" % np.mean(repeat_corr_conditionwise_avgthencomp_bynode)))

        if reliability_type=='conditionwise_avgthencomp':
                output = {'repeat_corr_conditionwise_avgthencomp_bynode':repeat_corr_conditionwise_avgthencomp_bynode,'repeat_corr_conditionwise_avgthencomp_bynode_meanR':repeat_corr_conditionwise_avgthencomp_bynode_meanR}

    #nodewise_compthenavg - Compare-then-average cross-node correlation between repetitions (whole-brain activation patterns)
    if full_report or reliability_type=='nodewise_compthenavg':
    
        repeat_corr_nodewise_compthenavg = np.zeros((nConds,nSubjs))
        for scount in list(range(nSubjs)):
            for cond_num in list(range(nConds)):
                run1_mean_activations=actvect_group[:,cond_num,0,scount].flatten()
                run2_mean_activations=actvect_group[:,cond_num,1,scount].flatten()
                repeat_corr_nodewise_compthenavg[cond_num,scount] = np.corrcoef(run1_mean_activations,run2_mean_activations)[0,1]
    
        repeat_corr_nodewise_compthenavg_meanR = np.nanmean(np.nanmean(repeat_corr_nodewise_compthenavg))

        if print_report:
            print('Compare-then-average subject-wise, cross-node correlations between repititions (whole brain activation patterns, cross-subject):')
            print('r = ', str("%.2f" % np.nanmean(repeat_corr_nodewise_compthenavg)))

        if reliability_type=='nodewise_compthenavg':
            output = {'repeat_corr_nodewise_compthenavg':repeat_corr_nodewise_compthenavg,'repeat_corr_nodewise_compthenavg_meanR':repeat_corr_nodewise_compthenavg_meanR}


    #nodewise_avgthencomp - Average-then-compare cross-node repeat reliability (whole-brain activation patterns)
    if full_report or reliability_type=='nodewise_avgthencomp':
        
        repeat_corr_nodewise_avgthencomp_bycond=np.zeros((nConds))
        for cond_num in list(range(nConds)):
            run1_mean_activations=np.mean(actvect_group[:,cond_num,0,:],axis=1).flatten()
            run2_mean_activations=np.mean(actvect_group[:,cond_num,1,:],axis=1).flatten()
            repeat_corr_nodewise_avgthencomp_bycond[cond_num]=np.corrcoef(run1_mean_activations,run2_mean_activations)[0,1]
            
        repeat_corr_nodewise_avgthencomp=np.nanmean(repeat_corr_nodewise_avgthencomp_bycond)
        
        if print_report:
            print("Average-then-compare cross-node repeat reliability (whole-brain activation patterns):")
            print('r = ', str("%.2f" % repeat_corr_nodewise_avgthencomp))

        if reliability_type=='nodewise_avgthencomp':
            output = {'repeat_corr_nodewise_avgthencomp':repeat_corr_nodewise_avgthencomp}

    if full_report:
        output = {'repeat_corr_conditionwise_compthenavg_bynode':repeat_corr_conditionwise_compthenavg_bynode,'repeat_corr_conditionwise_compthenavg_bynode_meanR':repeat_corr_conditionwise_compthenavg_bynode_meanR,'repeat_corr_conditionwise_avgthencomp_bynode':repeat_corr_conditionwise_avgthencomp_bynode,'repeat_corr_conditionwise_avgthencomp_bynode_meanR':repeat_corr_conditionwise_avgthencomp_bynode_meanR,'repeat_corr_nodewise_compthenavg':repeat_corr_nodewise_compthenavg,'repeat_corr_nodewise_compthenavg_meanR':repeat_corr_nodewise_compthenavg_meanR,'repeat_corr_nodewise_avgthencomp_bycond':repeat_corr_nodewise_avgthencomp_bycond,'repeat_corr_nodewise_avgthencomp':repeat_corr_nodewise_avgthencomp}
    
    return output
