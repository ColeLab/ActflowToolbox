
import numpy as np
import scipy.stats
from ..model_compare import *

def noiseceilingcalc(actvect_group_first, actvect_group_second, full_report=False, print_report=True, reliability_type='conditionwise_compthenavgthen', avgthencomp_fixedeffects=False):
    """
    Function to calculate the repeat reliability of the data in various ways. This is equivalent to calculating the "noise ceiling" for predictive models (such as encoding models like activity flow models), which identifies theoretical limits on the highest prediction accuracy (based on the assumption that the data predicting itself is the highest possible prediction accuracy).
    
    Note that incorporation of spontaneous activity to predict task-evoked activity might allow for predictions above the noise ceiling (since spontaneous activity is considered "noise" with the noise ceiling approach).
    
    INPUTS
    actvect_group_first: node x condition x subject matrix with activation values. This should be distinct data from actvect_group_second (ideally identical in all ways, except a repetition of data collection at a different time)
    
    actvect_group_second: node x condition x subject matrix with activation values. This should be distinct data from actvect_group_first (ideally identical in all ways, except a repetition of data collection at a different time)
    
    full_report: Calculate full report with all reliability types
    
    print_report: Print the reliability report to screen
    
    reliability_type: The kind of reliability to calculate (when full_report=False). Options are:
    
        conditionwise_compthenavgthen - Compare-then-average condition-wise correlation between repetitions. This is run separately for each node, computing the correlation between the activations across conditions (which characterizes each node's response profile). The comparisons are computed separately for each subject, then results are summarized via averaging.
        conditionwise_avgthencomp - Average-then-compare condition-wise correlation between repetitions. This is run separately for each node, computing the correlation between the cross-condition activation patterns (which characterizes each node's response profile). Activations are averaged across subjects prior to comparison (sometimes called a "fixed effects" analysis), boosting signal-to-noise ratio but likely reducing dimensionality (through inter-subject averaging) and reducing the ability to assess the consistency of the result across subjects relative to compare-then-average.
        nodewise_compthenavg - Compare-then-average cross-node correlation between repetitions (whole-brain activation patterns). This is run separately for each condition, computing the correlation between the cross-node activation patterns (which characterizes each condition's response profile). The comparisons are computed separately for each subject, then results are summarized via averaging (sometimes called a "random effects" analysis).
        nodewise_avgthencomp - Average-then-compare cross-node correlation between repetitions (whole-brain activation patterns). This is run separately for each condition, computing the correlation between the cross-node activation patterns (which characterizes each condition's response profile). The comparisons are computed after averaging across subjects (sometimes called a "fixed effects" analysis).
        [TODO: subjwise_compthenavg (each node & condition based on individual differences)]
    
    OUTPUT
    output: a dictionary containing different variables depending on user input for full_report & reliability_type.
        See documentation for model_compare function for details.
    
    USAGE EXAMPLES
    [TODO: Update usage examples]
    
    import noiseceilingcalc as nc
    output = nc.noiseceilingcalc(actvect_group,full_report=False,print_report=True,reliability_type='conditionwise_compthenavgthen')
    print('Noise ceiling variables available: ',list(output)) # will show all variables available from this usage of nc; in this case it will contain 2 variables corresponding to conditionwise_compthenavgthen (1 matrix of r values, 1 grand mean r value)
    noiseCeilVal = output['repeat_corr_conditionwise_compthenavg_bynode_meanR'] # access the variable containing the grand mean r and assign it to be the noise ceiling metric for this model
    
    import noiseceilingcalc as nc
    output = nc.noiseceilingcalc(actvect_group,full_report=True,print_report=True)
    print('Noise ceiling variables available: ',list(output)) # will show all variables available from this usage of nc; in this case it will contain all 7 results (because full_report=True)
    noiseCeilVal = output['repeat_corr_nodewise_avgthencomp'] # an example of accessing the r value associated with 'nodewise_avgthencomp'
    """
    
    model_compare_output = model_compare(target_actvect=actvect_group_second, model1_actvect=actvect_group_first, model2_actvect=None, full_report=full_report, print_report=print_report, avgthencomp_fixedeffects=avgthencomp_fixedeffects)
    
    return model_compare_output
