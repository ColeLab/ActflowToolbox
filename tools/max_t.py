# Code to perform permutation testing to control for family-wise error (FWE)
# Using max-T approach as described in Nichols & Holmes (2002)
# Nichols TE, Holmes AP. (2002). Nonparametric permutation tests for functional neuroimaging: A primer with Examples. Hum. Brain Mapp., 15: 1-25. doi:10.1002/hbm.1058

import numpy as np
import scipy.stats as stats
import multiprocessing as mp
from statsmodels.distributions.empirical_distribution import ECDF
from functools import partial


def max_t(input_arr, nullmean=0, alpha=.05, tail=2, permutations=1000, nproc=1, pvals=True, output_nulldist=False, nan_policy='propagate'):
    """
    Performs family-wise ersror correction using permutation testing (Nichols & Holmes 2002).
    This function runs a one-sample t-test vs. 0 or, equivalently, a paired t-test (if the user subtracts two conditions prior to input).
    Assumes a two-sided t-test (specify tail of test by tail parameter).

    Citation: 
        Nichols TE, Holmes AP. (2002). Nonparametric permutation tests for functional neuroimaging: A primer with Examples. Hum. Brain Mapp., 15: 1-25. doi:10.1002/hbm.1058
    Required Parameters:
        input_arr    =  MxN matrix of set of M independent observations across N subjects. M one-sample t-tests 
                        (condition 1 vs. nullmean) or M paired t-tests (condition 1 minus condition 2) will be conducted,
                        correcting for multiple comparisons via the maxT approach.
                        Note that the user must subtract the two conditions prior to using this function in the 
                        paired t-test case.
    Optional Parameters:
        nullmean    =   Expected value of the null hypothesis {default = 0, for a t-test against 0}
        alpha       =   Optional. alpha value to return the maxT threshold {default = .05}
        tail        =   Optional. [0, 1, or -1] 
                        If tail = 1, reject the null hypothesis if the statistic is greater than the null dist (upper tailed test).  
                        If tail = -1, reject the null hypothesis if the statistic is less than the null dist (lower tailed test). 
                        If tail = 2, reject the null hypothesis for a two-tailed test
                        {default : 2} 
        permutations =  Optional. Number of permutations to perform {default = 1000}
        nproc       =   Optional. number of processes to run in parallel {default = 1}. NOTE: Could crash your Python session if it's set to a number over 1; it appears there is a bug that needs to be fixed.
        pvals       =   Optional. if True, returns equivalent p-value distribution for all t-values {default = True}
        nan_policy  =   Optional. What to do with NaN values when being sent to the t-test function. See scipy.stats.ttest_1samp (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html) for details. The default is to pass NaN values into the t-test function rather than ignoring them. {default = 'propagate'}

    Returns:
        t: Array of T-values of correct contrast map (Mx1 vector, for M tests)
        maxTThreshold   : The t-value threshold corresponding to the corrected alpha value. If a two-tailed test is specified, the maxR is provided as an absolute value
        p (optional)    : Array of FWE-corrected p-values (Mx1 vector, for M tests);
        maxT_dist (optional): Array of maxT null distribution values

    """
    # Focus on difference matrix -- more computationally feasible (and less data to feed into parallel processing)

    # Prepare inputs for multiprocessing
    seeds = np.zeros(permutations)
    for i in np.arange(permutations):
       seeds[i] = np.random.randint(0,100000,1)[0]

    pool = mp.Pool(processes=nproc)
    _maxTpermutation_partial=partial(_maxTpermutation, input_arr=input_arr, nullmean=nullmean, tail=tail, nan_policy=nan_policy)
    result = pool.map_async(_maxTpermutation_partial,seeds).get()
    pool.close()
    pool.join()

    # Returns an array of T-values distributions (our null distribution of "max-T" values)
    maxT_dist = np.asarray(result)

    #Find threshold for alpha
    maxT_dist_sorted = np.sort(maxT_dist)
    # Specify which tail we want
    if tail == 1:
        topPercVal_maxT_inx = int(len(maxT_dist_sorted)*(1-alpha))
        maxT_thresh = maxT_dist_sorted[topPercVal_maxT_inx]
    elif tail == -1:
        topPercVal_maxT_inx = int(len(maxT_dist_sorted)*(alpha))
        maxT_thresh = maxT_dist_sorted[topPercVal_maxT_inx]
    elif tail == 2:
        topPercVal_maxT_inx = int(len(maxT_dist_sorted)*(1-alpha))
        maxT_thresh = maxT_dist_sorted[topPercVal_maxT_inx]

    # Obtain real t-values 
    t = stats.ttest_1samp(input_arr, nullmean, axis=1, nan_policy=nan_policy)[0]

    if pvals:
        #         # Construct ECDF from maxT_dist
        #         ecdf = ECDF(maxT_dist)
        #         # Return p-values from maxT_dist using our empirical CDF (FWE-corrected p-values)
        #         p_fwe = ecdf(t)
        #         if tail == 1 or tail == 2:
        #             p_fwe = 1.0 - p_fwes
         
        if tail==1:
            #Percent of null t-values greater than observed t-value
            p_fwe = np.array([np.mean(maxT_dist>=tval) for tval in t])
        elif tail==-1:
            #Percent of null t-values less than observed t-value
            p_fwe = np.array([np.mean(maxT_dist<=tval) for tval in t])
        elif tail==2:
            #Percent of null t-values greater or less than observed t-value (the abs value in null distribution accounts for 2 tails)
            p_fwe = np.array([np.mean(maxT_dist>=np.abs(tval)) for tval in t])
        
        if output_nulldist:
            return t, maxT_thresh, p_fwe, maxT_dist
        else:
            return t, maxT_thresh, p_fwe

    else:
        if output_nulldist:
            return t, maxT_thresh, maxT_dist
        else:
            return t, maxT_thresh


def _maxTpermutation(seed,input_arr,nullmean,tail,nan_policy='propagate'):
    """
    Helper function to perform a single permutation
    """

    np.random.seed(int(seed))

    # Create a random matrix to shuffle conditions (randomly multiply contrasts by 1 or -1)
    shufflemat = np.random.normal(0,1,input_arr.shape)
    pos = shufflemat > 0
    neg = shufflemat < 0
    # matrix of 1 and -1
    shufflemat = pos + neg*(-1)

    # Shuffle raw values
    input_arr = np.multiply(input_arr, shufflemat)

    # Take t-test against 0 for each independent test 
    t_matrix = stats.ttest_1samp(input_arr,nullmean,axis=1, nan_policy=nan_policy)[0] 

    if tail==1:
        maxT = np.max(t_matrix)
    elif tail==-1:
        maxT = np.min(t_matrix)
    elif tail==2:
        maxT = np.max(np.abs(t_matrix))
    
    return maxT





