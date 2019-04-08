# Main code for 'information transfer mapping procedures',  Ito et al., 2017
# Consists of two functions:
# 1. x to y activity flow mapping
# 2. predicted-to-actual similarity analysis

# Takuya Ito (takuya.ito@rutgers.edu)
# Citation: Ito T, Kulkarni KR, Schultz DH, Mill RD, Chen RH, Solomyak LI, Cole MW (2017). Cognitive task information is transferred between brain regions via resting-state netw    ork topology. bioRxiv. https://doi.org/10.1101/101782

## Import modules
import numpy as np
import scipy.stats as stats

def activityFlowXtoY(activity,connectivity):
    """
    Activity flow mapping between component (e.g., region or network) X to component Y
    
    Parameters:
        activity - a sample (e.g., miniblocks) X feature (e.g., regions/vertices) matrix
        connectivity - a functional connectivity matrix from X to Y; rows must have same number of columns as in activity

    Returns: 
        Predictions for component Y in terms of samples X features 

    """
    predictions = np.dot(activity, connectivity)

    return predictions

def predictedToActualSimilarity(predictions,actual,labels):
    """
    A cross-validated predicted-to-actual similarity analysis; serves to quantify how much information has transferred
    Similar to a cross-validated RSA

    Parameters:
        predictions - predicted activation patterns. A sample X features matrix.
        actual - actual activation patterns to compare predictions against. A sample X features matrix.
        labels - a label matrix organized as a 32 (samples) x 4 (conditions) matrix; columns indicate task condition (or task-rule); rows specify the miniblock index 

    Returns 
        ite_mean - The average information transfer estimate for a given prediction, averaged across all miniblocks.
    """

    nrules = labels.shape[1] # number of task conditions
    ncvs = labels.shape[0] # number of cross-validations; a leave-four-out cross validation in the case of the manuscript

    correct_matches = []
    incorrect_matches = []
    # Running cross-validation. Hold out one sample of each condition (leave-four-out) in each cross validation
    for cv in range(ncvs):
        
        # Obtain the *real* prototypes for each of the rules, but leave out the current trial (cv value)
        testset_ind = labels[cv,:]
        trainset_ind = np.delete(labels,cv,axis=0) # Delete the test set row from the train set
        corr = []
        err = []
        for cond1 in np.arange(nrules,dtype=int):
            # Find the miniblock we're comparing
            testmb = testset_ind[cond1]
            predicted_miniblock = predictions[testmb,:]

            # predicted-to-actual similarity
            for cond2 in np.arange(nrules,dtype=int):
                # Obtain specific miniblocks pertaining to cond2 condition
                trainmb = trainset_ind[:,cond2]
                trainmb = trainmb.astype('int')
                actualprototype = np.mean(actual[trainmb,:],axis=0) # average across training samples to obtain prototype

                r = stats.spearmanr(predicted_miniblock,actualprototype)[0]
                r = np.arctanh(r)
                # If condition matches
                if cond1==cond2:
                    corr.append(r)
                else:
                    err.append(r)

        # Get average matches for this cross-validation fold
        correct_matches.append(np.mean(corr))
        # Get average mismatches for this cross-validation fold
        incorrect_matches.append(np.mean(err))

    ite_mean = np.arctanh(np.mean(correct_matches)) - np.arctanh(np.mean(incorrect_matches))
    return ite_mean
