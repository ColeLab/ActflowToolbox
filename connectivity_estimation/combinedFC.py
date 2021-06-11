'''
Implements the combinedFC connectivity method published in:
Sanchez-Romero, R., & Cole, M. W. (2020). Combining multiple functional connectivity methods to improve causal inferences. Journal of Cognitive Neuroscience, 1-15.
Toolbox to reproduce the results of the paper in https://github.com/ColeLab/CombinedFC.

Version: adjusted to handle separate target and source timeseries, for the purpose of using resting-state data with non-circular adjustment (10 mm dilated mask). 
'''

#modules
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import covariance
from scipy import stats, linalg

#Note: All the sub-functions called by combinedFC are in this file.

def combinedFC(dataset,target_ts=None,parcelInt=None,source_cols=None,methodCondAsso = 'partialCorrelation',methodParcorr='inverseCovariance',alphaCondAsso = 0.01,
               methodAsso = 'correlation',alphaAsso = 0.01,equivalenceTestAsso = False,lower_bound = -0.1,upper_bound = +0.1):
    '''
    INPUT:
        dataset : in dimension [nNodes x nDatapoints] actflow convention. When calling via calcconn_parcelwise_noncircular.py, 
                  dataset will = 'activity_matrix' (or source timeseries). 
        target_ts : Optional, used when only a single target time series (returns 1 X nnodes matrix). 
                    Used when calling via calcconn_parcelwise_noncircular.py as target timseries (see function for notes and details)
                    https://github.com/ColeLab/ActflowToolbox/.
                    NOTE: if using this, the recommended output formatting is: M[parcelInt,source_cols]
        parcelInt : Optional, only used with target_ts set and calling via calcconn_parcelwise_noncircular.py. This input helps indexing final network.
                    If using separate source and target timeseries and NOT calling via calcconn_parcelwise_noncircular.py, then this is the indexing value
                    of the target node. If there's only 1 target node, it can be set to 0. 
        source_cols : Optional, only used with target_ts set and calling via calcconn_parcelwise_noncircular.py. This input helps indexing final network. 
                      If using separate source and target timeseries and NOT calling via calcconn_parcelwise_noncircular.py, then this is the indexing vector
                      of the source nodes (with target held out). If there are no special corrections, this can be set like so...
                      ex if parcelInt is 0 and there are 360 source nodes: 
                      parcelInt=0; nSourceNodes=360; source_cols=np.arange(nSourceNodes); source_cols=np.delete(source_cols,parcelInt)
        methodCondAsso : a string "partialCorrelation" or "multipleRegression" for the conditional association
                         first step
        methodParcorr : a string if partial correlation is chosen, "inverseCovariance" or "regression"
        alphaCondAsso : alpha significance cutoff for the conditional association. Default = 0.01
        methodAsso : a string "correlation" or "simpleRegression" for the unconditional association.
        alphaAsso : alpha significance cutoff for the unconditional association. Default = 0.01
        equivalenceTestAsso : if True perform the equivalence test, otherwise perform the two-sided null
                              hypothesis test
        lower_bound : a negative bound for the minimum r of interest in the equivalence test. Default = -0.1
        upper_bound : a positive bound for the minimum r of interest in the equivalence test. Default = +0.1
            *note, the bounds are not required to be symmetric.
    OUTPUT:
        M: a connectivity matrix with significant partial correlation coefficients after
            removing possible spurious edges from conditioning on colliders.

    The recommended use of combinedFC with activity flow mapping involves first running combinedFC with partial correlation,
     followed by running multiple regression using only signficant combinedFC connections. This allows the combinedFC partial
     correlation matrix to be analyzed (e.g., using graph analyses), along with using the multiple regression weights to improve
     prediction accuracies with activity flow mapping. This is also very efficient in terms of compute time, since it uses
     inverse covariance and Pearson correlation with combinedFC (very fast) and reduces the number of predictors for the
     multiple regression step (which speeds up the regression step).
     Example of this use:
    combFC_output=actflow.connectivity_estimation.combinedFC(restdata)
    combFC_multregweights_output=actflow.connectivity_estimation.multregconn(restdata,conn_mask=(combFC_output!=0))
    '''

    #transpose the dataset to [nDatapoints x nNodes] combinedFC convention
    if target_ts is None:
        dataset = np.transpose(dataset)
    else:
        # Set up for computation via separate target and source timeseries. 
        # Note: this requires more inputs than other methods (eg, multregconn) because the procedure requires more specification (specifically step 1: finding precision matrix)
        nParcels = 360; # Glasser et al., 2016 MMP atlas for cortical regions only. ### TO-DO: maybe make this an input so subcortical can be an option? 
        sourceDataOrig = dataset.copy().T; 
        targetData = target_ts.copy(); 
        targetNode = parcelInt; # Only if target_ts is set, otherwise set to None and ignored.
        sourceColsHere = source_cols; # Only if target_ts is set, otherwise set to None and ignored.
        
        # Set up indexing vector for sources; for when calcconn_parcelwise_noncircular.py is used and some sources are completely masked (ie "empties");
        # Note: this is flexible to inputing data as separate source & target timeseries, but with no "empties" (see INPUT notes at top for proper handling). 
        tempSources = sourceDataOrig.copy(); 
        tempTarget = targetData.copy(); 
        tempIxVec = np.arange(tempSources.shape[1]+1); 
        tempIxVec = np.delete(tempIxVec,targetNode); 
        sourceData = np.zeros((tempSources.shape[0],tempSources.shape[1]+1)); 
        sourceData[:,targetNode] = tempTarget; 
        sourceData[:,tempIxVec] = tempSources; 
        
        # Helper code for number of sources etc.
        sourceNodes = np.ones(nParcels, dtype=np.bool); excludeSources = [];
        for sourceNum in range(nParcels):
            if np.where(sourceColsHere==sourceNum)[0].shape[0]==0 and sourceNum!=targetNode: 
                sourceNodes[sourceNum] = False; excludeSources.append(sourceNum); 
        sourceIxVecHere = np.arange(nParcels); sourceIxVecHere = np.delete(sourceIxVecHere,excludeSources); nSourceNodesHere = sourceIxVecHere.shape[0];

        # The data for each target node analysis will contain the non-zero source nodes and the target node 
        # always in the proper column for easy indexing (done above)
        dataset = sourceData.copy(); 
        numVar = dataset.shape[1]; 
        nDatapoints = dataset.shape[0];

    #first step: evaluate full conditional associations
    if methodCondAsso == 'partialCorrelation':
        #partial correlation with a two-sided null hypothesis test for the Ho: parcorr = 0
        #using the method chosen by the user
        Mca = partialCorrelationSig(dataset, alpha=alphaCondAsso, method=methodParcorr)
        if target_ts is not None:
            McaThisTarget = Mca[targetNode,:]; 
            
    if methodCondAsso == 'multipleRegression':
        #multiple regression for each node x on the rest of the nodes in the set
        #with a two-sided t-test for the Ho : beta = 0
        Mca = multipleRegressionSig(dataset, alpha=alphaCondAsso, sigTest=True)
        if target_ts is not None:
            McaThisTarget = Mca[targetNode,:]; 

    nNodes = dataset.shape[1]
    #second step
    #start with the conditional association matrix, and then make the collider check using correlation
    M = Mca.copy();
    if target_ts is not None:
        M_CopyThisTarget = McaThisTarget.copy();

    if methodAsso == 'correlation' and equivalenceTestAsso == True:
        #correlation with the equivalence test for r = 0
        Mcorr = correlationSig(dataset, alpha=alphaAsso, lower_bound=lower_bound,
                            upper_bound=upper_bound, equivalenceTest=True)
        #test if two nodes have a significant partial correlation but a zero correlation
        #this will be evidence of a spurious edge from conditioning on a collider
        if target_ts is not None:
            McorrThisTarget = Mcorr[targetNode,:]; 
            for sourceNum in range(nSourceNodesHere):
                if McaThisTarget[sourceNum]!=0 and McorrThisTarget[sourceNum]==0:
                    M_CopyThisTarget[sourceNum] = 0; #remove the edge [x,y] from the connectivity network based on multiple regression or partial correlation
            M = M_CopyThisTarget.copy();
        else: 
            for x in range(nNodes-1):
                for y in range(x+1,nNodes):
                     if Mca[x,y] != 0 and Mcorr[x,y] == 0:
                        M[x,y] = M[y,x] = 0 #remove the edge from the connectivity network


    elif methodAsso == 'correlation' and equivalenceTestAsso == False:
        #correlation with a two-sided null hypothesis test for the null hypothesis Ho: r = 0
        Mcorr = correlationSig(dataset, alpha=alphaAsso, equivalenceTest=False)
        #test if two nodes have a significant partial correlation but a not significant correlation
        #this will be evidence of a spurious edge from conditioning on a collider
        if target_ts is not None:
            McorrThisTarget = Mcorr[targetNode,:]; 
            for sourceNum in range(nSourceNodesHere):
                if McaThisTarget[sourceNum]!=0 and McorrThisTarget[sourceNum]==0:
                    M_CopyThisTarget[sourceNum] = 0; #remove the edge [x,y] from the connectivity network based on multiple regression or partial correlation
            M = M_CopyThisTarget.copy(); 
        else:
            for x in range(nNodes-1):
                for y in range(x+1,nNodes):
                    if Mca[x,y] != 0 and Mcorr[x,y] == 0:
                        M[x,y] = 0 #remove the edge from the connectivity network
                    if Mca[y,x] != 0 and Mcorr[y,x] == 0:
                        M[y,x] = 0
                        
    elif methodAsso == 'simpleRegression':
        #simple regression for each pair of nodes that have a significant conditional association
        if target_ts is not None:
            for sourceNum in range(nSourceNodesHere):
                sourceNodeHere = sourceIxVecHere[sourceNum];
                if McaThisTarget[sourceNum]!=0:
                    b = simpleRegressionSig(dataset[:,targetNode],dataset[:,sourceNodeHere],alpha=alphaAsso,sigTest=True);
                    if b==0:
                        M_CopyThisTarget[sourceNum] = 0;
            M = M_CopyThisTarget.copy();
        else: 
            for x in range(nNodes-1):
                for y in range(x+1,nNodes):
                    #do both sides, regression coefficients are not symmetric
                    if Mca[x,y] != 0:
                        b = simpleRegressionSig(dataset[:,x],dataset[:,y],alpha=alphaAsso,sigTest=True)
                        if b == 0:
                            M[x,y] = 0 #remove the edge from the connectivity network
                    if Mca[y,x] != 0:
                        b = simpleRegressionSig(dataset[:,y],dataset[:,x],alpha=alphaAsso,sigTest=True)
                        if b == 0:
                            M[y,x] = 0
                            
    if target_ts is not None:
        tempVec = M.copy(); tempVec = np.delete(tempVec,targetNode); M = tempVec.copy();
        
    return M


#compute the partial correlation matrix and only keep the statistically significant coefficients
#according to a two-sided null hypothesis test at the chosen alpha.
#significant partial correlation coefficients represent edges in the connectivity network.

def partialCorrelationSig(dataset, alpha = 0.01, method = 'inverseCovariance'):
    '''
    INPUT:
        dataset : in dimension [nDatapoints x nNodes]
        alpha : cutoff for the significance decision. Default is 0.01
        method : a string, 'regression','inverseCovariance'
    OUTPUT:
        M : a connectivity network of significant partial correlations
    '''

    D = dataset
    nNodes = D.shape[1] #number of variables in the dataset
    nDatapoints = D.shape[0] #number of datapoints

    if method == 'regression':
        #compute the partial correlation matrix using the regression approach
        Mparcorr = parCorrRegression(D)
        condSetSize = nNodes-2

    elif method == 'inverseCovariance':
        #compute the partial correlation matrix using the inverse covariance approach
        Mparcorr = parCorrInvCov(D)
        condSetSize = nNodes-2

    #two-sided null hypothesis test of partial correlation = 0.
    #get the Zalpha cutoff
    Zalpha = Zcutoff(alpha = alpha, kind = 'two-sided')
    #Fisher z-transformation of the partial correlation matrix for the null hypothesis Ho: parcorr = 0
    #partial correlation of 2 nodes conditions on all the rest of the nodes,
    #so the size of the conditioning set is the number of total nodes minus 2.
    Fz = fisherZTrans(Mparcorr, nDatapoints=nDatapoints, Ho=0, condSetSize=condSetSize)
    #threshold the par.corr. matrix using the significant decision abs(Fz) >= Zalpha
    M = np.multiply(Mparcorr, abs(Fz) >= Zalpha) + 0 #+0 is to avoid -0 in the output


    return M


#compute the partial correlation using the regression approach.
#https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
#In this approach, for two nodes X and Y, the partial correlation conditions on all the nodes in the dataset except X and Y.

def parCorrRegression(dataset):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
    OUTPUT:
        M : matrix of partial correlation coefficients
    '''
    D = dataset
    nNodes = D.shape[1]
    #allocate memory
    M = np.zeros((nNodes,nNodes))
    #compute the partial correlation of x and each remaining variable y, conditioning on all except x and y
    for x in range(nNodes-1):
        for y in range(x+1, nNodes):
            #create some indices
            idx = np.ones(nNodes, dtype=np.bool)
            #to not include x and y on the regressors
            idx[x] = False
            idx[y] = False

            #regressed out the rest of the variables from x and from y, independently
            reg_x = LinearRegression().fit(D[:,idx], D[:,x])
            reg_y = LinearRegression().fit(D[:,idx], D[:,y])
            #compute the residuals for x and for y
            #residual = x - Z*B_hat,
            #where B_hat are the estimated coefficients and Z the data for the rest of the variables
            res_x = D[:,x] - np.dot(D[:,idx],reg_x.coef_)
            res_y = D[:,y] - np.dot(D[:,idx],reg_y.coef_)
            #compute the correlation of the residuals which are equal to
            #the partial correlation of x and y conditioning on the rest of the variables
            parcorr = np.corrcoef(res_x, res_y, rowvar=False)[0,1]
            #partial_correlation is symmetric, meaning that:
            M[x,y] = M[y,x] = parcorr

    return M


#compute the partial correlation using the inverse covariance approach.
#The partial correlation matrix is the negative of the off-diagonal elements of the inverse covariance,
#divided by the squared root of the corresponding diagonal elements.
#https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
#in this approach, for two nodes X and Y, the partial correlation conditions on all the nodes except X and Y.

def parCorrInvCov(dataset):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
    OUTPUT:
        M : matrix of partial correlation coefficients
    '''
    D = dataset
    #compute the covariance matrix of the dataset and invert it. This is known as the precision matrix.
    #use the (Moore-Penrose) pseudo-inverse of a matrix.
    invCovM = linalg.pinv(np.cov(D,rowvar=False))
    #transform the precision matrix into partial correlation coefficients
    denom = np.atleast_2d(1. / np.sqrt(np.diag(invCovM)))
    M = -invCovM * denom * denom.T
    #make the diagonal zero.
    np.fill_diagonal(M,0)

    return M

#compute the multiple regression for each node Xi on the rest of the variable set V\{Xi}, V={X1,...,Xp}
#e.g. X1 = b0 +b2X2 + b3X3 + ... + bpXp + E1
#build a connectivity matrix M where each row contains the Betas for the regression of Xi on the rest of the variables
#significant regression betas represent edges in the connectivity model.
#non-significant betas are zeroed out

def multipleRegressionSig(dataset, alpha = 0.01, sigTest = False):
    '''
    INPUT:
        dataset : in dimension [nDatapoints x nNodes]
        alpha : cutoff for the significance decision. Default is 0.01
        sigTest : if True, perform the t-test of significance for the Beta coefficients
    OUTPUT:
        M : a connectivity network of beta coefficients. If sigTest = True, only significant betas
            Note: It is not a symmetric matrix.
    '''


    D = dataset
    nNodes = D.shape[1] #number of variables
    nDatapoints = D.shape[0]

    M = np.zeros((nNodes,nNodes))
    for x in range(nNodes):
        #create some indices
        idx = np.ones(nNodes, dtype=np.bool)
        #to not include x on the set of regressors, ie. V\{x}
        idx[x] = False
        #regressed x on the rest of the variables in the set
        reg_x = LinearRegression().fit(D[:,idx], D[:,x])

        if sigTest == True:
            #parameters estimated =  intercept and the beta coefficients
            params = np.append(reg_x.intercept_,reg_x.coef_)
            #number of parameters estimated
            nParams = len(params)
            #obtain predicted data x
            x_hat = reg_x.predict(D[:,idx])
            #append a column of 1s (for intercept) to the regressors dataset
            newR = np.append(np.ones((nDatapoints,1)),D[:,idx],axis=1)
            #see chapter 12 and 13 of Devore's Probability textbook
            #mean squared errors MSE = SSE/(n-k-1), where k is the number of covariates
            #pg.519 Devore's
            MSE = (np.sum(np.square(D[:,x] - x_hat)))/(nDatapoints - nParams)
            #compute variance of parameters (intercept and betas)
            var_params = MSE*(np.linalg.inv(np.dot(newR.T,newR)).diagonal())
            #compute standard deviation
            std_params = np.sqrt(var_params)
            #transform parameters into t-statistics under the null of B = 0
            Bho = 0 #beta under the null
            ts_params = (params - Bho)/std_params
            #p-value for a t-statistic in a two-sided one sample t-test
            p_values = 2*(1-stats.t.cdf(np.abs(ts_params),df = nDatapoints-1))

            #remove the intercept p-value
            p_values = np.delete(p_values,0)
            #record the Betas with p-values < alpha
            M[x,idx] = np.multiply(reg_x.coef_, p_values < alpha)

        if sigTest == False:
            #save the beta coefficients in the corresponding x row without significance test
            M[x,idx] = reg_x.coef_

    return M


#Compute correlation r and use an equivalence test to determine if r is equal to zero,
#Lakens, D. (2017), Equivalence Test... 8(4), 355-362.
#or a null hypothesis test to determine if r is different from zero,
#a significant r different from zero represents an edges in the connectivity network.

def correlationSig(dataset,
                   alpha = 0.01,
                   lower_bound = -0.1,
                   upper_bound = +0.1,
                   equivalenceTest = False):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
        alpha : significance level for the equivalence test or for the two-sided null hypothesis test. Default = 0.01
        lower_bound : a negative bound for the minimum r of interest in the equivalence test. Default = -0.1
        upper_bound : a positive bound for the minimum r of interest in the equivalence test. Default = +0.1
            *note, the bounds are not required to be symmetric.
        equivalenceTest = if True perform the equivalence test, otherwise perform the two-sided null hypothesis test
    OUTPUT:
        M : a matrix such that:
            if equivalenceTest = True, M contains r values judged to be zero, according to the equivalence test.
            if equivalenceTest = False, M contains r values judged different from zero, as per the null hypothesis test.
    '''

    D = dataset
    nNodes = D.shape[1] #number of variables
    nDatapoints = D.shape[0] #number of datapoints

    #compute the full correlation matrix
    Mcorr = np.corrcoef(D, rowvar=False)

    if equivalenceTest == False:
        #get the Zalpha cutoff
        Zalpha = Zcutoff(alpha = alpha, kind = 'two-sided')
        #make the correlation matrix diagonal equal to zero to avoid problems with Fisher z-transformation
        np.fill_diagonal(Mcorr,0)
        #Fisher z-transformation of the correlation matrix for the null hypothesis of Ho: r = 0
        Fz = fisherZTrans(Mcorr, nDatapoints = nDatapoints, Ho = 0)
        #threshold the correlation matrix judging significance if: abs(Fz) >= +Zalpha
        #(this is equivalent to test Fz >= +Zalpha or Fz <= -Zalpha)
        M = np.multiply(Mcorr, abs(Fz) >= Zalpha)+0 #+0 is to avoid -0 in the output


    elif equivalenceTest == True:
        #the equivalence test is formed by two one-sided tests
        #Zalpha cutoffs for two one-sided test at a chosen alpha
        #upper bound
        Zalpha_u = Zcutoff(alpha = alpha, kind = 'one-sided-left')
        #lower bound
        Zalpha_l = Zcutoff(alpha = alpha, kind = 'one-sided-right')
        #make the correlation matrix diagonal equal to zero to avoid problems with Fisher z-transformation
        np.fill_diagonal(Mcorr,0)
        #Fisher z-transform using Ho: r = upper_bound, Ha: r < upper_bound
        Fz_u = fisherZTrans(Mcorr, nDatapoints = nDatapoints, Ho = upper_bound)
        #and Fisher z-transform using Ho: r = lower_bound, Ha: r > lower_bound
        Fz_l = fisherZTrans(Mcorr, nDatapoints = nDatapoints, Ho = lower_bound)
        #Fz_u = -Fz_l, expressing it as two variables is just for clarity of exposition.

        #threshold the correlation matrix judging significantly equal to zero if:
        #Fz_u <= Zalpha_u & Fz_l >= Zalpha_l
        #if both inequalities hold then we judge Fz ~ 0, and thus r ~ 0
        #M contains the correlation values r that were judged close to zero in the equivalence test
        M = np.multiply(Mcorr, np.multiply(Fz_u <= Zalpha_u, Fz_l >= Zalpha_l))+0

    return M


#compute the linear simple regression Y = a + bX + E
#and a significance test for the b coefficient

def simpleRegressionSig(y, x, alpha = 0.01, sigTest = True):
    '''
    INPUT:
        y : vector [nDatapoints x 1] with node y data, for the regression y = a + bx + E
        x : vector [nDatapoints x 1] with regressor node x data
        alpha : cutoff for the significance decision. Default is 0.01
        sigTest : if True, perform the t-test of significance for the Beta coefficients
    OUTPUT:
        b : the b regression coefficient. If sigTest = True, return b if significant, return 0 for non-significant

    '''

    #check dimensions, if < 2, expand to get [nDatapoints x 1]
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)

    nDatapoints = y.shape[0]

    reg_y = LinearRegression().fit(x, y)

    if sigTest == True:
        #parameters estimated =  intercept and the beta coefficients
        params = np.append(reg_y.intercept_,reg_y.coef_)
        #number of parameters estimated
        nParams = len(params)
        #obtain predicted data y
        y_hat = reg_y.predict(x)
        #append a column of 1s (for intercept) to the regressor data
        newR = np.append(np.ones((nDatapoints,1)), x, axis=1)
        #mean squared errors MSE, adjusted by 1/(datapoints - num.parameters)
        MSE = (np.sum(np.square(y - y_hat)))/(nDatapoints - nParams)
        #compute variance of parameters (intercept and betas)
        var_params = MSE*(np.linalg.inv(np.dot(newR.T,newR)).diagonal())
        #compute standard deviation
        std_params = np.sqrt(var_params)
        #transform parameters into t-statistics
        ts_params = params/std_params
        #p-value for a t-statistic in a two-sided one sample t-test
        p_values = [2*(1-stats.t.cdf(np.abs(i),df = nDatapoints-1)) for i in ts_params]

        #remove the intercept p-value
        p_value = np.delete(p_values,0)
        #record the Betas with p-values < alpha
        b = np.multiply(reg_y.coef_, p_value < alpha)

        return b, p_value

    if sigTest == False:
        #save the beta coefficients in the corresponding x row without significance test
        b = reg_y.coef_

        return b


#Computes the Zalpha cutoff value from a standard normal distribution N(mean=0,std.dev=1)
#that is used to reject or not the null hypothesis Ho.
#the function works for single values or matrices.

def Zcutoff(alpha = 0.01, kind = 'two-sided'):
    '''
    INPUT:
        alpha : level of significance, a value between (1 and 0). Lower alpha implies higher Zalpha. Default = 0.01.
        kind : 'two-sided', 'one-sided-right' or 'one-sided-left'. Depending on the alternative hypothesis Ha. See below.
    OUTPUT:
        Zalpha : the Z cutoff statistic for the input alpha and kind of alternative hypothesis Ha.
    '''

    if kind == 'two-sided':
        #For null Ho: r = 0 and alternative Ha: r != 0, where r  is correlation/partial correlation coefficient.
        #using cumulative distribution function (cdf) of the standard normal
        #alpha = 2*(1-cdf(Zalpha)), solve for Zalpha = inverse_of_cdf(1-alpha/2)
        #Zalpha defines the null hypothesis rejection regions such that:
        #reject the null hypothesis Ho if Fz >= +Zalpha or Fz <= -Zalpha
        #Fz is the Fisher z-transform of the r value observed
        #(from scipy use stats.norm.ppf to compute the inverse_of_cdf)
        Zalpha = stats.norm.ppf(1-alpha/2,loc=0,scale=1)

    elif kind == 'one-sided-right':
        #For null Ho: r = 0 and alternative Ha: r > 0
        #alpha = 1 - cdf(Zalpha), solve for Zalpha = inverse_of_cdf(1-alpha)
        #reject the null hypothesis if Fz >= +Zalpha
        Zalpha = stats.norm.ppf(1-alpha,loc=0,scale=1)

    elif kind == 'one-sided-left':
        #For null Ho: r = 0 and alternative Ha: r < 0
        #alpha = cdf(Zalpha), solve for Zalpha = inverse_of_cdf(alpha)
        #reject the null hypothesis if Fz <= -Zalpha
        Zalpha = stats.norm.ppf(alpha,loc=0,scale=1)

    return Zalpha


#Fisher z-transformation Fz, for the correlation/partial correlation coefficient r
#The statistic Fz is approximately distributed as a standard normal ~ N(mean=0,std.dev=1)
#the function works for single values of matrices.

def fisherZTrans(r, nDatapoints, Ho=0, condSetSize=0):
    '''
    INPUT:
        r : correlation(or partial correlation) coefficient or matrix of coefficients
        Ho : value for the null hypothesis of r, default Ho: r = 0
        nDatapoints : sample size
        condSetSize : the size of the conditioning set of the partial correlation for the Fisher z transform.
            more info:  https://en.wikipedia.org/wiki/Partial_correlation#As_conditional_independence_test
            For correlation the condSetSize = 0, for partial correlation condSetSize > 0.
    OUTPUT:
        Fz: a single Fisher z-transform value or matrix of values
    '''

    Fz = np.multiply(np.subtract(np.arctanh(r),np.arctanh(Ho)), np.sqrt(nDatapoints-condSetSize-3))

    return Fz