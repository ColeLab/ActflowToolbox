# Katelyn Arnemann
# May 14, 2019

# Compute latent connectivity using factor analysis.

import numpy as np
import pandas as pd

from rpy2.robjects.packages import importr
psych = importr('psych')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def factor_analysis(k, k_data, subjs, states, n_factors=1, fm='minres'):
    '''Factor analysis accomplished using the psych.fa function in R.
    
     Parameters
    ----------
    k : int
        Index of connection in data of shape=(subjects, states, connections).
    k_data : array-like, shape=(subjects, states)
        The data matrix of the connection k.
    subjs : list
        Subject IDs associated with the data matrix.
    states : list
        States associated with the data matrix.
    n_factors : int
        Optional: Number of factors in model.
    fm : str
        Optional: Factoring method. Default=minres. Choices: minres (minimum residual), ols (ordinary least squares), wls (weighted least squares), gls (generalized weighted least squares), pa (principal factor), ml (maximum likelihood, minchi (chi square), minrank (minimum rank), alpha (alpha). See documentation on psych.fa.

    Returns
    -------
    model_df : pandas.DataFrame, shape=(1, parameters)
        Parameters to evaluate model fit
    load_df : pandas.DataFrame, shape=(1, states)
        factor loadings   
    '''
    n_subjs, n_states = k_data.shape
    # initiate output
    model_cols = ['fit', 'dof', 'tli', 'rmsea_pval', 'rmsea_pval_lwrci', \
                'rmsea_pval_uprci']
    model_df = pd.DataFrame(columns=model_cols)
    load_df = pd.DataFrame(columns=states)
    # run factor analysis
    fa = psych.fa(r=k_data, nfactors=n_factors, fm=fm, n_obs=n_subjs)
    # get information about the model
    model_params = [float(fa.rx('fit')[0][0]), float(fa.rx('dof')[0][0]), \
                    float(fa.rx('TLI')[0][0]), float(fa.rx('RMSEA')[0][0]), \
                    float(fa.rx('RMSEA')[0][1]), float(fa.rx('RMSEA')[0][2])]
    model_df.loc[k] = model_params
    # get factor loadings
    load_df.loc[k] = np.array(fa.rx('loadings')[0]).T[0]
    return model_df, load_df

def run_factor_analysis(data, subjs, states, n_factors=1, fm='minres'):
    '''
    Runs factor analysis by connection.
    
    Parameters
    ----------
    data : array-like, shape=(subjects, states, connections)
        The data matrix. 
    subjs : list
        Subject IDs associated with the data matrix.
    states : list
        States associated with the data matrix.
    n_factors : int
        Optional: Number of factors in model.
    fm : str
        Optional: Factoring method. Default=minres. Choices: minres (minimum residual), ols (ordinary least squares), wls (weighted least squares), gls (generalized weighted least squares), pa (principal factor), ml (maximum likelihood, minchi (chi square), minrank (minimum rank), alpha (alpha). See documentation on psych.fa.

    Returns
    -------
    model_df : pandas.DataFrame, shape=(connections, parameters)
        Parameters to evaluate model fit.
    load_df : pandas.DataFrame, shape=(connections, states)
        Factor loadings.
    '''
    n_subjs, n_states, n_connections = data.shape
    # initiate output
    model_cols = ['fit', 'dof', 'tli', 'rmsea_pval', 'rmsea_pval_lwrci', \
                'rmsea_pval_uprci', 'rms']
    connections = np.arange(n_connections, dtype=int)
    model_df = pd.DataFrame(index=connections, columns=model_cols)
    load_df = pd.DataFrame(index=connections, columns=states)
    # run factor analysis separately for each connection k
    for k in connections:
        print('Running factor analysis on %i of %i connections' % (k, n_connections))
        k_data = data[:,:,k]# get data for connection k
        k_model_df, k_load_df = factor_analysis(k, k_data, subjs, states, n_factors, fm)
        model_df.loc[k] = k_model_df.loc[k]
        load_df.loc[k] = k_load_df.loc[k]
    return model_df, load_df

def run_loso_factor_analysis(data, subjs, states, n_factors=1, fm='minres'):
    '''
    Runs leave-one-state out factor analysis by connection, iteratively leaving out each state.
    
    Parameters
    ----------
    data : numpy.array
        The data matrix. shape=(subjects, states, connections)
    subjs : list
        Subject IDs associated with the data matrix.
    states : list
        States associated with the data matrix.
    n_factors : int
        Optional: Number of factors in model.
    fm : str
        Optional: Factoring method. Default=minres. Choices: minres (minimum residual), ols (ordinary least squares), wls (weighted least squares), gls (generalized weighted least squares), pa (principal factor), ml (maximum likelihood, minchi (chi square), minrank (minimum rank), alpha (alpha). See documentation on psych.fa.

    Returns
    -------
    model_df : pandas.DataFrame
        parameters to evaluate model fit, # connections x # parameters
    load_df : pandas.DataFrame
        factor loadings, # connections x # states
    '''
    n_subjs, n_states, n_connections = data.shape
    for state in states:
        loo_states = list(states)
        idx = loo_states.index(state)# get the index of the to be left-out state
        loo_idx = range(n_states)
        loo_idx.remove(idx)# remove the to be left-out state from the indices
        loo_states.remove(state)# remove the to be left-out state
        loo_data = data[:, loo_idx, :]# select data for all other states
        print('Running LOO factor analysis for', state)
        run_factor_analysis(loo_data, subjs, loo_states, n_factors, fm)
    return

def estimate_latent_connectivity(subj_data, load_df):
    '''
    Estimate latent connectivity for a given subject as the factor score computed by the weighted average (weights=factor loadings).

    Note: Alternate methods for estimating latent functional connectivity in psych.factor.scores.

    Parameters
    ----------
    subj_data : array-like, shape=(states, connections)
        A subjects data matrix.
    load_df : pandas.DataFrame, shape=(connections, states)
        Factor loadings.
    
    Returns
    -------
    latent_connectivity : array-like, shape=(n_connections)
    '''
    weighted_sums = []
    n_states, n_connections = subj_data
    for k in np.arange(n_connections):
        weights = np.multiply(load_df.loc[k], subj_data[:,k])#weighted sum by factor loading
        weighted_sums.append(np.sum(weights))
    return np.array(weighted_sums) / float(n_states)