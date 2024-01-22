import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import time
import scipy.io
import scipy.stats
import bdb
import mat73


def compute_pcaFC_opt(data_input, use_PCA, opt_mode, target_regions, model_order, zscore_tseries, include_contemp, include_autoreg, outfile, cv_max, cv_increment, cv_run, sub_id, out_dir):
    data_input = np.asarray(data_input)
    use_PCA = int(use_PCA)
    opt_mode = int(opt_mode)
    target_regions = np.asarray(target_regions)
    model_order = int(model_order)
    zscore_tseries = int(zscore_tseries)
    include_contemp = int(include_contemp)
    include_autoreg = int(include_autoreg)
    cv_max = int(cv_max)
    cv_increment = int(cv_increment)
    cv_run = int(cv_run)
    sub_id = int(sub_id)
    out_dir = str(out_dir)

<<<<<<< HEAD
=======
    print("data input : ", data_input)
    print("use_PCA : ", use_PCA)
    print("opt_mode : ", opt_mode)
    print("target_regions : ", target_regions)
    print("model_order : ", model_order)
    print("zscore_tseries : ", zscore_tseries)

    print("include_contemp : ", include_contemp)
    print("include_autoreg : ", include_autoreg)
    print("outfile : ", outfile)
    print("cv_max : ", cv_max)
    print("cv_increment : ", cv_increment)
    print("cv_run : ", cv_run)
    print("sub_id : ", sub_id)
    print("out_dir : ", out_dir)
    print("Here")

>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
    num_trials = len(data_input)
    num_regions = data_input[0].shape[0]
    num_target_regions = len(target_regions)
    num_tp = data_input[0].shape[1]

    if 'cv_max' not in locals():
        cv_max = None
    if 'cv_increment' not in locals():
        cv_increment = None
    if 'cv_run' not in locals():
        cv_run = None
    if 'sub_id' not in locals():
        sub_id = None
    if 'out_dir' not in locals():
        out_dir = None

    bdb.set_trace()
    # Set MVAR parameters
    if include_contemp == 1:
        increments = model_order + 1  # Lags will include t0 term
    else:
        increments = model_order

    # Set num_predictors
    num_predictors = (num_regions - 1) * increments
    if include_autoreg == 1:
        num_predictors += model_order  # Add lags for target autoregs, excluding t0 contemporaneous term

    # Set up output variables
    trial_data = np.zeros((num_tp, num_regions, num_trials))
    sub_restFC = np.zeros((num_regions, num_regions, num_trials))
    sub_restMVAR = np.zeros((num_predictors, num_target_regions, num_trials))
    sub_restMVAR_viz = np.zeros((num_predictors + increments, num_target_regions, num_trials))
    start_time = time.time()
    # Set up 3d trial_data array (zscore each region per trial if requested); also compute corrFC (static)
    for t in range(num_trials):
        # Rearrange into cols=tseries, rows=regions
        trial_tseries = data_input[t].T

        # Z-score
        if zscore_tseries == 1:
            trial_tseries = (trial_tseries - np.mean(trial_tseries, axis=0)) / np.std(trial_tseries, axis=0)

        trial_data[:, :, t] = trial_tseries

        # Compute restFC and add to sub
        r = np.corrcoef(trial_tseries, rowvar=False)
        sub_restFC[:, :, t] = r

    # Run PCA optimization (if requested)
    nObs = num_tp - model_order  # Take into account first timepoint which is model_order + 1
    nVars = num_predictors

    # Initialize optimization variables - *set maxPC first
    maxPC = min(nVars, nObs - 1) if nObs > nVars else nVars

<<<<<<< HEAD
=======
    # # Reshape trial_data for PCA
    # trial_data_reshaped = trial_data.reshape((num_tp, -1))

    # # Perform PCA
    # pca = PCA(n_components=maxPC)
    # pca.fit(trial_data_reshaped)

    # # Extract PCA components
    # pca_components = pca.components_.T

>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
    MVAR_PCopt = {}
    # set up PC range and output vars
    if use_PCA == 3:
        PC_range = list(range(0, cv_max + 1, cv_increment))
        # replace 0 with 1
        PC_range[0] = 1

        # v4: sets which nPC is estimated for this job
        nextPC_ind = cv_run  # sets which pc will be looped over

        # note num_cv_loops is num_trials - 1
        num_cv_loops = num_trials - 1
        MVAR_PCopt['nPC'] = np.zeros((1, 3, num_target_regions, num_cv_loops))  # array with dimensions: PCnum(1:maxPC), MSE, PCA variance explained; number target regions, number trials)
        MVAR_PCopt['optPC'] = []  # PCnum, MSE, PC variance explained, for the optimal PCnum i.e. that which yields the minimum MSE after averaging nPC across regions and trials

        # initialize next* variables
        nextTrial_ind = 1
        nextTarget_ind = 1

    else:
        # initialize even if not running optimization to save output more consistently
        MVAR_PCopt['nPC'] = np.zeros((maxPC, 3, num_target_regions, num_trials))  # array with dimensions: PCnum(1:maxPC), MSE, PCA variance explained; number target regions, number trials)
        MVAR_PCopt['optPC'] = []  # PCnum, MSE, PC variance explained, for the optimal PCnum i.e. that which yields the minimum MSE after averaging nPC across regions and trials


    # set up PC range and output vars
    if use_PCA == 3:
        if opt_mode == 1:  # run opt

            first_tpoint = model_order + 1  # must have enough lags prior to first_tpoint
            num_tp_reg = num_tp - first_tpoint + 1
            num_sources = num_regions - 1

            # loop through and assemble pca and x/y mats, if these haven't been assembled already
            # need to save each trial PCVar to its own .mat file to avoid out of memory errors
            pc_dir = os.path.join(out_dir, 'FullPCvars')
            if not os.path.exists(pc_dir):
                os.makedirs(pc_dir)

            for t in range(nextTrial_ind - 1, num_trials):
                pc_out = os.path.join(pc_dir, f"{sub_id}_fullPCvars_trial{t + 1}.npz")
                if not os.path.exists(pc_out):
                    # run through all trials/targets, compute PCA and store (more time-efficient than computing PCA for each nPC loop)
                    coef_mat = np.zeros((num_predictors, cv_max, num_target_regions))
                    scores_mat = np.zeros((num_tp_reg, cv_max, num_target_regions))
                    ex_mat = np.zeros((maxPC, num_target_regions))
                    X_mat = np.zeros((num_tp_reg, num_predictors, num_target_regions))
                    y_mat = np.zeros((num_tp_reg, num_target_regions))

                    trial_tseries = trial_data[:, :, t]
                    for r1 in range(nextTarget_ind - 1, num_target_regions):
                        source_inds = list(range(num_regions))  # for looping later
                        target_r = int(target_regions[r1])
                        source_inds[target_r] = []

                        # init
                        y = np.empty((0, 1))
                        X = np.empty((0, num_predictors))
                        for tp in range(first_tpoint, num_tp + 1):
                            # assign y
                            y = np.vstack((y, trial_tseries[tp - 1, target_r]))

                            # set up X, format is:
                            # 1. all other regions over all lags 1->n (incl contemp if applicable; e.g. region1: t0, t0-m -> model order, region 2: t0, t0-m -> model order etc
                            # 2. autoregs over lags (if applicable)
                            # 3. constant
                            tpoint_X = []
                            for rr in source_inds:
                                # add contemp if necessary
                                if include_contemp == 1:
                                    tpoint_X.append(trial_tseries[tp - 1, rr])

                                # add lags
                                for lag in range(1, model_order + 1):
                                    tpoint_X.append(trial_tseries[tp - lag - 1, rr])

                            # add autoregs for target over lags
                            if include_autoreg == 1:
                                for lag in range(1, model_order + 1):
                                    tpoint_X.append(trial_tseries[tp - lag - 1, target_r])

                            # assign to X
                            X = np.vstack((X, tpoint_X))

                        # demean prior to PCA
                        X = X - np.mean(X, axis=0)
                        # run pca (up till max range)
                        pca = PCA(n_components=cv_max)
                        pca.fit(X)
                        Loadings = pca.components_.T
                        Scores = pca.transform(X)
                        explained = pca.explained_variance_ratio_

                        # add to mats
                        coef_mat[:, :, r1] = Loadings
                        scores_mat[:, :, r1] = Scores
                        ex_mat[:, r1] = explained
                        X_mat[:, :, r1] = X
                        y_mat[:, r1] = y

                        print(f'Time taken to assemble trial x target vars for PCA opt, for target {r1 + 1}, trial {t + 1} = {time.time() - start_time}')

                    # save
                    np.savez_compressed(pc_out, coef_mat=coef_mat, scores_mat=scores_mat, ex_mat=ex_mat, X_mat=X_mat, y_mat=y_mat)

            # loop through number of PCs and number of targets and compute opt metrics: cross-trial MSE and PCexplained
            for pc in range(nextPC_ind, nextPC_ind + 1):
                nPCs = pc

                # set output file for this nPC
                nPC_out = os.path.join(out_dir, f"{sub_id}_out_nPC{nPCs}.npz")

                for r1 in range(nextTarget_ind - 1, num_target_regions):

                    # loop through folds training/testing MVAR model
                    for t in range(1, num_cv_loops + 1):
                        # assign train/test trial
                        train_trial = t
                        test_trial = t + 1

                        # load in train trial and assign appropriate vars
                        train_mat = os.path.join(pc_dir, f"{sub_id}_fullPCvars_trial{train_trial}.npz")
                        with np.load(train_mat) as data:
                            PCAScores = data['scores_mat'][:, :nPCs, r1]
                            PCAScores = np.hstack((PCAScores, np.ones((PCAScores.shape[0], 1))))
                            PCALoadings = data['coef_mat'][:, :nPCs, r1]
                            PCAEx = np.sum(data['ex_mat'][:nPCs, r1])
                            y = data['y_mat'][:, r1]

                        # set up test vars
                        test_mat = os.path.join(pc_dir, f"{sub_id}_fullPCvars_trial{test_trial}.npz")
                        with np.load(test_mat) as data:
                            X_test = data['X_mat'][:, :, r1]
                            y_test = data['y_mat'][:, r1]

                        # regress
                        lr = LinearRegression()
                        lr.fit(PCAScores, y)
                        b = lr.coef_
                        b_constant = lr.intercept_

                        # transform back to predictors
                        b = PCALoadings.dot(b)

                        # loop through test tp and use b to predict test data, and store actual and pred data
                        actual_data = []
                        pred_data = []
                        for tt in range(num_tp_reg):
                            actual_data.append(y_test[tt])
                            # predict
                            pred = np.dot(X_test[tt, :], b) + b_constant
                            pred_data.append(pred)

                        # compute mse
                        err = mean_squared_error(actual_data, pred_data)
                        pc_dat = [nPCs, err, PCAEx]
                        MVAR_PCopt['nPC'][0, :, r1, t - 1] = pc_dat

                        print(f'Time  taken to estimate OPTIMIZED MVAR for region {r1 + 1}, trial loop {t}, pc {nPCs} = {time.time() - start_time}')

                # save output for this pc
                np.savez_compressed(nPC_out, MVAR_PCopt=MVAR_PCopt)

        elif opt_mode == 2:
            # load in nPC_out over full PC_range
            MVAR_PCopt_full = np.zeros((len(PC_range), 3, num_target_regions, num_cv_loops))
        
            for n in range(len(PC_range)):
                nPCs = PC_range[n]
                nPC_out = os.path.join(out_dir, f"{sub_id}_out_nPC{nPCs}.mat")
<<<<<<< HEAD
                data = mat73.loadmat(nPC_out)
=======
                print("Here4")
                data = mat73.loadmat(nPC_out)
                print("Here43")
>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
                MVAR_PCopt_full[n, :, :, :] = data['MVAR_PCopt']['nPC']
            
            # reassign to MVAR_PCopt.nPC for consistency with below
            MVAR_PCopt['nPC'] = MVAR_PCopt_full
            
            # identify optimal nPCs (min over avg MSE across trials and targets) and add that row to MVAR_PCopt.optPC
            mean_opts = np.mean(np.mean(MVAR_PCopt['nPC'], axis=3), axis=2)
            opt_mse, ind = np.min(mean_opts[:, 1]), np.argmin(mean_opts[:, 1])
            MVAR_PCopt['optPC'] = mean_opts[ind, :]

    if opt_mode == 2:
        if use_PCA == 3:
            nPCs = int(MVAR_PCopt['optPC'][0])  # first column = optimal number of PCs after crossval
        elif use_PCA > 3:
            nPCs = use_PCA

        # also save PCA var explained
        sub_varEx = np.zeros((num_trials, num_target_regions))

        # loop through trials and target regions and compute pca regression
        for t in range(num_trials):
<<<<<<< HEAD
=======
            print(f"Trial {t + 1} of {num_trials}")
>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
            trial_MVAR = np.zeros((num_predictors, num_target_regions))
            trial_MVAR_viz = np.zeros((num_predictors + increments, num_target_regions))
            trial_varEx = []

            trial_tseries = trial_data[:, :, t]

            for r1 in range(num_target_regions):
                first_tpoint = model_order + 1  # must have enough lags prior to first_tpoint

                source_inds = list(range(num_regions))  # for looping later
                target_r = int(target_regions[r1])
                source_inds.remove(target_r)

                y = []
                X = []

                for tp in range(first_tpoint, num_tp + 1):
                    # assign y
                    y.append(trial_tseries[tp - 1, target_r])

                    # set up X
                    tpoint_X = []
                    for rr_ind in source_inds:
                        # add contemp if necessary
                        if include_contemp == 1:
                            tpoint_X.append(trial_tseries[tp - 1, rr_ind])

                        # add lags
                        for lag in range(1, model_order + 1):
                            tpoint_X.append(trial_tseries[tp - lag - 1, rr_ind])

                    # add autoregs for target over lags
                    if include_autoreg == 1:
                        for lag in range(1, model_order + 1):
                            tpoint_X.append(trial_tseries[tp - lag - 1, target_r])

                    # assign to X
                    X.append(tpoint_X)

                # Convert lists to NumPy arrays
<<<<<<< HEAD
=======
                # y = np.array(y)
                # X = np.array(X, dtype=object)
>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
                # Demean prior to PCA
                X = X - np.mean(X, axis=0)
                # Run PCA
                pca = PCA(n_components=nPCs)
                PCAScores = pca.fit_transform(X)
                PCALoadings = pca.components_
                explained = pca.explained_variance_ratio_

                # check for rank deficiency of PCAScores
                PCA_rank = np.linalg.matrix_rank(PCAScores)

                if PCA_rank < nPCs:
                    PCAScores = PCAScores[:, :PCA_rank]
                    PCALoadings = PCALoadings[:PCA_rank, :]
                    print(f"****WARNING: rank of PCAScores (rank={PCA_rank}) is lower than retained PCs ({nPCs})")
                    print("*****Setting retained PCs to equal PCA rank for this trial/region model,"
                        " but consider choosing a priori lower retained PCs (using use_PCA) for better consistency across trials/regions")

                # regress
                PCAScores = np.column_stack((PCAScores, np.ones_like(PCAScores[:, 0])))  # add constant
                b = np.linalg.lstsq(PCAScores, y, rcond=None)[0]  # least squares solution

                # remove constant
                b = b[:-1]

                # transform back to predictors
                b = np.dot(PCALoadings.T, b)

                # for mvar_viz: re-arrange to consistent format over regions
                row_start = increments * (target_r - 1)
                row_end = target_r * increments

                b_viz = np.zeros((num_predictors + increments,))
                if target_r == 0:
                    b_viz[row_start:row_end] = b
                else:
                    b_viz[:row_start] = b[:row_start]
                    b_viz[row_end:] = b[row_start:]

                # add to trial_MVAR
                trial_MVAR[:, r1] = b
                trial_MVAR_viz[:, r1] = b_viz
                trial_varEx.append(np.sum(explained[:nPCs]))

                print(f"Time taken to estimate MVAR for region {r1}, trial {t} = {time.time() - start_time}")

            # add to sub_restMVAR
            sub_restMVAR[:, :, t] = trial_MVAR
            sub_restMVAR_viz[:, :, t] = trial_MVAR_viz
            sub_varEx[t, :] = trial_varEx


    output = {}
    output['sub_restFC_avg'] = np.mean(sub_restFC, axis=2)
    output['sub_restMVAR_avg'] = np.mean(sub_restMVAR, axis=2)
    output['sub_restMVAR_viz_avg'] = np.mean(sub_restMVAR_viz, axis=2)
    output['sub_varEx_avg'] = np.mean(sub_varEx, axis=0)
    output['MVAR_PCopt'] = MVAR_PCopt

    # write output
    sub_restFC_avg = output['sub_restFC_avg']
    sub_restMVAR_avg = output['sub_restMVAR_avg']
    sub_restMVAR_viz_avg = output['sub_restMVAR_viz_avg']
    sub_varEx_avg = output['sub_varEx_avg']
    MVAR_PCopt = output['MVAR_PCopt']  # contains nPC array, and optPC (PCnum, MSE, PC variance explained)

    # save each subject separately
    if use_PCA == 3:
        scipy.io.savemat(outfile, {'sub_restFC_avg': sub_restFC_avg,
                                'sub_restMVAR_avg': sub_restMVAR_avg,
                                'sub_restMVAR_viz_avg': sub_restMVAR_viz_avg,
                                'sub_varEx_avg': sub_varEx_avg,
                                'MVAR_PCopt': MVAR_PCopt})
    else:
        scipy.io.savemat(outfile, {'sub_restFC_avg': sub_restFC_avg,
                                'sub_restMVAR_avg': sub_restMVAR_avg,
                                'sub_restMVAR_viz_avg': sub_restMVAR_viz_avg,
                                'sub_varEx_avg': sub_varEx_avg})