import numpy as np
import dynamic_sensory_motor_egi
from scipy.stats import zscore

def dynamic_actflow(activity, restMVAR, target_inds, MVAR_increments, cond_inds, include_contemp, include_autoreg, exclude_target_net, zscore_sub, regress_targetAct, titrate_lags, pairwise_lags, actflow_source_contemp, actflow_target_autoregs):
<<<<<<< HEAD
=======
    numTasks = len(np.unique(cond_inds))
    numTrials, numRegions_all, _ = activity.shape
    numRegions_targets = len(target_inds)

>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)


    activity = np.asarray(activity)
    target_inds = np.asarray(target_inds)
    cond_inds = np.asarray(cond_inds)
    MVAR_increments = int(MVAR_increments)
    include_contemp = int(include_contemp)
    include_autoreg = int(include_autoreg)
    exclude_target_net = int(exclude_target_net)
    zscore_sub = int(zscore_sub)
    regress_targetAct = int(regress_targetAct)
    titrate_lags = int(titrate_lags)
    pairwise_lags = int(pairwise_lags)
    actflow_source_contemp = int(actflow_source_contemp)
    actflow_target_autoregs = int(actflow_target_autoregs)
<<<<<<< HEAD

    numTasks = len(np.unique(cond_inds))
    numTrials, numRegions_all, _ = activity.shape
    numRegions_targets = len(target_inds)
=======
    print(f"restMVAR: {restMVAR}")
    print(f"numTasks: {numTasks}")
    print(f"activity: {activity}")
    print(f"activity shape: {activity.shape}")
    print(f"numTrials: {numTrials}")
    print(f"numRegions_all: {numRegions_all}")
    print(f"numRegions_targets: {numRegions_targets}")
    print(f"target_inds: {target_inds}")
    print(f"cond_inds: {cond_inds}")
    print(f"MVAR_increments: {MVAR_increments} include_contemp {include_contemp}")
    print(f"include_autoreg: {include_autoreg} exclude_target_net: {exclude_target_net}")
    print(f"zscore_sub: {zscore_sub} regress_targetAct: {regress_targetAct}")
    print(f"titrate_lags: {titrate_lags} pairwise_lags: {pairwise_lags}")
    print(f"actflow_source_contemp: {actflow_source_contemp} actflow_target_autoregs: {actflow_target_autoregs}")

>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
    
    if include_contemp == 1 and actflow_source_contemp == 0:
        model_order = MVAR_increments - 1
    else:
        model_order = MVAR_increments

    first_tpoint = model_order
    numTpoints = activity.shape[2] - first_tpoint

    if include_contemp == 1 and actflow_source_contemp == 0:
        contemp_inds = np.arange(0, int(numRegions_all * MVAR_increments), int(MVAR_increments))
        restMVAR = np.delete(restMVAR, contemp_inds, axis=0)
<<<<<<< HEAD

=======
    print("restMVAR : ", restMVAR)
>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
    if include_autoreg == 1 and actflow_target_autoregs == 0:
        if include_contemp == 1:
            restMVAR = restMVAR[:-MVAR_increments + 1, :]
        else:
            restMVAR = restMVAR[:-MVAR_increments, :]

    MVAR_regions = np.repeat(np.arange(numRegions_all), model_order)

    pred = np.zeros((numTrials, numRegions_targets, numTpoints))
    actual = np.zeros((numTrials, numRegions_targets, numTpoints))
    regionNumList = np.arange(numRegions_all)
    if 'pairwise_lags' not in locals():
        pairwise_lags = 0

    if titrate_lags == 1 and pairwise_lags == 1:
        raise ValueError('titrate_lags and pairwise_lags cannot both be set to 1!')


    for trialNum in range(numTrials):
        

        for regionNum in range(numRegions_targets):
            target = int(target_inds[regionNum])
            sources = regionNumList.copy()
            if exclude_target_net == 1:
                sources = np.setdiff1d(sources, target_inds)
            elif exclude_target_net == 0:
                sources = np.delete(sources, np.where(sources == target))
            target_act = activity[trialNum, target, :]
            source_act = activity[trialNum, sources, :]
            if regress_targetAct == 1:
                reg_target = np.vstack((target_act[:], np.ones(target_act.shape[0]))).T
                for ss in range(source_act.shape[1]):
                    # Perform linear regression
                    # np.linalg.lstsq returns several values, where the third value is the residuals
<<<<<<< HEAD
=======
                    # print(f"target_FC0: {target_FC}")
>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
                    _, _, residuals, _ = np.linalg.lstsq(reg_target, source_act[ss, :], rcond=None)
                    # Replace part of source_act with residuals
                    # Note: lstsq does not return residuals in the same shape, so reshape might be needed
                    source_act[ss, :] = residuals.reshape(-1)


            
            target_FC = restMVAR[:, regionNum]
            if exclude_target_net == 1:
                _, ind, _ = np.intersect1d(MVAR_regions, target_inds, return_indices=True)
            elif exclude_target_net == 0:
                _, ind, _ = np.intersect1d(MVAR_regions, target, return_indices=True)


            rem_inds = []
            for e in ind:
                rem_inds.extend(range(e, e + model_order))

            target_FC = np.delete(target_FC, rem_inds, axis=0)




        t0 = first_tpoint
        actual_region = []
        pred_region = []
        for tpoint in range(numTpoints):
            target_FC_tpoint = target_FC.copy()
            actual_region.append(target_act[t0-1])

            activity_lags = []
            for m in range(model_order):
                m_act = source_act[:, t0-m-1]
                activity_lags.append(m_act)

            activity_lags = np.concatenate(activity_lags).ravel()
            if include_autoreg == 1 and actflow_target_autoregs == 1:
                target_lags = []
                for m in range(model_order):
                    m_act = target_act[t0-m-1]
                    target_lags.append(m_act)

<<<<<<< HEAD
=======
                # target_lags = np.concatenate(target_lags).ravel()
>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
                activity_lags = np.concatenate([activity_lags, target_lags])
                            
                
            if titrate_lags != 0:
                m_orders = np.arange(1, model_order+1)
                lags_to_exclude = np.where(m_orders < titrate_lags)
                num_sources = sources.shape[1]
                lag_vec = []
                for mm in lags_to_exclude:
                    mm_lag = mm
                    mm_vec = np.arange(mm_lag, model_order*num_sources, model_order)
                    lag_vec.extend(mm_vec)
                

                target_FC_tpoint = np.delete(target_FC_tpoint, lag_vec)
                activity_lags = np.delete(activity_lags, lag_vec)

            elif pairwise_lags != 0:
                num_sources = sources.shape[1]
                lag_vec = np.arange(pairwise_lags, model_order*num_sources, model_order)

                target_FC_tpoint = target_FC_tpoint[lag_vec]
                activity_lags = activity_lags[lag_vec]

            pred_region.append(np.dot(target_FC_tpoint, activity_lags))
            t0 += 1
        

        if zscore_sub == 1:
            actual[trialNum, regionNum, :] = zscore(actual_region)

            pred[trialNum, regionNum, :] = zscore(pred_region)

        else:
            actual[trialNum, regionNum, :] = actual_region
            pred[trialNum, regionNum, :] = pred_region
        
<<<<<<< HEAD

=======
    # print("actual ", actual)
    # print("pred ", pred)
    # print("actual shape ", actual.shape)
    # print("pred shape ", pred.shape)
>>>>>>> 794e6bc (added compute_pcaFC_opt and dynamic_actflow support)
    return [pred, actual]