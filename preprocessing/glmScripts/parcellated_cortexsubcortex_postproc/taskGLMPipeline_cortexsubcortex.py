# Takuya Ito
# 03/28/2018

# Functions to run a GLM analysis

import numpy as np
import os
import glob
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import multiprocessing as mp
import h5py
import scipy.stats as stats
from scipy import signal
import regression

## Define GLOBAL variables (variables accessible to all functions
# Define base data directory
basedir = '/projects/f_mc1689_1/'
datadir = basedir + 'HCP352Data/data/hcppreprocessedmsmall/'
# Define number of frames to skip
framesToSkip = 5
# Define list of subjects
subjNums = ['100206','108020','117930','126325','133928','143224','153934','164636','174437','183034','194443','204521','212823','268749','322224','385450','463040','529953','587664','656253','731140','814548','877269','978578','100408','108222','118124','126426','134021','144832','154229','164939','175338','185139','194645','204622','213017','268850','329844','389357','467351','530635','588565','657659','737960','816653','878877','987074','101006','110007','118225','127933','134324','146331','154532','165638','175742','185341','195445','205119','213421','274542','341834','393247','479762','545345','597869','664757','742549','820745','887373','989987','102311','111009','118831','128632','135528','146432','154936','167036','176441','186141','196144','205725','213522','285345','342129','394956','480141','552241','598568','671855','744553','826454','896879','990366','102513','112516','118932','129028','135629','146533','156031','167440','176845','187850','196346','205826','214423','285446','348545','395756','481042','553344','599671','675661','749058','832651','899885','991267','102614','112920','119126','129129','135932','147636','157336','168745','177645','188145','198350','208226','214726','286347','349244','406432','486759','555651','604537','679568','749361','835657','901442','992774','103111','113316','120212','130013','136227','148133','157437','169545','178748','188549','198451','208327','217429','290136','352738','414229','497865','559457','615744','679770','753150','837560','907656','993675','103414','113619','120414','130114','136833','150726','157942','171330']
#subjNums = ['178950','189450','199453','209228','220721','298455','356948','419239','499566','561444','618952','680452','757764','841349','908860','103818','113922','121618','130619','137229','151829','158035','171633','179346','190031','200008','210112','221319','299154','361234','424939','500222','570243','622236','687163','769064','845458','911849','104416','114217','122317','130720','137532','151930','159744','172029','180230','191235','200614','211316','228434','300618','361941','432332','513130','571144','623844','692964','773257','857263','926862','105014','114419','122822','130821','137633','152427','160123','172938','180432','192035','200917','211417','239944','303119','365343','436239','513736','579665','638049','702133','774663','865363','930449','106521','114823','123521','130922','137936','152831','160729','173334','180533','192136','201111','211619','249947','305830','366042','436845','516742','580650','645450','715041','782561','871762','942658','106824','117021','123925','131823','138332','153025','162026','173536','180735','192439','201414','211821','251833','310621','371843','445543','519950','580751','647858','720337','800941','871964','955465','107018','117122','125222','132017','138837','153227','162329','173637','180937','193239','201818','211922','257542','314225','378857','454140','523032','585862','654350','725751','803240','872562','959574','107422','117324','125424','133827','142828','153631','164030','173940','182739','194140','202719','212015','257845','316633','381543','459453','525541','586460','654754','727553','812746','873968','966975']
# Define all runs you want to preprocess
allRuns = ['rfMRI_REST1_RL', 'rfMRI_REST1_LR','rfMRI_REST2_RL', 'rfMRI_REST2_LR','tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR','tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR','tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']
restRuns = ['rfMRI_REST1_RL', 'rfMRI_REST1_LR','rfMRI_REST2_RL', 'rfMRI_REST2_LR']
taskRuns = ['tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR','tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR','tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']
taskNames = ['tfMRI_EMOTION', 'tfMRI_GAMBLING','tfMRI_LANGUAGE','tfMRI_MOTOR','tfMRI_RELATIONAL','tfMRI_SOCIAL','tfMRI_WM']
taskLength = {'EMOTION':176,'GAMBLING':253,'LANGUAGE':316,'MOTOR':284,'RELATIONAL':232,'SOCIAL':274,'WM':405};
# Define the *output* directory for nuisance regressors
nuis_reg_dir = datadir + 'nuisanceRegressors/'
# Create directory if it doesn't exist
if not os.path.exists(nuis_reg_dir): os.makedirs(nuis_reg_dir)
# Define the *output* directory for preprocessed data
outputdir = datadir + 'parcellated_cortexsubcortex_postproc/'
# TRlength
trLength = .720

taskEV_Identifier = {'EMOTION':[0,1],
                     'GAMBLING':[2,3],
                     'LANGUAGE':[4,5],
                     'MOTOR':[6,7,8,9,10,11],
                     'RELATIONAL':[12,13],
                     'SOCIAL':[14,15],
                     'WM':[16,17,18,19,20,21,22,23]}

def runGroupTaskGLM(nuisModel='24pXaCompCorXVolterra', taskModel='FIR', byRun=False, zscore=False):
    scount = 1
    for subj in subjNums:
        print('Running task GLM for subject', subj, '|', scount, '/', len(subjNums), '... model:', taskModel, '... zscore:', zscore)
        if byRun:
            for task in taskNames:
                taskGLM_byRun(subj, task, nuisModel=nuisModel, zscore=zscore)
        else:
            for task in taskNames:
                taskGLM(subj, task, taskModel=taskModel, nuisModel=nuisModel, zscore=zscore)
        
        scount += 1

def runGroupRestGLM(nuisModel='24pXaCompCorXVolterra', taskModel='FIR'):
    scount = 1
    for subj in subjNums:
        print('Running task regression matrix on resting-state data for subject', subj, '|', scount, '/', len(subjNums), '... model:', taskModel)
        taskGLM_onRest(subj, taskModel=taskModel, nuisModel=nuisModel)
        
        scount += 1

def taskGLM_byRun(subj, task, nuisModel='24pXaCompCorXVolterra', zscore=False):
    """
    This function runs a task-based GLM on a single subject
    Will only regress out task-timing, using either a canonical HRF model or FIR model

    Input parameters:
        subj        :   subject number as a string
        task        :   fMRI task name (2 runs per task for HCP data)
        nuisModel   :   nuisance regression model (to identify input data)
        zscore      :   z-scores the data prior to fitting the GLM
    """

    #h5f = h5py.File(outputdir + subj + '_glmOutput_data.h5','a')
    h5f = h5py.File(outputdir + subj + '_glmOutput_cortexsubcortex_data.h5','r')

    run1 = h5f[task+'_RL']['nuisanceReg_resid_'+nuisModel][:].copy()
    run2 = h5f[task+'_LR']['nuisanceReg_resid_'+nuisModel][:].copy()
    h5f.close()

    # Identify number of ROIs
    nROIs = run1.shape[0]
    # Identify number of TRs
    run1_nTRs = run1.shape[1]
    run2_nTRs = run2.shape[1]

    # Load regressors for data
    X = loadTaskTiming(subj, task, taskModel='canonical', nRegsFIR=25)

    taskRegs = X['taskRegressors'] # These include the two binary regressors

    taskRegs_run1 = taskRegs[:run1_nTRs,:]
    taskRegs_run2 = taskRegs[run1_nTRs:(run1_nTRs+run2_nTRs),:]

    if zscore:
        zscore_str = '_zscore'
        run1 = stats.zscore(run1,axis=1)
    else:
        zscore_str = ''

    ## Run regression on first run (RL)
    betas, resid = regression.regression(run1.T, taskRegs_run1, constant=True)
    
    betas = betas.T # Exclude nuisance regressors
    residual_ts = resid.T

    h5f = h5py.File(outputdir + subj + '_glmOutput_cortexsubcortex_data.h5','a')
    outname1 = 'taskRegression/' + task + '_RL_' + nuisModel + '_taskReg_resid_canonical' + zscore_str
    outname2 = 'taskRegression/' + task + '_RL_' + nuisModel + '_taskReg_betas_canonical' + zscore_str
    try:
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    except:
        del h5f[outname1], h5f[outname2]
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    h5f.close()
    
    if zscore:
        zscore_str = '_zscore'
        run2= stats.zscore(run2,axis=1)
    else:
        zscore_str = ''
    
    ## Run regression on first run (LR)
    betas, resid = regression.regression(run2.T, taskRegs_run2, constant=True)
    
    betas = betas.T # Exclude nuisance regressors
    residual_ts = resid.T

    h5f = h5py.File(outputdir + subj + '_glmOutput_cortexsubcortex_data.h5','a')
    outname1 = 'taskRegression/' + task + '_LR_' + nuisModel + '_taskReg_resid_canonical' + zscore_str
    outname2 = 'taskRegression/' + task + '_LR_' + nuisModel + '_taskReg_betas_canonical' + zscore_str
    try:
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    except:
        del h5f[outname1], h5f[outname2]
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    h5f.close()

def taskGLM(subj, task, taskModel='canonical', nuisModel='24pXaCompCorXVolterra', zscore=False):
    """
    This function runs a task-based GLM on a single subject
    Will only regress out task-timing, using either a canonical HRF model or FIR model

    Input parameters:
        subj        :   subject number as a string
        task        :   fMRI task name (2 runs per task for HCP data)
        nuisModel   :   nuisance regression model (to identify input data)
        zscore      :   z-scores the data prior to fitting the GLM
    """

    print('Loading task data for subject', subj, 'and task', task, '... zscore:', zscore)
    h5f = h5py.File(outputdir + subj + '_glmOutput_cortexsubcortex_data.h5','r')

    run1 = h5f[task+'_RL']['nuisanceReg_resid_'+nuisModel][:].copy()
    run2 = h5f[task+'_LR']['nuisanceReg_resid_'+nuisModel][:].copy()
    data = np.hstack((run1,run2))
    h5f.close()

    # Identify number of ROIs
    nROIs = data.shape[0]
    # Identify number of TRs
    nTRs = data.shape[1]

    # Load regressors for data
    X = loadTaskTiming(subj, task, taskModel='canonical', nRegsFIR=25)

    taskRegs = X['taskRegressors'] # These include the two binary regressors

    print('\tRunning regression on input matrix', data.T.shape, 'with regressor matrix', taskRegs.shape, '...')
    if zscore:
        zscore_str = '_zscore'
        data = stats.zscore(data,axis=1)
    else:
        zscore_str = ''

    betas, resid = regression.regression(data.T, taskRegs, constant=True)
    
    residual_ts = resid.T
    betas = betas.T

    
    print('\tSaving out data...')
    h5f = h5py.File(outputdir + subj + '_glmOutput_cortexsubcortex_data.h5','a')
    outname1 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_resid_' + taskModel + zscore_str
    outname2 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_betas_' + taskModel + zscore_str
    try:
        if taskModel=='FIR':
            h5f.create_dataset(outname1,data=residual_ts)
        else:
            h5f.create_dataset(outname2,data=betas)
    except:
        if taskModel=='FIR':
            del h5f[outname1]
            h5f.create_dataset(outname1,data=residual_ts)
        else:
            del h5f[outname2]
            h5f.create_dataset(outname2,data=betas)
    h5f.close()

def taskGLM_onRest(subj, taskModel='canonical', nuisModel='24pXaCompCorXVolterra'):
    """
    This function runs a task-based GLM on a single subject
    Will only regress out task-timing, using either a canonical HRF model or FIR model

    Input parameters:
        subj        :   subject number as a string
        task        :   fMRI task name (2 runs per task for HCP data)
        nuisModel   :   nuisance regression model (to identify input data)
    """

    h5f = h5py.File(outputdir + subj + '_glmOutput_data.h5','a')

    restname = 'rfMRI_REST'
    run1 = h5f[restname+'1_RL']['nuisanceReg_resid_'+nuisModel][:].copy()
    run2 = h5f[restname+'1_LR']['nuisanceReg_resid_'+nuisModel][:].copy()
    run3 = h5f[restname+'2_RL']['nuisanceReg_resid_'+nuisModel][:].copy()
    run4 = h5f[restname+'2_LR']['nuisanceReg_resid_'+nuisModel][:].copy()
    data = np.hstack((run1,run2,run3,run4))
    h5f.close()

    # Identify number of ROIs
    nROIs = data.shape[0]
    # Identify number of TRs
    nTRs = data.shape[1]

    # Load regressors for data, for each task, and 
    trcount = 0
    data_resids = []
    for task in taskNames:
        X = loadTaskTiming(subj, task, taskModel=taskModel, nRegsFIR=25)
        taskRegs = X['taskRegressors']

        trstart = trcount
        trend = trstart + taskRegs.shape[0]

        betas, resid = regression.regression(data[:,trstart:trend].T, taskRegs, constant=True)
        
        betas = betas.T # Exclude nuisance regressors
        residual_ts = resid.T

        data_resids.extend(residual_ts.T)

        trcount = trend

    # Append the rest of the resting-state time series, just in case
#    trstart = trcount
#    data_resids.extend(data[:,trstart:].T)
    
    data_resids = np.asarray(data_resids)
    residual_ts = data_resids.T

    h5f = h5py.File(outputdir + subj + '_glmOutput_cortexsubcortex_data.h5','a')
    outname1 = 'taskRegression/' + restname + '_' + nuisModel + '_taskReg_resid_' + taskModel
    try:
        h5f.create_dataset(outname1,data=residual_ts)
    except:
        del h5f[outname1]
        h5f.create_dataset(outname1,data=residual_ts)
    h5f.close()

    return residual_ts

def loadTaskTiming(subj, task, taskModel='canonical', nRegsFIR=25):
    nRunsPerTask = 2

    taskkey = task[6:] # Define string identifier for tasks
    taskEVs = taskEV_Identifier[taskkey]
    stimMat = np.zeros((taskLength[taskkey]*nRunsPerTask,len(taskEV_Identifier[taskkey])))
    stimdir = basedir + 'HCP352Data/data/timingfiles3/'
    stimfiles = glob.glob(stimdir + subj + '*EV*' + taskkey + '*1D')
    
    for stimcount in range(len(taskEVs)):
        ev = taskEVs[stimcount] + 1
        stimfile = glob.glob(stimdir + subj + '*EV' + str(ev) + '_' + taskkey + '*1D')
        stimMat[:,stimcount] = np.loadtxt(stimfile[0])

    nTRsPerRun = int(stimMat.shape[0]/2.0)

    ## 
    if taskModel=='FIR':
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)

        ## First set up FIR design matrix
        stim_index = []
        taskStims_FIR = [] 
        for stim in range(stimMat.shape[1]):
            taskStims_FIR.append([])
            time_ind = np.where(stimMat[:,stim]==1)[0]
            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
            # Identify the longest block - set FIR duration to longest block
            maxRegsForBlocks = 0
            for block in blocks:
                if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
            taskStims_FIR[stim] = np.zeros((stimMat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag
            stim_index.extend(np.repeat(stim,maxRegsForBlocks+nRegsFIR))
        stim_index = np.asarray(stim_index)

        ## Now fill in FIR design matrix
        # Make sure to cut-off FIR models for each run separately
        trcount = 0

        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun
                
            for stim in range(stimMat.shape[1]):
                time_ind = np.where(stimMat[:,stim]==1)[0]
                blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
                for block in blocks:
                    reg = 0
                    for tr in block:
                        # Set impulses for this run/task only
                        if trstart < tr < trend:
                            taskStims_FIR[stim][tr,reg] = 1
                            reg += 1

                        if not trstart < tr < trend: continue # If TR not in this run, skip this block

                    # If TR is not in this run, skip this block
                    if not trstart < tr < trend: continue

                    # Set lag due to HRF
                    for lag in range(1,nRegsFIR+1):
                        # Set impulses for this run/task only
                        if trstart < tr+lag < trend:
                            taskStims_FIR[stim][tr+lag,reg] = 1
                            reg += 1
            trcount += nTRsPerRun
        

        taskStims_FIR2 = np.zeros((stimMat.shape[0],1))
        task_index = []
        for stim in range(stimMat.shape[1]):
            task_index.extend(np.repeat(stim,taskStims_FIR[stim].shape[1]))
            taskStims_FIR2 = np.hstack((taskStims_FIR2,taskStims_FIR[stim]))

        taskStims_FIR2 = np.delete(taskStims_FIR2,0,axis=1)

        #taskRegressors = np.asarray(taskStims_FIR)
        taskRegressors = taskStims_FIR2
    
        # To prevent SVD does not converge error, make sure there are no columns with 0s
        zero_cols = np.where(np.sum(taskRegressors,axis=0)==0)[0]
        taskRegressors = np.delete(taskRegressors, zero_cols, axis=1)
        stim_index = np.delete(stim_index, zero_cols)

    elif taskModel=='canonical':
        ## 
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
        taskStims_HRF = np.zeros(stimMat.shape)
        spm_hrfTS = spm_hrf(trLength,oversampling=1)
       
        trcount = 0
        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun

            for stim in range(stimMat.shape[1]):

                # Perform convolution
                tmpconvolve = np.convolve(stimMat[trstart:trend,stim],spm_hrfTS)
                tmpconvolve_run = tmpconvolve[:nTRsPerRun] # Make sure to cut off at the end of the run
                taskStims_HRF[trstart:trend,stim] = tmpconvolve_run

            trcount += nTRsPerRun

        taskRegressors = taskStims_HRF.copy()
    
        stim_index = []
        for stim in range(stimMat.shape[1]):
            stim_index.append(stim)
        stim_index = np.asarray(stim_index)


    # Create temporal mask (skipping which frames?)
    tMask = []
    tmp = np.ones((nTRsPerRun,), dtype=bool)
    tmp[:framesToSkip] = False
    tMask.extend(tmp)
    tMask.extend(tmp)
    tMask = np.asarray(tMask,dtype=bool)

    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors[tMask,:]
    output['taskDesignMat'] = stimMat[tMask,:]
    output['stimIndex'] = stim_index

    return output

def loadTaskTimingForAllTasks(subj, taskModel='canonical', nRegsFIR=25):
    # Calculate total number of regressors for FIR model
    total_stims = 0
    total_trs = 0
    total_cond = 0
    for task in taskNames:
        total_cond += loadTaskTiming(subj,task,taskModel=taskModel,nRegsFIR=nRegsFIR)['taskDesignMat'].shape[1]
        total_stims += loadTaskTiming(subj,task,taskModel=taskModel,nRegsFIR=nRegsFIR)['taskRegressors'].shape[1]
        total_trs += loadTaskTiming(subj,task,taskModel=taskModel,nRegsFIR=nRegsFIR)['taskRegressors'].shape[0]
    
    stimMat = np.zeros((total_trs,total_cond))
    taskRegressors = np.zeros((total_trs, total_stims))

    trcount = 0
    stimcount = 0
    firregcount = 0
    for task in taskNames:
        
        X = loadTaskTiming(subj,task,taskModel=taskModel,nRegsFIR=nRegsFIR)

        n_stims = X['taskDesignMat'].shape[1]
        n_trs = X['taskDesignMat'].shape[0]
        n_firregs = X['taskRegressors'].shape[1]

        trstart = trcount
        trend = trstart + n_trs
        
        stimstart = stimcount
        stimend = stimstart + n_stims

        firstart = firregcount
        firend = firstart + n_firregs


        stimMat[trstart:trend,stimstart:stimend] = X['taskDesignMat']
        if taskModel=='canonical':
            taskRegressors[trstart:trend,stimstart:stimend] = X['taskRegressors']
        elif taskModel=='FIR':
            taskRegressors[trstart:trend,firstart:firend] = X['taskRegressors']

        stimcount = stimend
        trcount = trend
        firregcount = firend

    output = {}
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stimMat

    return output


def _group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

