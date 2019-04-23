# Taku Ito
# 4/11/2019
# Demo to implement multiple regression FC on glasser parcels after removing neighboring vertices (10mm dilation) from fc estimates

import numpy as np
import h5py
from . import multregfc_glasser_parcellation as fccalc
import importlib
#fccalc = importlib.reload(fccalc)

data = h5py.File('100206_glmOutput_64k_data.h5','r')

timeseries_64k = data['rfMRI_REST1_LR']['nuisanceReg_resid_24pXaCompCorXVolterra'][:].copy()

dlabelfile_forparcels='/projects/f_mc1689_1/AnalysisTools/ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'

fcmatrix = fccalc.compute_parcellation_fc(timeseries_64k.real,dilated_parcels=True,verbose=True,dlabelfile=dlabelfile_forparcels)
