# Taku Ito
# 4/8/2019

# This script dilates individual parcels by x mm (default 10mm)
# Purpose is to use this as a mask to exclude any vertices within 10mm of a parcel when estimaing FC via either ridge or multiple linear regression

import numpy as np
import nibabel as nib
from tools import regression
import os

nParcels = 360
dilateMM = 10

labeldir = '/projects/AnalysisTools/ParcelsGlasser2016/'
dlabelfile = labeldir + 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'

surfacedir = labeldir + 'Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/'
leftSurface = surfacedir + 'Q1-Q6_RelatedParcellation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'
rightSurface = surfacedir + 'Q1-Q6_RelatedParcellation210.R.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'

def compute_parcellation_fc(data, dlabelfile, dilated_parcels=True):
    """
    This function computes multiple regression FC for a parcellation scheme
    Takes in vertex-wise data and generates a parcel X parcel FC matrix based on multiple linear regression
    PARAMETERS:
        data            :       vertex-wise data... vertices x time
        dlabelfile      :       parcellation file; each vertex indicates the number corresponding to each parcel
        dilated_parcels :       If True, will exclude vertices within 10mm of a target parcel's borders when computing mult regression fc (reducing spatial autocorrelation inflation)
    RETURNS:
        fc_matrix
    """

    # Load dlabel file (cifti)
    dlabels = np.squeeze(nib.load(dlabelfile).get_data())
    # Identify number of unique parcels (excluding 0)
    unique_parcels = np.unique(dlabels)

     





