# Taku Ito
# 4/8/2019

# This script dilates individual parcels by x mm (default 10mm)
# Purpose is to use this as a mask to exclude any vertices within 10mm of a parcel when estimaing FC via either ridge or multiple linear regression

import numpy as np
import nibabel as nib
from tools import regression
import os

dilateMM = 10

partitiondir = 'ColeAnticevicNetPartition/'
defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'

surfacedir = labeldir + 'Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/'
leftSurface = surfacedir + 'Q1-Q6_RelatedParcellation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'
rightSurface = surfacedir + 'Q1-Q6_RelatedParcellation210.R.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'

def compute_parcellation_fc(data, dlabelfile=defaultdlabelfile, dilated_parcels=True):
    """
    This function computes multiple regression FC for a parcellation scheme
    Takes in vertex-wise data and generates a parcel X parcel FC matrix based on multiple linear regression
    Currently only works for cortex FC 
    PARAMETERS:
        data            :       vertex-wise data... vertices x time; default assumes that data is 96k dense array
        dlabelfile      :       parcellation file; each vertex indicates the number corresponding to each parcel. dlabelfile needs to match same vertex dimensions of data
        dilated_parcels :       If True, will exclude vertices within 10mm of a target parcel's borders when computing mult regression fc (reducing spatial autocorrelation inflation)
    RETURNS:
        fc_matrix
    """

    nparcels = 360
    parcel_arr = np.arange(nparcels)
    # Load dlabel file (cifti)
    dlabels = np.squeeze(nib.load(dlabelfile).get_data())
    # Find and sort unique parcels
    unique_parcels = np.sort(np.unique(dlabels))
    # Only include cortex
    unique_parcels = unique_parcels[:nparcels]

    for parcel in unique_parcels:
        if dilated_parcels:
            parcel_mask = np.squeeze(nib.load('surfaceMasks/GlasserParcel' + str(parcel) + '_dilated_10mm.dscalar.nii').get_data())
        else:
            parcel_mask = np.squeeze(nib.load('surfaceMasks/GlasserParcel' + str(parcel) + '.dscalar.nii').get_data())

        # get all target ROI indices
        target_ind = np.squeeze(nib.load('surfaceMasks/GlasserParcel' + str(parcel) + '.dscalar.nii').get_data())





     





