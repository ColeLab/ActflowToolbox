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
        fc_matrix       :       Region X Region FC Matrix. Sources-to-target mappings are organized as rows (sources) to each column (target)
    """

    nparcels = 360
    parcel_arr = np.arange(nparcels)
    # Load dlabel file (cifti)
    dlabels = np.squeeze(nib.load(dlabelfile).get_data())
    # Find and sort unique parcels
    unique_parcels = np.sort(np.unique(dlabels))
    # Only include cortex
    unique_parcels = unique_parcels[:nparcels]

    # Instantiate empty fc matrix
    fc_matrix = np.zeros((nparcels,nparcels))

    for parcel in unique_parcels:
        # Find where this parcel is in the unique parcel array
        parcel_ind = np.where(unique_parcels==parcel)[0]
        # Load in mask for target parcel
        if dilated_parcels:
            parcel_mask = np.squeeze(nib.load('surfaceMasks/GlasserParcel' + str(parcel) + '_dilated_10mm.dscalar.nii').get_data())
        else:
            parcel_mask = np.squeeze(nib.load('surfaceMasks/GlasserParcel' + str(parcel) + '.dscalar.nii').get_data())

        # get all target ROI indices
        target_ind = np.squeeze(nib.load('surfaceMasks/GlasserParcel' + str(parcel) + '.dscalar.nii').get_data())

        # remove target parcel's mask from set of possible source vertices
        mask_ind = np.where(parcel_mask=1.0)[0] # find mask indices
        source_indices = dlabels.copy() # copy the original parcellation dlabel file
        source_indices[mask_ind] = 0 # modify original dlabel file to remove any vertices that are in the mask

        # Identify all 'source' parcels to include when computing FC
        source_parcels = np.delete(unique_parcels, parcel_ind)

        # Now compute mean time series of each ROI using modified dlabel file after removing target parcel's mask (ie source_indices)
        source_parcel_ts = np.zeros((len(source_parcels),data.shape[1])) # source regions X time matrix
        i = 0
        for source in source_parcels:
            source_ind = np.where(source_indices==source)[0] # Find source parcel indices (from modified dlabel file)
            source_parcel_ts[i,:] = np.mean(data[source_ind,:],axis=0) # compute averaged time series of source parcel
            i += 1

        # compute averaged time series of TARGET
        target_parcel_ts = np.mean(data[target_ind,:],axis=0)
        # run multiple regression, and add constant
        beta_fc, resid = regression.regression(target_parcel_ts,source_parcel_ts.T,alpha=0, constant=True) # increase alpha if want to apply a ridge penalty

        # Find matrix indices for all source parcels;
        source_rows = source_parcels - 1 # subtract by 1 since source_parcels are organized from 1-360, and need to transform to python indices
        target_col = parcel - 1 # subtract by 1 to fit to python indices
        fc_matrix[source_rows,target_col] = beta_fc[1:] # exclude 1st coef; first coef is beta_0 (or mean)

    return fc_matrix


        





     





