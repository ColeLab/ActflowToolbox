
# This function computes FC for parcel-to-parcel FC, while excluding vertices in the neighborhood of a given target parcel
# Excludes all vertices within a 10mm dilated mask of the target parcel when computing parcel-to-parcel cortical FC.

import numpy as np
import nibabel as nib
import os
import pkg_resources

dilateMM = 10

defaulttoolboxpath='/projects/f_mc1689_1/AnalysisTools/ActflowToolbox/'
partitiondir = defaulttoolboxpath+'dependencies/ColeAnticevicNetPartition/'
defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'
dilatedmaskdir = defaulttoolboxpath + 'network_definitions/Glasser2016/surfaceMasks/'

def calcconn_parcelwise_noncircular_surface(data, connmethod='multreg', toolboxpath=defaulttoolboxpath, dlabelfile=defaultdlabelfile, dilated_parcels=True,verbose=False):
    """
    This function computes multiple regression FC for a parcellation scheme
    Takes in vertex-wise data and generates a parcel X parcel FC matrix based on multiple linear regression
    Currently only works for cortex FC
    PARAMETERS:
        data            :       vertex-wise data... vertices x time; default assumes that data is 96k dense array
		connmethod		:		a string indicating what connectivity method to use. Options: 'multreg' (default), 'pcreg'
        dlabelfile      :       parcellation file; each vertex indicates the number corresponding to each parcel. dlabelfile needs to match same vertex dimensions of data
        dilated_parcels :       If True, will exclude vertices within 10mm of a target parcel's borders when computing mult regression fc (reducing spatial autocorrelation inflation)
    RETURNS:
        fc_matrix       :       Region X Region FC Matrix. Sources-to-target mappings are organized as rows (sources) to each column (target)
    """

    nparcels = 360
    parcel_arr = np.arange(nparcels)
    # Load dlabel file (cifti)
    if verbose: print('Loading in CIFTI dlabel file')
    dlabels = np.squeeze(nib.load(dlabelfile).get_data())
    # Find and sort unique parcels
    unique_parcels = np.sort(np.unique(dlabels))
    # Only include cortex
    unique_parcels = unique_parcels[:nparcels]

    # Instantiate empty fc matrix
    fc_matrix = np.zeros((nparcels,nparcels))

    for parcel in unique_parcels:
        if verbose: print('Computing FC for target parcel', int(parcel))

        # Find where this parcel is in the unique parcel array
        parcel_ind = np.where(unique_parcels==parcel)[0]
        # Load in mask for target parcel
        if dilated_parcels:
            parcel_mask = np.squeeze(nib.load(dilatedmaskdir+'GlasserParcel' + str(int(parcel)) + '_dilated_10mm.dscalar.nii').get_data())
        else:
            parcel_mask = np.squeeze(nib.load(dilatedmaskdir+'GlasserParcel' + str(int(parcel)) + '.dscalar.nii').get_data())

        # get all target ROI indices
        target_ind = np.squeeze(nib.load(dilatedmaskdir+'GlasserParcel' + str(int(parcel)) + '.dscalar.nii').get_data())
        target_ind = np.asarray(target_ind,dtype=bool)

        # remove target parcel's mask from set of possible source vertices
        mask_ind = np.where(parcel_mask==1.0)[0] # find mask indices
        source_indices = dlabels.copy() # copy the original parcellation dlabel file
        source_indices[mask_ind] = 0 # modify original dlabel file to remove any vertices that are in the mask

        # Identify all 'source' parcels to include when computing FC
        source_parcels = np.delete(unique_parcels, parcel_ind)

        # Now compute mean time series of each ROI using modified dlabel file after removing target parcel's mask (ie source_indices)
        source_parcel_ts = np.zeros((len(source_parcels),data.shape[1])) # source regions X time matrix
        i = 0
        for source in source_parcels:
            source_ind = np.where(source_indices==source)[0] # Find source parcel indices (from modified dlabel file)
            # If the entire parcel is excluded (i.e, the time series is all 0s, then skip computing the mean for this parcel)
            if len(source_ind)==0:
                i += 1
                # Go to next source parcel
                continue

            source_parcel_ts[i,:] = np.nanmean(data[source_ind,:],axis=0) # compute averaged time series of source parcel
            i += 1

        # compute averaged time series of TARGET
        target_parcel_ts = np.mean(data[target_ind,:],axis=0)

		if connmethod == 'multreg'
        	# run multiple regression, and add constant
			fc_matrix[source_rows,target_col] = multregressionconnectivity(source_parcel_ts,target_parcel_ts)

    return fc_matrix