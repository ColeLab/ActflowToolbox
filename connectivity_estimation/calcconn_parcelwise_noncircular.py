import numpy as np
import nibabel as nib
import h5py
import os
import pkg_resources
from .multregconn import *
from .corrcoefconn import *
from .pc_multregconn import *
from .combinedFC import *
import ActflowToolbox as actflow
from .. import tools

dilateMM = 10

partitiondir = pkg_resources.resource_filename('ActflowToolbox.dependencies', 'ColeAnticevicNetPartition/')
defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'
# network definitions dir
networkdefdir = pkg_resources.resource_filename('ActflowToolbox', 'network_definitions/')

def calcconn_parcelwise_noncircular(data, connmethod='multreg', dlabelfile=defaultdlabelfile, dilated_parcels=True, precomputedRegularTS=None, cortex_only=True, verbose=False):
    """
    This function produces a parcel-to-parcel connectivity matrix while excluding vertices in the neighborhood of a given target parcel.
    Excludes all vertices within a 10mm (default) dilated mask of the target parcel when computing parcel-to-parcel connectivity.
    Takes in vertex-wise data and generates a parcel X parcel connectivity matrix based on provided connmethod
    Currently only works for surface-based cortex connectivity
    
    PARAMETERS:
        data        : vertex-wise data... vertices x time; default assumes that data is 96k dense array
        connmethod  : a string indicating what connectivity method to use. Options: 'multreg' (default), 'pearsoncorr', 'pc_multregconn', 'combinedFC'
        dlabelfile  : parcellation file; each vertex indicates the number corresponding to each parcel. dlabelfile needs to match same vertex dimensions of data
        dilated_parcels :       If True, will exclude vertices within 10mm of a target parcel's borders when computing mult regression fc (reducing spatial autocorrelation inflation)
        precomputedRegularTS:  optional input of precomputed 'regular' mean time series with original region set. This might cut down on computation time if provided.
        cortex_only       :       If False, will include subcortical volume rois from the CAB-NP
        verbose  :    indicate if additional print commands should be used to update user on progress
    RETURNS:
        fc_matrix       :       Target X Source FC Matrix. Sources-to-target mappings are organized as rows (targets) from each column (source)
    """
    if cortex_only: 
        nparcels = 360
    else: 
        nparcels = 718

    # Load dlabel file (cifti)
    if verbose: print('Loading in CIFTI dlabel file')
    dlabels = np.squeeze(nib.load(dlabelfile).get_data())
    #if cortex_only:
        #Restrict to cortex grayordinates
        #dlabels = dlabels[0:59412]
    # Find and sort unique parcels
    unique_parcels = np.sort(np.unique(dlabels))
    # Only include cortex (if flagged)
    unique_parcels = unique_parcels[:nparcels]
    
    # Instantiate empty time series matrix for regular mean time series, or load from memory if provided
    if precomputedRegularTS is not None:
        regular_ts_matrix = precomputedRegularTS
        regular_ts_computed = np.ones((nparcels,1),dtype=bool)
    else:
        regular_ts_matrix = np.zeros((nparcels,data.shape[1]))
        regular_ts_computed = np.zeros((nparcels,1))

    # Instantiate empty fc matrix
    fc_matrix = np.zeros((nparcels,nparcels)); 
    net_mask = np.zeros((nparcels,nparcels)); # only used for combined-FC
    
    #Load coords_to_remove_indices file
    if cortex_only: 
        outputfilename = networkdefdir + 'coords_to_remove_indices_cortexonly_data.h5'
    else: 
        outputfilename = networkdefdir + 'coords_to_remove_indices_cortexsubcortex_data.h5'
    h5f = h5py.File(outputfilename,'r')
    #Load in dictionary with grayordinates to remove for each target parcel
    coords_to_remove_indices={} 
    for parcelInt,parcel in enumerate(unique_parcels):
        outname1 = 'coords_to_remove_indices'+'/'+str(parcel)
        coords_to_remove_indices[parcel] = h5f[outname1][:].copy()
    h5f.close()
        
    targetDataAll = np.zeros((nparcels,data.shape[1])); # only used for combined-FC 
    for parcelInt,parcel in enumerate(unique_parcels):
        if verbose: print('Computing FC for target parcel',parcelInt,'-',int(parcel),'/',len(unique_parcels))
        
        #Identify target parcel indices
        target_ind = np.where(np.asarray(dlabels==parcel,dtype=bool))[0]
        if verbose: print('\t size of target:', len(target_ind))
            
        # Find where this parcel is in the unique parcel array
        parcel_ind = np.where(unique_parcels==parcel)[0]
        
        #Identify grayordinates to remove (from source parcels) for this target parcel
        coords_to_remove_indices_thisparcel=coords_to_remove_indices[parcel]
        
        #Remove dilated target parcel's grayordinates from source parcels
        source_parcellation = dlabels.copy() # copy the original parcellation dlabel file
        source_parcellation[coords_to_remove_indices_thisparcel] = parcel # modify original dlabel file to remove any vertices that are in the mask (marked with same label as target parcel, which is removed later)

        # Identify all 'source' parcels to include when computing FC
        source_parcels = np.delete(unique_parcels, parcel_ind)

        # Now compute mean time series of each ROI using modified dlabel file after removing target parcel's mask (ie source_parcellation)
        source_parcel_ts = np.zeros((len(source_parcels),data.shape[1])) # source regions X time matrix
        empty_source_row = [] # empty array to keep track of the row index of any sources that might be excluced
        i = 0            
        for source in source_parcels:
            source_ind = np.where(source_parcellation==source)[0] # Find source parcel indices (from modified dlabel file)
            sourceInt = np.where(unique_parcels==source)[0]

            #Determine if this source parcel was modified (if not, then use standard time series)
            source_ind_orig = np.where(dlabels==source)[0]
            if np.array_equal(source_ind,source_ind_orig):

                if regular_ts_computed[sourceInt]:
                    source_parcel_ts[i,:] = regular_ts_matrix[sourceInt,:]
                else:
                    source_parcel_ts[i,:] = np.nanmean(np.real(data[source_ind,:]),axis=0) # compute averaged time series of source parcel
                    #Save time series for future use
                    regular_ts_matrix[sourceInt,:] = source_parcel_ts[i,:].copy()
                    regular_ts_computed[sourceInt] = True

            else:

                # If the entire parcel is excluded (i.e, the time series is all 0s), then skip computing the mean for this parcel
                if len(source_ind)==0:
                    empty_source_row.append(i) # if this source is empty, remember its row (to delete it from the regressor matrix later)
                    i += 1
                    # Go to next source parcel
                    continue

                source_parcel_ts[i,:] = np.nanmean(np.real(data[source_ind,:]),axis=0) # compute averaged time series of source parcel

            i += 1

        # Delete source regions that have been entirely excluded from the source_parcels due to the dilation
        if len(empty_source_row)>0:
            source_parcel_ts = np.delete(source_parcel_ts,empty_source_row,axis=0) # delete all ROIs with all 0s from regressor matrix
            source_parcels = np.delete(source_parcels,empty_source_row,axis=0) # Delete the 0-variance ROI from the list of sources

        # compute averaged time series of TARGET
        if regular_ts_computed[parcelInt]:
            target_parcel_ts = regular_ts_matrix[parcelInt,:]
        else:
            target_parcel_ts = np.nanmean(np.real(data[target_ind,:]),axis=0)
            #Save time series for future use
            regular_ts_matrix[parcelInt,:] = target_parcel_ts.copy()
            regular_ts_computed[parcelInt] = True

        # Find matrix indices for all source parcels
        source_cols = np.where(np.in1d(unique_parcels,source_parcels))[0]
        target_row = parcelInt

        if connmethod == 'multreg':
            # run multiple regression, and add constant
            fc_matrix[target_row,source_cols] = actflow.connectivity_estimation.multregconn(source_parcel_ts,target_parcel_ts)
        elif connmethod == 'pearsoncorr':
            fc_matrix[target_row,source_cols] = actflow.connectivity_estimation.corrcoefconn(source_parcel_ts,target_parcel_ts)
        elif connmethod == 'pc_multregconn':
            fc_matrix[target_row,source_cols] = actflow.connectivity_estimation.pc_multregconn(source_parcel_ts,target_parcel_ts)
        elif connmethod == 'combinedFC':             
            # Run combined-FC with separate target_parcel_ts: this step generates the combined-FC core result (net_mask); which can be used by itself as an FC matrix.
            # The suggested use, however, is to run an additional multiple regression step on the weights validated by combined-FC to add precision to the act-flow predictive model (see below)
            net_mask[target_row,source_cols] = actflow.connectivity_estimation.combinedFC(source_parcel_ts,target_parcel_ts,parcelInt,source_cols)
            targetDataAll[parcelInt,:] = target_parcel_ts.copy(); # all target data required for 2nd combined-FC step to generate final fc_matrix (see below)
    
    # Multiple regression step for combined-FC (note: do not need to index by target/sources here; it's done above for net_mask): 
    if connmethod == 'combinedFC':   
        fc_matrix = actflow.connectivity_estimation.multregconn(targetDataAll,conn_mask=(net_mask!=0))
            
    return fc_matrix 