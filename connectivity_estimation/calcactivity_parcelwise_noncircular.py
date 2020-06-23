import numpy as np
import nibabel as nib
import h5py
import os
import pkg_resources

dilateMM = 10

partitiondir = pkg_resources.resource_filename('ActflowToolbox.dependencies', 'ColeAnticevicNetPartition/')
defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'
# network definitions dir
networkdefdir = pkg_resources.resource_filename('ActflowToolbox', 'network_definitions/')

def calcactivity_parcelwise_noncircular(data, dlabelfile=defaultdlabelfile, dilated_parcels=True,cortex_only=True, verbose=False):
    """
    This function produces a parcel-to-parcel activity (GLM beta) matrix while excluding vertices in the neighborhood of a given target parcel.
    Excludes all vertices within a 10mm (default) dilated mask of the target parcel when computing parcel-level mean activity.
    Takes in vertex-wise data and generates a parcelA X parcelB activity matrix, with parcelA being the to-be-predicted 'target' and parcelB being the 'source'
    Currently only works for surface-based cortex data
    
    PARAMETERS:
        data            :       vertex-wise data... vertices x conditions; default assumes that data is 96k dense array
        dlabelfile      :       parcellation file; each vertex indicates the number corresponding to each parcel. dlabelfile needs to match same vertex dimensions of data
        dilated_parcels :       If True, will exclude vertices within 10mm of a target parcel's borders when computing mult regression fc (reducing spatial autocorrelation inflation)
        cortex_only       :       If False, will include subcortical volume rois from the CAB-NP
        verbose  :    indicate if additional print commands should be used to update user on progress
    RETURNS:
        activation_matrix       :       Target X Source activity Matrix. Sources-to-target mappings are organized as rows (targets) from each column (source)
    """
    
    if cortex_only: 
        nparcels = 360
    else: 
        nparcels = 718
    # Load dlabel file (cifti)
    if verbose: print('Loading in CIFTI dlabel file')
    dlabels = np.squeeze(nib.load(dlabelfile).get_data())
    # Find and sort unique parcels
    unique_parcels = np.sort(np.unique(dlabels))
    # Only include cortex
    unique_parcels = unique_parcels[:nparcels]
                                            
    # Instantiate empty activation matrix for regular mean time series
    regular_activation_matrix = np.zeros((nparcels,data.shape[1]))
    regular_activation_computed = np.zeros((nparcels,1))
                                                 
    # Instantiate empty activation matrix
    activation_matrix = np.zeros((nparcels,nparcels,data.shape[1]))
    
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

    for parcelInt,parcel in enumerate(unique_parcels):
            
        if verbose: print('Computing activations for target parcel', int(parcel))
            
        # Find where this parcel is in the unique parcel array
        parcel_ind = np.where(unique_parcels==parcel)[0]
        
        #Identify grayordinates to remove (from source parcels) for this target parcel
        coords_to_remove_indices_thisparcel=coords_to_remove_indices[parcel]
        
        #Remove dilated target parcel's grayordinates from source parcels
        source_parcellation = dlabels.copy() # copy the original parcellation dlabel file
        source_parcellation[coords_to_remove_indices_thisparcel] = parcel # modify original dlabel file to remove any vertices that are in the mask (marked with same label as target parcel, which is removed later)

        # get all target ROI indices
        target_ind = np.where(dlabels==parcel)[0] # Find target parcel indices (from dlabel file)

        # Identify all 'source' parcels to include when computing FC
        source_parcels = np.delete(unique_parcels, parcel_ind)

        # Now compute mean activations of each ROI using modified dlabel file after removing target parcel's mask (ie source_parcellation)
        source_parcel_ts = np.zeros((len(source_parcels),data.shape[1])) # source regions X time matrix
        empty_source_row = [] # empty array to keep track of the row index of any sources that might be excluced
        i = 0
        for source in source_parcels:
            source_ind = np.where(source_parcellation==source)[0] # Find source parcel indices (from modified dlabel file)
            sourceInt = np.where(unique_parcels==source)[0]
            
            #Determine if this source parcel was modified (if not, then use standard time series)
            source_ind_orig = np.where(dlabels==source)[0]
            if np.array_equal(source_ind,source_ind_orig):
                
                if regular_activation_computed[sourceInt]:
                    source_parcel_ts[i,:] = regular_activation_matrix[sourceInt,:]
                else:
                    source_parcel_ts[i,:] = np.nanmean(np.real(data[source_ind,:]),axis=0) # compute averaged time series of source parcel
                    #Save time series for future use
                    regular_activation_matrix[sourceInt,:] = source_parcel_ts[i,:].copy()
                    regular_activation_computed[sourceInt] = True
            
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
            source_parcel_ts = np.delete(source_parcel_ts,empty_source_row,axis=0) # delete all ROIs with all 0s
            source_parcels = np.delete(source_parcels,empty_source_row,axis=0) # Delete the 0-variance ROI from the list of sources

        # compute averaged time series of TARGET
        if regular_activation_computed[parcelInt]:
            target_parcel_ts = regular_activation_matrix[parcelInt,:]
        else:
            target_parcel_ts = np.nanmean(np.real(data[target_ind,:]),axis=0)
            #Save time series for future use
            regular_activation_matrix[parcelInt,:] = target_parcel_ts.copy()
            regular_activation_computed[parcelInt] = True

        # Find matrix indices for all source parcels
        source_cols = np.where(np.in1d(unique_parcels,source_parcels))[0]
        target_row = parcelInt
        
        activation_matrix[target_row,source_cols,:] = source_parcel_ts
        activation_matrix[target_row,target_row,:] = target_parcel_ts

    return activation_matrix