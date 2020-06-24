

import numpy as np
import nibabel as nib
import h5py
import os
import pkg_resources

dilateMM = 10

partitiondir = pkg_resources.resource_filename('ActflowToolbox.dependencies', 'ColeAnticevicNetPartition/')
defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'
# maskdir cortex
dilatedmaskdir_cortex = pkg_resources.resource_filename('ActflowToolbox.network_definitions', 'CAB-NP/surfaceMasks/')
# maskdir subcortex
dilatedmaskdir_subcortex = pkg_resources.resource_filename('ActflowToolbox.network_definitions', 'CAB-NP/volumeMasks/')
# network definitions dir
networkdefdir = pkg_resources.resource_filename('ActflowToolbox', 'network_definitions/')

def calc_removal_coords(dlabelfile=defaultdlabelfile, cortex_only=True, verbose=False, parcel_level=False):
    
    #Function to calculate which vertices to remove for each target parcel for the spatially non-circular code
    #It saves to the 'network_definitions/' directory as coords_to_remove_indices_data.h5. 
    #This can be an input to "non-circular" code, which removes grayordinates/coordinates within dilateMM distance
    #from the to-be-predicted target parcel (to help account for spatial spread of BOLD signal).
    #Note: This function assumes the parcels are loaded in ascending order (1-360 or 1-720)

    
    if cortex_only: 
        nparcels = 360
    else: 
        nparcels = 718

    # Load dlabel file (cifti)
    if verbose: print('Loading in CIFTI dlabel file')
    dlabels = np.squeeze(nib.load(dlabelfile).get_data())
    if cortex_only:
        #Restrict to cortex grayordinates
        dlabels = dlabels[0:59412]
    # Find unique parcels
    unique_parcels = np.sort(np.unique(dlabels).astype(int))
    # Only include cortex (if flagged)
    unique_parcels = unique_parcels[:nparcels]
    
    #Create a dictionary for converting between parcel values and indices
    parcel_valueindex_dict={}
    for parcelind in np.arange(nparcels):
        parcel_valueindex_dict[unique_parcels[parcelind]]=parcelind
    
    if parcel_level:
        parcels_to_remove={}
    else:
        coords_to_remove_indices={}    

    for parcelInt,parcel in enumerate(unique_parcels):
        if verbose: print('Identifying grayordinates for target parcel',parcelInt,'-',int(parcel),'/',len(unique_parcels))
        
        # setup cortex/subcortex definitions
        if parcelInt < 360:
            dilatedmaskdir = dilatedmaskdir_cortex
            atlas_label = 'Cabnp'
        else:
            dilatedmaskdir = dilatedmaskdir_subcortex
            atlas_label = 'Cabnp'
            
        # Find where this parcel is in the unique parcel array
        parcel_ind = np.where(unique_parcels==parcel)[0]
        
        # Load in mask for target parcel
        parcel_mask = np.squeeze(nib.load(dilatedmaskdir + atlas_label + 'Parcel' + str(int(parcel)) + '_dilated_10mm.dscalar.nii').get_data())
        if cortex_only:
            #Restrict to cortex grayordinates
            parcel_mask = parcel_mask[0:59412]

        # get all target ROI indices
        #target_ind = np.squeeze(nib.load(dilatedmaskdir + atlas_label + 'Parcel' + str(int(parcel)) + '.dscalar.nii').get_data())
        #target_ind = np.where(dlabels==parcel)[0] # Find target parcel indices (from dlabel file)
        target_ind = np.asarray(dlabels==parcel,dtype=bool)
        if verbose: print('\t size of target:', np.sum(target_ind))
            
        #Identify set of vertices (outside target) to remove
        coords_to_remove=parcel_mask-target_ind
        coords_to_remove_indices_thisparcel=np.where(coords_to_remove)[0]
        
        #Record coordinates to remove for this target parcel
        if parcel_level:
            dlabels_coordstoremove=dlabels[coords_to_remove_indices_thisparcel]
            unique_parcelvals=np.unique(dlabels_coordstoremove).astype(int)
            #Convert from dlabel values to indices (0 to 359 for cortex, 0 to 717 for cortex+subcortex)
            unique_parcelvals_indices=np.zeros(len(unique_parcelvals))
            for i in np.arange(len(unique_parcelvals)):
                unique_parcelvals_indices[i]=int(parcel_valueindex_dict[unique_parcelvals[i]])
            parcels_to_remove[parcelInt]=unique_parcelvals_indices
            if verbose:
                print('parcels_to_remove:', parcels_to_remove[parcelInt])
        else:
            coords_to_remove_indices[parcel]=coords_to_remove_indices_thisparcel

    #Save to h5 file
    if parcel_level:
        
        if cortex_only:
            outputfilename = networkdefdir + 'parcels_to_remove_indices_cortexonly_data.h5'
        else:
            outputfilename = networkdefdir + 'parcels_to_remove_indices_cortexsubcortex_data.h5'
        print('Saving to h5 file:',outputfilename)
        h5f = h5py.File(outputfilename,'a')
        for parcelInt,parcel in enumerate(unique_parcels):
            outname1 = 'parcels_to_remove_indices'+'/'+str(int(parcelInt))
            try:
                h5f.create_dataset(outname1,data=parcels_to_remove[parcelInt])
            except:
                del h5f[outname1]
                h5f.create_dataset(outname1,data=parcels_to_remove[parcelInt])
        h5f.close()    
            
    else:
        
        if cortex_only:
            outputfilename = networkdefdir + 'coords_to_remove_indices_cortexonly_data.h5'
        else:
            outputfilename = networkdefdir + 'coords_to_remove_indices_cortexsubcortex_data.h5'
        print('Saving to h5 file:',outputfilename)
        h5f = h5py.File(outputfilename,'a')
        for parcelInt,parcel in enumerate(unique_parcels):
            outname1 = 'coords_to_remove_indices'+'/'+str(parcel)
            try:
                h5f.create_dataset(outname1,data=coords_to_remove_indices[parcel])
            except:
                del h5f[outname1]
                h5f.create_dataset(outname1,data=coords_to_remove_indices[parcel])
        h5f.close()
