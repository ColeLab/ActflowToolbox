# Taku Ito
# 12/12/2018
# Luke Hearne
# Feb, 2020
# dilateParcels.py
# On rutgers HCP with 1 node/8 cores/32gb mem - this script takes about 4 hrs to run

import numpy as np
import nibabel as nib
import pkg_resources
import multiprocessing as mp
import os
import time

## global variables for directories and atlas information

# CAB-NP directory
partitiondir = pkg_resources.resource_filename('ActflowToolbox.dependencies', 'ColeAnticevicNetPartition/')

# label file
defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'

# surface file
leftSurface = partitiondir +  'S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii'
rightSurface = partitiondir +  'S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii'

# output cortex file directory
dilatedmaskdir_cortex = pkg_resources.resource_filename('ActflowToolbox.network_definitions', 'CAB-NP/surfaceMasks/')

# output subcortex file directory
dilatedmaskdir_subcortex = pkg_resources.resource_filename('ActflowToolbox.network_definitions', 'CAB-NP/volumeMasks/')

##global settings that are used in sub functions!
dilateMM = 10 #default is 10mm
dlabels = np.squeeze(nib.load(defaultdlabelfile).get_data())  

def dilateParcels(dilateMM=10,nproc=10,verbose=True):

    '''
    This script dilates individual parcels by x mm
    Purpose is to use this as a mask to exclude any vertices within 10mm 
    of a parcel when estimaing FC via either ridge or multiple linear regression.
    Dilates all parcels (cortex and subcortex)
    Requires connectome workbench
    
    PARAMETERS:
        dilateMM : dilation in mm (default=10)
        verbose  : prints current roi
    '''
    #dlabels = np.squeeze(nib.load(defaultdlabelfile).get_data())
    parcel_list = np.unique(dlabels)

    inputs = []
    for parcel in parcel_list: inputs.append((parcel,verbose))
    
    pool = mp.Pool(processes=nproc)
    pool.starmap_async(_dilateOneParcel,inputs).get()
    pool.close()
    pool.join()

def _dilateOneParcel(parcel,verbose=False):
        if verbose:
            print('Dilating parcel', np.int(parcel))
            
        #dlabels = np.squeeze(nib.load(defaultdlabelfile).get_data())    
        parcel_array = np.zeros(dlabels.shape)

        # Find all vertices that don't correspond to this ROI`
        roi_ind = np.where(dlabels==parcel)[0]
        parcel_array[roi_ind] = 1.0

        if parcel < 361:
            maskfile   = dilatedmaskdir_cortex + 'CabnpParcel' + str(np.int(parcel))
            dilatedfile= dilatedmaskdir_cortex + 'CabnpParcel' + str(np.int(parcel)) + '_dilated_' + str(dilateMM) + 'mm'
        else:
            maskfile   = dilatedmaskdir_subcortex + 'CabnpParcel' + str(np.int(parcel))
            dilatedfile= dilatedmaskdir_subcortex + 'CabnpParcel' + str(np.int(parcel)) + '_dilated_' + str(dilateMM) + 'mm'

        # Write out masks to a dscalar file
        np.savetxt(maskfile + '.csv', parcel_array, fmt='%s')

        # Specify output of ROI specific mask and workbench commands
        wb_command = 'wb_command -cifti-convert -from-text ' + maskfile + '.csv ' + defaultdlabelfile + ' ' + maskfile + '.dscalar.nii -reset-scalars'
        os.system(wb_command)

        # Now dilate masks
        wb_command = 'wb_command -cifti-dilate ' + maskfile + '.dscalar.nii COLUMN ' + str(dilateMM) + ' ' + str(dilateMM) + ' ' + dilatedfile + '.dscalar.nii -left-surface ' + leftSurface + ' -right-surface ' + rightSurface
        os.system(wb_command)
        
        # Remove the original .csv file
        os.remove(maskfile + '.csv')
        
if __name__ == '__main__':
    # default settings
    start = time.time()
    dilateParcels(dilateMM=dilateMM,nproc=8,verbose=True)
    finish = time.time()
    print('Time elapsed:',finish-start)