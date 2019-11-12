import numpy as np
import nibabel as nib
import os
import pkg_resources
#import subprocess

#toolsdir = pkg_resources.resource_filename('ActflowToolbox.tools', '/')
#glasserfile2=toolsdir+'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
dependenciesdir = pkg_resources.resource_filename('ActflowToolbox.dependencies', '/')
glasserfile2=dependenciesdir+'ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'

def map_to_surface(mat,filename,nParcels=360,glasserfile2=glasserfile2,fliphemispheres=False):
    """
    Maps a region X column 2d matrix into a dscalar file with 64k vertices
    Uses the Glasser et al. 2016 ROI parcellation
    
    Input Parameters:
        mat     :   region x column (features/activations, etc.) 2D MATRIX to be mapped onto the surface. MUST BE A 2D MATRIX.
                    mat can either be 360 mat or ~59k mat. If 360, will automatically map back to ~59k

        filename:   a string indicating the directory + filename of the output. Do not include a suffix (e.g., ".dscalar.nii" to the file. Suffixes will be added automatically.
        
        fliphemispheres: If the data were originally loaded using RL (right hemisphere then left) convention the data should be
        flipped, since CAB-NP uses LR (left hemisphere then right). A setting of True will flip the hemispheres.

    """
    
    if fliphemispheres:
        print('Flipping hemispheres')
        newmat=np.zeros(mat.shape)
        newmat[0:180]=mat[180:360]
        newmat[180:360]=mat[0:180]
        mat=newmat.copy() 
    
    #### Load glasser atlas
    glasser2 = nib.load(glasserfile2).get_data()
    glasser2 = np.squeeze(glasser2)
    #### Map back to surface
    if mat.shape[0]==nParcels:
        out_mat = np.zeros((glasser2.shape[0],mat.shape[1]))
        for roi in np.arange(nParcels):
            vertex_ind = np.where(glasser2==roi+1)[0]
            for col in range(mat.shape[1]):
                out_mat[vertex_ind,col] = mat[roi,col]
    else:
        out_mat = mat

    #### 
    # Write file to csv and run wb_command
    np.savetxt(filename + '.csv', out_mat,fmt='%s')
    wb_file = filename + '.dscalar.nii'
    wb_command = 'wb_command -cifti-convert -from-text ' + filename + '.csv ' + glasserfile2 + ' ' + wb_file + ' -reset-scalars'
    #os.system(wb_command)
    print('Command:')
    print(wb_command)
    try:
        #subprocess.call(wb_command)
        os.system(wb_command)
        os.remove(filename + '.csv')
        print("CIFTI dscalar is output as:" + wb_file)
    except OSError:
        print ('wb_command does not exist')
    
