# Taku Ito
# 12/12/2018

# This script dilates individual parcels by x mm (default 10mm)
# Purpose is to use this as a mask to exclude any vertices within 10mm of a parcel when estimaing FC via either ridge or multiple linear regression

import numpy as np
import nibabel as nib
import os

nParcels = 360
dilateMM = 10

labeldir = '/projects/AnalysisTools/ParcelsGlasser2016/'
dlabelfile = labeldir + 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'

surfacedir = labeldir + 'Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/'
leftSurface = surfacedir + 'Q1-Q6_RelatedParcellation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'
rightSurface = surfacedir + 'Q1-Q6_RelatedParcellation210.R.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'

maskdir = '/projects3/SRActFlow/data/results/surfaceMasks/'

execute = 1
if execute == 1:
    glasser = np.squeeze(nib.load(dlabelfile).get_data())

    for parcel in range(1,nParcels+1):
        print('Dilating parcel', parcel)

        parcel_array = np.zeros(glasser.shape)
        # Find all vertices that don't correspond to this ROI`
        roi_ind = np.where(glasser==parcel)[0]
        parcel_array[roi_ind] = 1.0

        ## Write out masks to a dscalar file
        maskfile = maskdir + 'GlasserParcel' + str(parcel)
        np.savetxt(maskfile + '.csv', parcel_array, fmt='%s')
        # Specify output of ROI specific mask and workbench commands
        wb_command = 'wb_command -cifti-convert -from-text ' + maskfile + '.csv ' + dlabelfile + ' ' + maskfile + '.dscalar.nii -reset-scalars'
        os.system(wb_command)
        
        ## Now dilate masks
        dilatedfile = maskdir + 'GlasserParcel' + str(parcel) + '_dilated_' + str(dilateMM) + 'mm'
        wb_command = 'wb_command -cifti-dilate ' + maskfile + '.dscalar.nii COLUMN ' + str(dilateMM) + ' ' + str(dilateMM) + ' ' + dilatedfile + '.dscalar.nii -left-surface ' + leftSurface + ' -right-surface ' + rightSurface
        os.system(wb_command)


