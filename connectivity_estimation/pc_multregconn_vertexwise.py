
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np
import os
import pkg_resources
import nibabel as nib

partitiondir = pkg_resources.resource_filename('ActflowToolbox.dependencies', 'ColeAnticevicNetPartition/');
defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'; 
dilatedmaskdir_cortex = pkg_resources.resource_filename('ActflowToolbox.network_definitions', 'Glasser2016/surfaceMasks/');
    
def pc_multregconn_vertexwise(activity_matrix, parcel_num, dlabelfile=defaultdlabelfile, n_components=None):
    """
    A function to estimate functional connectivity with principal component regression, vertex-wise and with 10 mm dilated mask exclusion (in source timeseries), as in Cole et al., 2016 [Nat. Neuro.] but with  Glasser (2016) parcels (also: surface vertices instead of voxels) and the locally non-circular modification (10 mm mask). Incorporates code from pc_multregconn.py and calcconn_parcelwise_noncircular_surface.py. The main usage difference is that this will output connectivity estimates in vertex space (note: more computationally intensive, see input notes below for recommendations on increasing speed). 
    
    INPUTS
    activity_matrix: vertices (variables) x time or TRs (observations). Example: resting-state residuals yielded by pre-processing GLM(s). Note: source and target time-series will be generated by parcel_num input.
    parcel_num: specifies the Glasser (2016) parcel to process; scalar value in range of 1-360 (see supplement of 2016 paper for parcel list). Whole-cortex results can be obtained by iterating over parcel_num (1-360). 
    dlabelfile: parcellation file; each vertex indicates the number corresponding to each parcel. dlabelfile needs to match same vertex dimensions of data
    n_components: Optional (but highly recommended for speed in vertex-wise application). Number of PCA components to use. If None, the full set of components will be used (equal to time dimension - 1; for example, number of resting-state TRs in activity_matrix - 1). 
    
    
    OUTPUT
    connectivity_mat: formatted as targets x sources. Main difference from pc_multregconn is that it returns data in vertex space. Target and source sizes will be different for each parcel (each contains a different number of vertices). Note: this will be of size 59412 x 59412, with zeros in rows and columns that are not indexed as target and source vertices (bigger matrix, but allows later concatenation when iterating over whole-cortex; otherwise can remove rows/cols that have zeros and use that data as parcel vertices; average on row dimension to get 1 parcel value for each vertex source)
    
    POTENTIAL TO-DO LIST: 
    - modify output format? currently has zeros in non-parcel vertex locations; maybe return target/source indices as separate outputs (along with FC matrix; non-zero'd) and let user index themselves? 
    - add in a cross validation procedure to optimize number of components (as an alternative to n_components parameter input above); for consistency, use n_comp_search, n_components_min, and n_components_max syntax from pc_multregconn.py 
    - subcortex options & input to set the default dlabel file & directory 
    - a sister script to re-combine whole-cortex results into 1 graph (reorganized back into original vertex order, and/or into CAB-NP order)? 
    - inquire about hemi switching in 10mm dilated mask file
    """
    
    # Define key variables, paths, etc.
    [numVertices,numTRs] = activity_matrix.shape; 
    
    #partitiondir = pkg_resources.resource_filename('ActflowToolbox.dependencies', 'ColeAnticevicNetPartition/'); 
    #defaultdlabelfile = partitiondir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'; 
    #dilatedmaskdir_cortex = pkg_resources.resource_filename('ActflowToolbox.network_definitions', 'Glasser2016/surfaceMasks/');
    #dilatedmaskdir_subcortex = pkg_resources.resource_filename('ActflowToolbox.network_definitions', 'CAB-NP/volumeMasks/');

    dlabels = np.squeeze(nib.load(dlabelfile).get_data()); # 91282 x 1 (includes cortex + subcortex, first 59412 are cortical vertices), dlabels[:nVertices] would give parcel #s 1-360
    dlabelsCortOnly = dlabels[:numVertices]; 
    dilatedmaskdir = dilatedmaskdir_cortex; 
    atlas_label = 'Glasser'; 
    
    # Define target data: vertices in ROI, used as y in regression (1 vertex at a time)
    targetVertices = np.where(dlabelsCortOnly==parcel_num)[0]; 
    targetData = activity_matrix[targetVertices,:]; 
    
    # Define source data: vertices not in ROI or 10mm out from ROI, used in PC step (variables or features = vertices; observations or samples = TRs)
    if parcel_num<=180: # hemis seem to be switched?
        parcel_num_adjhemi = parcel_num + 180;
    elif parcel_num>=181:
        parcel_num_adjhemi = parcel_num - 180;
    parcelMask = np.squeeze(nib.load(dilatedmaskdir + atlas_label + 'Parcel' + str(int(parcel_num_adjhemi)) + '_dilated_10mm.dscalar.nii').get_data()); # binary, yes or no as 10 mm outside 
    parcelMaskVertices = np.where(parcelMask==1.0)[0]; 
    data_copy = np.copy(activity_matrix); 
    data_copy[targetVertices,:] = 0; 
    data_copy[parcelMaskVertices,:] = 0; 
    roiDelIx = np.where(data_copy[:,0]==0)[0];
    sourceData = np.delete(data_copy,roiDelIx,0); 
    
    # run PCA on source data (https://nirpyresearch.com/principal-component-regression-python/)
    # xReg: PCA scores; pcComps: coefficients or loadings
    if n_components is not None: 
        pca = PCA(n_components=n_components); 
        xReg = pca.fit_transform(sourceData.T); # TRs x n_components
        pcComps = pca.components_.T; # vertices x n_components 
        #varExplained = pca.explained_variance_ratio_; # will not add to 100% because it is constrained by n_components 
    else: 
        pca = PCA(); 
        xReg = pca.fit_transform(sourceData.T); # if # features (vertices) > # observations (TRs), xReg will be square (TRs x TRs)
        pcComps = pca.components_.T; # full size (and transposed to original dims), or vertices x TRs 
        #varExplained = pca.explained_variance_ratio_; # should add to 100%  
    
    # run regression, looped over target vertices & tranformed to get get functional connectivity estimates 
    numTargetVertices = targetData.shape[0]; 
    numSourceVertices = sourceData.shape[0]; 
    betasPCregAll = np.zeros((numSourceVertices,numTargetVertices)); 
    for vertexNum in list(range(numTargetVertices)):
        targetVertexData = targetData[vertexNum,:]; 
        regr = LinearRegression();
        regr.fit(xReg, targetVertexData); 
        regBetasHere = regr.coef_; 
        betasPCregAll[:,vertexNum] = pca.inverse_transform(regBetasHere); 
    
    # output 
    connEstimates = betasPCregAll.T;
    connectivity_mat = np.zeros((numVertices,numVertices)); # initialize the full matrix 

    for rowTarget in list(range(numTargetVertices)):
        for colSource in list(range(numSourceVertices)): 
            connEstimate = testConn[rowTarget,colSource]; 
            targetIndex = targetVertices[rowTarget]; sourceIndex = xx[colSource]; 
            connectivity_mat[targetIndex,sourceIndex] = connEstimate;
    
    return connectivity_mat; # TO-DO: a helper function for when this is iterated over all cortical parcels; to concatenate into 1 full graph 
    