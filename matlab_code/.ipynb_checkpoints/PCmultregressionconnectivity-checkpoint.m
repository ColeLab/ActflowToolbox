function [netMat_betas] = PCmultregressionconnectivity(activityMatrix)

%[netMat_betas] = PCmultregressionconnectivity(activityMatrix)
%
%This function computes multiple regression-based functional connectivity, but allows for more regions/voxels than time points. This is accomplished by decomposing the provided time series into a fewer number of components than time points. This allows multiple regression to be run properly, though some variance is ignored in this process. The multiple regression procedure is equivalent to running a general linear model (GLM) separately for each region, with all other regions' time series as regressors/predictors.
%
%Input: activityMatrix should be region/voxel X time (row=space, col=time)
%Output: netMat_betas will be a region/voxel X region/voxel connectivity matrix. The beta values indicate the value that each row region's (a source) time series needs to be linearly multiplied by to optimally (linearly) match a given column region's time series (the target), controlling for all other source regions' time series. This produces an asymmetric matrix, with the asymmetry due to different regions' time series being scaled differently. This is ideal for activity flow mapping but may be problematic for other uses. Consider using the MATLAB parcorr function as an alternative measure that produces symmetric functional connectivity matrices that are similar to multiple regression-based functional connectivity.
%
%If you have more time points than regions/voxels consider using multregressionconnectivity.
%
%Author: Michael W. Cole
%mwcole@mwcole.net
%http://www.colelab.org
%
%Version 1.1
%2016-08-30

numRegions=size(activityMatrix,1);
netMat_betas=zeros(numRegions,numRegions);

%De-mean time series
activityMatrix=activityMatrix-repmat(mean(activityMatrix,2),1,size(activityMatrix,2));

for targetRegion=1:numRegions
    otherRegions=1:numRegions;
    otherRegions(targetRegion)=[];

    %Run PCA
    numComponents=min(size(activityMatrix,1)-1,size(activityMatrix,2)-1);
    [PCALoadings,PCAScores] = pca(activityMatrix(otherRegions,:)','NumComponents',numComponents);

    %PCA regression to calculate FC
    seedTS=mean(activityMatrix(targetRegion,:),1);
    pcabetas = regress(seedTS', PCAScores(:,1:numComponents));
    betaPCR = PCALoadings*pcabetas;
    netMat_betas(otherRegions,targetRegion)=betaPCR;

end
