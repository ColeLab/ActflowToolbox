function [netMat_betas] = multregressionconnectivity(activityMatrix)

%[netMat_betas] = multregressionconnectivity(activityMatrix)
%
%This function computes multiple regression-based functional connectivity. This is equivalent to running a general linear model (GLM) separately for each region, with all other regions' time series as regressors/predictors.
%
%Input: activityMatrix should be region/voxel X time (row=space, col=time)
%Output: netMat_betas will be a region/voxel X region/voxel connectivity matrix. The beta values indicate the value that each row region's (a source) time series needs to be linearly multiplied by to optimally (linearly) match a given column region's time series (the target), controlling for all other source regions' time series. This produces an asymmetric matrix, with the asymmetry due to different regions' time series being scaled differently. This is ideal for activity flow mapping but may be problematic for other uses. Consider using the MATLAB parcorr function as an alternative measure that produces symmetric functional connectivity matrices that are similar to multiple regression-based functional connectivity.
%
%This function requires that you have more time points than regions/voxels. Consider using PCmultregressionconnectivity if you have fewer time points than regions/voxels.
%
%Author: Michael W. Cole
%mwcole@mwcole.net
%http://www.colelab.org
%
%Version 1.0
%2016-08-24

if size(activityMatrix,1) > size(activityMatrix,2)
  disp('WARNING: You need to have more time points than voxels/regions to use this function.')
end

numRegions=size(activityMatrix,1);
netMat_betas=zeros(numRegions,numRegions);

%De-mean time series
activityMatrix=activityMatrix-repmat(mean(activityMatrix,2),1,size(activityMatrix,2));

for targetRegion=1:numRegions
    otherRegions=1:numRegions;
    otherRegions(targetRegion)=[];

    %Fit all regions' time series to current region's time series using regression
    stats = regstats(activityMatrix(targetRegion,:)', activityMatrix(otherRegions,:)', 'linear', {'beta'});
    netMat_betas(otherRegions,targetRegion)=stats.beta(2:end);
end
