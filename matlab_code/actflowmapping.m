function [r_overall, p_overall, t_overall, r_bytask, p_bytask, taskActualMatrix, taskPredMatrix, r_bysubj, r_avgfirst_bytask, r_avgfirst_mean] = actflowmapping(taskActMatrix, connMatrix)

%actflowmapping(taskActVector, connMatrix)
%
%Input variables:
%taskActMatrix - A regionXtaskXsubject matrix with activation levels
%connMatrix - A regionXregionXstateXsubject matrix with connectivity values (e.g., Fisher's z-transformed Pearson correlations or multiple regression betas between resting-state fMRI time series)
%
%This function takes a set of task-evoked activations (e.g., fMRI GLM beta activations) and a set of connections (e.g., fMRI resting-state functional connectivity) and applies the "activity flow mapping" procedure. See Cole et al. (2016; Nature Neuroscience) for more information. Briefly, activity flow mapping quantifies the extent to which the connectivity patterns explain the linear transformations (activity flow) between brain regions for a given activation pattern. This involves predicting each region's activation level, one-at-a-time, based on multiplying all other activations by their connectivity strength with the to-be-predicted region. Activity flow mapping is applied here across multiple regions, allowing for single or multiple connectivity states (e.g., task-state functional connectivity), and allowing for multiple task-evoked activation patterns.
%
%Output variables:
%r_overall - The r-value comparing predicted to actual activation patterns averaged across all tasks
%p_overall - The p-value comparing predicted to actual activation patterns averaged across all tasks. This p-value is based on an across-subject paired t-test of the Fisher's z-transformed across-task average r-values.
%t_overall - The t-value associated with the p_overall value. Degrees of freedom = number of subjects - 1.
%r_bytask - The r-value comparing predicted to actual activation patterns for each task separately.
%p_bytask - The p-value associated with r_by task. This p-value is based on an across-subject paired t-test of the Fisher's z-transformed r-values for each task separately.
%taskActualMatrix - The actual task-evoked activation patterns. This is the same as the input taskActMatrix, execept that each task is (separately) z-normalized, facilitating comparing activation patterns via visualization (e.g., using imagesc). Format: regionXtaskXsubject.
%taskPredMatrix - The predicted task-evoked activation patterns via the activity flow mapping approach. Format: regionXtaskXsubject.
%r_bysubj - The r-value comparing predicted to actual activation patterns, separately for each task and each subject. Format: taskXsubject.
%r_avgfirst_bytask - The r-values (separately for each task) comparing predicted to actual activation patterns, computed after averaging the predicted activations across subjects and the actual activations across subjects. This tends to produce more accurate predictions, possibly due to higher signal-to-noise through averaging across subjects.
%r_avgfirst_mean - The across-task average of r_avgfirst_bytask.
%
%
%Author: Michael W. Cole
%mwcole@mwcole.net
%http://www.colelab.org
%
%Version 1.1
%2016-10-27
%
%Version history:
%1.0: Initial version
%1.1: Added error messages and warning messages when input data are not formatted appropriately
%

if ndims(connMatrix)==3
    msg='The connectivity matrix should be 4 dimensions (regions X regions X states X subjects). An empty third dimension is being added (assuming you have only a single connectivity state per subject (e.g., resting-state or structural connectivity data))';
    warning(msg);
    connMatrix_orig=connMatrix;
    connMatrix=zeros(size(connMatrix_orig,1),size(connMatrix_orig,2),1,size(connMatrix_orig,3));
    connMatrix(:,:,1,:)=connMatrix_orig;
elseif ndims(connMatrix)<3
    msg='The connectivity matrix should be 4 dimensions (regions X regions X states X subjects)';
    error(msg);
end
if ndims(taskActMatrix)<3
    msg='The task activity matrix must be 3 dimensions (regions X states/tasks X subjects)';
    error(msg);
end
    
numTasks=size(taskActMatrix,2);
numRegions=size(taskActMatrix,1);
numConnStates=size(connMatrix,3);
numSubjs=size(connMatrix,4);

%Setup for prediction
taskPredMatrix=zeros(numRegions,numTasks,numSubjs);
taskPredRs=zeros(numTasks,numSubjs);
taskActualMatrix=taskActMatrix;
regionNumList=1:numRegions;

for subjNum=1:numSubjs
    for taskNum=1:numTasks

        %Get this subject's activation pattern for this task
        taskActVect=taskActMatrix(:,taskNum,subjNum);

        for regionNum=1:numRegions

            %Hold out region whose activity is being predicted
            otherRegions=regionNumList;
            otherRegions(regionNum)=[];

            %Get this region's connectivity pattern
            if numConnStates > 1
                stateFCVect=connMatrix(:,regionNum,taskNum,subjNum);
            else
                %If using resting-state (or any single state) data
                stateFCVect=connMatrix(:,regionNum,1,subjNum);
            end

            %Calculate activity flow prediction
            taskPredMatrix(regionNum,taskNum,subjNum)=sum(taskActVect(otherRegions).*stateFCVect(otherRegions));

        end

        %Normalize values (z-score)
        taskPredMatrix(:,taskNum,subjNum)=(taskPredMatrix(:,taskNum,subjNum)-mean(taskPredMatrix(:,taskNum,subjNum)))./std(taskPredMatrix(:,taskNum,subjNum));
        taskActualMatrix(:,taskNum,subjNum)=(taskActMatrix(:,taskNum,subjNum)-mean(taskActMatrix(:,taskNum,subjNum)))./std(taskActMatrix(:,taskNum,subjNum));

        %Calculate predicted to actual similarity for this task
        r=corrcoef(taskPredMatrix(:,taskNum,subjNum),taskActualMatrix(:,taskNum,subjNum));
        taskPredRs(taskNum,subjNum)=r(1,2);
    end
end

%Calculate average r, across-subject p-value
r_bytask=tanh(mean(atanh(taskPredRs),2));
p_bytask=ones(numTasks,1);
for taskNum=1:numTasks
    [~, p_bytask(taskNum)]=ttest(atanh(taskPredRs(taskNum,:)));
end
r_overall=tanh(mean(mean(atanh(taskPredRs),1),2));
[~, p_overall, ~, stats]=ttest(mean(atanh(taskPredRs(taskNum,:)),1));
%By subj
r_bysubj=taskPredRs;
t_overall=stats.tstat;

%Calculate average-then-compare results
r_avgfirst_bytask=zeros(numTasks,1);
for taskNum=1:numTasks
    r_avgfirst_bytask(taskNum)=corr(mean(taskPredMatrix(:,taskNum,:),3),mean(taskActualMatrix(:,taskNum,:),3));
end
r_avgfirst_mean=tanh(mean(atanh(r_avgfirst_bytask)));
