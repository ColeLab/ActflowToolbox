function firDesignMat = converTaskTimingToFIRDesignMat(taskDesignMat,firLag)
    %Taku Ito (02/19/2018)
    %Takes in a timepoint x task regressor matrix (task design matrix filled with 1s and 0s), and returns a design matrix for FIR regression
    %
    %PARAMETERS:
    %
    %taskDesignMat       -       timepoint x task regressor matrix (1s and 0s). Number of columns corresponds to number of conditions
    %firLag              -       Number of time points after block offset to include in FIR model. This is to account for the HRF undershoot. Usually want to include 15-20s after block offset. For HCP sampling rate of 0.785s or 0.72s, 25 TRs is recommended (would account for ~19s of offshoot after block ends)
    %
    %returns
    %    firDesignMat    -       timepoint x FIR regressor matrix (1s and 0s). Number of columns correspond to (length of the block + some additional lag) * number of conditions
    %
    %

    n_conditions = size(taskDesignMat,2);
    n_timepoints = size(taskDesignMat,1);

    % Create FIR regressors for each condition
    firDesignMat = [];
    for cond=1:n_conditions
        % Find time points for this particular condition
        task_timing = taskDesignMat(:,cond);

        % block onsets are when the task timing transitions from 0 to 1, and then the 
        block_onsets = find(diff(task_timing)==1) + 1;
        % block offsets are when the task timing transitions from 1 to 0 (i.e., the temporal derivative is -1)
        block_offsets = find(diff(task_timing)==-1) + 1;
        

        % Identify the longest block (that will be the number of regressors we have, and add the lag for HRF)
        block_length = max(block_offsets-block_onsets) + firLag;

        % count number of blocks
        n_blocks = length(block_onsets);

        fir_cond_mat = zeros(n_timepoints, block_length);

        for block=1:n_blocks
            % Define the TR where the block starts
            trcount = block_onsets(block);
            for i=1:block_length
                % Make sure the TRcount doesn't exceed number of TRs in the run... don't want to make the FIR regressor matrix larger than the original regressor matrix
                if trcount <= n_timepoints
                    fir_cond_mat(trcount,i) = 1;
                    trcount = trcount + 1;
                else
                    continue
                end
            end
        end

        % Horizontally stack this design matrix with the design matrix designed for other regressors
        firDesignMat = [firDesignMat fir_cond_mat];

end


