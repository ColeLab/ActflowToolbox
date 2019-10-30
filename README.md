# ActflowToolbox
## The Brain Activity Flow ("Actflow") Toolbox

## Version 0.1.0 (alpha version)

## Overview
This toolbox facilitates linking brain connectivity with cognitive/behavioral functions and task-evoked activity. The primary focus is on _activity flow mapping_ (http://rdcu.be/kOJq) and related methods such as _information transfer mapping_ (http://rdcu.be/wQ1M). These approaches can be used to create and test _network coding models_ (http://arxiv.org/abs/1907.03612), which are empirically-constrained neural network models.

Other included methods that can be used along with activity flow mapping (or not) include advanced versions of task-state functional connectivity, resting-state functional connectivity, and general linear modeling (multiple regression). Supporting methods such as preprocessing and simulations for validation will also be included [planned]. The primary focus (for now) is on fMRI and EEG/MEG data, but in principle these approaches can be applied to other kinds of data.

This code repository was created by the Cole Neurocognition Lab at Rutgers University: http://www.colelab.org/ We are happy for others to contribute to the code as well (contact Michael Cole if you are interested).

### Core algorithm and principles behind activity flow mapping
Activity flow mapping is a method validated and published in Nature Neuroscience in 2016 (see http://rdcu.be/kOJq), with multiple follow-up studies further validating the approach.

**Activity flow mapping has two major steps**:
1) _Activity flow modeling_: Simulating the flow (movement) of activity between neural populations, using empirical data to estimate activity levels and routes of flow (connectivity). This is done by running the standard "propagation rule" and "activation rule" steps used in most neural network simulations (see http://arxiv.org/abs/1907.03612). In the standard approach each neural population's activity level is predicted based on the flow from every other recorded neural population:

![Alt text](images/actflowfigure1.png "Illustration of activity flow mapping algorithm")

2) _Testing prediction accuracy_: The predicted brain activity pattern is compared to the actual recorded activity, providing evidence for or against the activity flow model used to make the predictions. Predictions of multiple models can be compared to quantify model validity.

In principle, any measure of neural activity and any measure of neural connectivity can be used with activity flow mapping. Some modifications are necessary for fast temporal resolution data such as EEG/MEG or multi-unit recording, but these should be available soon (we are working on it!).

### Included connectivity-activity mapping methods
* Activity flow mapping (http://rdcu.be/kOJq)
* Information transfer mapping (http://rdcu.be/wQ1M) [planned]

### Included connectivity mapping methods
* _All methods can be applied to resting-state or task-state data_
* Correlation-based functional connectivity
* Multiple-regression functional connectivity
	* Ordinary least squares multiple regression connectivity
	* Regularized multiple regression connectivity
		* Principle components regression connectivity (PCR)
* Partial-correlation functional connectivity
	* Inverse covariance-based partial correlation
	* Regularized partial correlation [planned]
* Special preprocessing for task-state functional connectivity
	* Flexible mean task-evoked response removal (http://www.colelab.org/pubs/ColeEtAl2019NeuroImage.pdf) [planned]
* Causal connectivity (fGES; https://doi.org/10.1007/s41060-016-0032-z) [planned]

### Cite as:
Cole MW, Ito T, Bassett DS, Schultz DH (2016). "Activity flow over resting-state networks shapes cognitive task activations". Nature Neuroscience. 19:1718–1726.http://dx.doi.org/10.1038/nn.4406
and https://github.com/ColeLab/ActflowToolbox/ 
and the article that describes the specific toolbox functions used in most detail.

## How to install

git clone --recurse-submodules https://github.com/ColeLab/ActflowToolbox.git

## Email list/forum
We strongly encourage you to join the ColeNeuroLab Users Group (https://groups.google.com/forum/#!forum/coleneurolab_users), so you can be informed about major updates in this repository and others hosted by the Cole Neurocognition Lab.

## Conventions
* Data matrices all node X time
* Directed connectivity matrices all target X source
* Primary (default) brain parcellation: CAB-NP (https://github.com/ColeLab/ColeAnticevicNetPartition), which uses the Glasser2016 parcellation for cortex (https://balsa.wustl.edu/study/show/RVVG) and includes an additional 358 subcortical parcels. Cortex-only analyses with only the Glasser2016 parcels are easily implemented. Other parcellations can be used, and we anticipate integrating other parcellations in the future.

## Example
Calculating activity flow mapping predictions using multiple-regression FC and standard task-evoked activations with fMRI data (in Python 3; assumes task activations and resting-state fMRI data are already loaded):
```import ActflowToolbox as actflow
restFC_mreg=actflow.connectivity_estimation.multregconn(restdata)
print("==Activity flow mapping results, multiple-regression-based resting-state FC, 24 task conditions==")
actflowOutput_restFCMReg_bycond = actflow.actflowcomp.actflowtest(activations_bycond, restFC_mreg, print_by_condition=False)
```
Output:
```
==Activity flow mapping results, multiple-regression-based resting-state FC, 24 task conditions==
===Comparing prediction accuracies between models (similarity between predicted and actual brain activation patterns)===
 
==Condition-wise correlations between predicted and actual activation patterns (calculated for each node separetely):==
--Compare-then-average (calculating prediction accuracies before cross-subject averaging):
Each correlation based on N conditions: 24, p-values based on N subjects (cross-subject variance in correlations): 176
Mean Pearson r=0.89, t-value vs. 0: 171.05, p-value vs. 0: 1.0573286644474424e-196
Mean rank-correlation rho=0.79, t-value vs. 0: 173.30, p-value vs. 0: 1.0891488047331093e-197
 
==Condition-wise correlations between predicted and actual activation patterns (calculated for each node separetely):==
--Average-then-compare (calculating prediction accuracies after cross-subject averaging):
Each correlation based on N conditions: 24
Mean Pearson r=0.98
Mean rank-correlation rho=0.94
 
==Node-wise (spatial) correlations between predicted and actual activation patterns (calculated for each condition separetely):==
--Compare-then-average (calculating prediction accuracies before cross-subject averaging):
Each correlation based on N nodes: 360, p-values based on N subjects (cross-subject variance in correlations): 176
Cross-condition mean r=0.87, t-value vs. 0: 152.54, p-value vs. 0: 4.6587650313354306e-188
 
==Node-wise (spatial) correlations between predicted and actual activation patterns (calculated for each condition separetely):==
--Average-then-compare (calculating prediction accuracies after cross-subject averaging):
Each correlation based on N nodes: 360, p-values based on N subjects (cross-subject variance in correlations): 176
Mean r=0.97
```

## Software development guidelines
* Primary language: Python 3
* Secondary language (for select functions, minimally maintained/updated): MATLAB
* Versioning guidelines: Semantic Versioning 2.0.0 (https://semver.org/); used loosely prior to v1.0.0, strictly after
* Using GitHub for version control
	* Those new to Git should go through a tutorial for branching, etc.: https://www.youtube.com/watch?v=oFYyTZwMyAg and https://guides.github.com/activities/hello-world/
	* Use branching for adding new features, making sure code isn't broken by changes
	* Considering using unit tests and Travis CI (https://travis-ci.org) in future
* Style specifications:
	* PEP8 style as general guidelines (loosely applied for now): https://www.python.org/dev/peps/pep-0008/
	* Soft tabs (4 spaces) for indentations [ideally set "soft tabs" setting in editor, so pressing tab key produces 4 spaces]
	* Use intuitive variable and function names
	* Add detailed comments to explain what code does (especially when not obvious)

## Contents
* _Directory_: actflowcomp - Calculating activity flow mapping
	* actflowcalc.py - Main function for calculating activity flow mapping predictions
	* actflowtest.py - A convenience function for calculating activity-flow-based predictions and testing prediction accuracies (across multiple subjects)
	* noiseceilingcalc.py - A convenience function for calculating the theoretical limit on activity-flow-based prediction accuracies (based on noise in the data being used)
* _Directory_: connectivity_estimation - Connectivity estimation methods
	* calcactivity_parcelwise_noncircular_surface.py: High-level function for calculating parcelwise actflow with parcels that are touching (e.g., the Glasser 2016 parcellation), focusing on task activations. This can create circularity in the actflow predictions due to spatial autocorrelation. This function excludes vertices within X mm (10 mm by default) of each to-be-predicted parcel.
	* calcconn_parcelwise_noncircular_surface.py: High-level function for calculating parcelwise actflow with parcels that are touching (e.g., the Glasser 2016 parcellation), focusing on connectivity estimation. This can create circularity in the actflow predictions due to spatial autocorrelation. This function excludes vertices within X mm (10 mm by default) of each to-be-predicted parcel.
	* corrcoefconn.py: Calculation of Pearson correlation functional connectivity
	* multregconn.py: Calculation of multiple-regression functional connectivity
	* partial_corrconn.py: Calculation of partial-correlation functional connectivity
	* pc_multregconn.py: Calculation of regularized multiple-regression functional connectivity using principle components regression (PCR). Useful when there are fewer time points than nodes, for instance.
* _Directory_: dependencies - Other packages Actflow Toolbox depends on
* _Directory_: infotransfermapping - Calculating information transfer mapping
* _Directory_: latent_connectivity - Calculating latent functional connectivity via factor analysis [planned]
* _Directory_: matlab_code - Limited functions for activity flow mapping in MATLAB
* _Directory_: model_compare - Comparing prediction accuracies across models
	* model_compare_predicted_to_actual.py - Calculation of predictive model performance
	* model_compare.py - Reporting of model prediction performance, and comparison of prediction performance across models
* _Directory_: network_definitions - Data supporting parcel/region sets and network definitions
* _Directory_: pipelines - Example pipelines for data analyses
* _Directory_: preprocessing - Functions for preprocessing (after "minimal" preprocessing)
* _Directory_: simulations - Simulations used for validating methods
* _Directory_: tests - Code for testing various parts of the toolbox
* _Directory_: tools - Miscellaneous tools
