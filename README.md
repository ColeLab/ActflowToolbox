# ActflowToolbox
## The Brain Activity Flow ("Actflow") Toolbox
A toolbox to facilitate discovery of how cognition & behavior are generated via brain network interactions

## Version 0.3.0

## Visit [https://colelab.github.io/ActflowToolbox/](https://colelab.github.io/ActflowToolbox/) for more information

### Version info:
* Version 0.3.0: Added Glasso FC to the set of connectivity methods (new recommended best practice for activity flow mapping); see updated HCP_example Jupyter notebook for a demo [2023-09-24]
* Version 0.2.6: Added combinedFC to the set of connectivity methods (current recommended best practice for activity flow mapping); see updated HCP_example Jupyter notebook for a demo
* Version 0.2.5: Fixed minor bug related to applying parcel level non-circular code to subcortical data.
* Version 0.2.4: Updated the non-circular code to be more efficient. Also created an easier and faster version of the non-circular approach that is at the parcel level (excluding all parcels within 10mm of the target parcel).

### Cite as:
1) Cole MW, Ito T, Bassett DS, Schultz DH (2016). "Activity flow over resting-state networks shapes cognitive task activations". Nature Neuroscience. 19:1718–1726.http://dx.doi.org/10.1038/nn.4406
2) https://github.com/ColeLab/ActflowToolbox/
3) The article that describes the specific toolbox functions being used in most detail

## How to install

1) git clone --recurse-submodules https://github.com/ColeLab/ActflowToolbox.git
2) We recommend using Anaconda (for Python 3), with JupyterLab (or Jupyter Notebooks). Many of the Python package dependencies for Actflow Toolbox will be included in Anaconda. You may need to add additional packages, however, such as GGlasso. For example, to add GGlasso, run this from the command line (after installing Anaconda): pip install gglasso

## How to use
1) See this paper for an overview of how to use the Brain Activity Flow Toolbox:
Cocuzza CV, Sanchez-Romero R, Cole MW (2022). "<a href="https://doi.org/10.1016/j.xpro.2021.101094">Protocol for activity flow mapping of neurocognitive computations using the Brain Activity Flow Toolbox</a>". STAR Protocols. 3, 1. doi:10.1016/j.xpro.2021.101094
2) Example notebook: https://colelab.github.io/ActflowToolbox/HCP_example.html

## Email list/forum
We strongly encourage you to join the ColeNeuroLab Users Group (https://groups.google.com/forum/#!forum/coleneurolab_users), so you can be informed about major updates in this repository and others hosted by the Cole Neurocognition Lab.

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
_Note: only a subset of files are listed and described_
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
* _Directory_: examples - Example analyses that use the Actflow Toolbox (Jupyter notebook)
* _Directory_: images - Example images generated by the Actflow Toolbox
* _Directory_: matlab_code - Limited functions for activity flow mapping in MATLAB
	* PCmultregressionconnectivity.m - Compute multiple regression-based functional connectivity; PC allows for more regions/voxels than time points.
	* actflowmapping.m - MATLAB version of actflowcalc.py; Main function for computing activity flow mapping predictions
	* multregressionconnectivity.m - Compute multiple regrression-based functional connectivity
* _Directory_: model_compare - Comparing prediction accuracies across models
	* model_compare_predicted_to_actual.py - Calculation of predictive model performance
	* model_compare.py - Reporting of model prediction performance, and comparison of prediction performance across models
* _Directory_: network_definitions - Data supporting parcel/region sets and network definitions
	* dilateParcels.py - Dilate individual parcels (cortex and subcortex) and produce masks to exclude vertices within 10 mm; requires Connectome workbench
* _Directory_: simulations - Simulations used for validating methods
* _Directory_: tools - Miscellaneous tools
	* addNetColors.py - Generates a heatmap figure with The Cole-Anticevic Brain-wide Network Partition (CAB-NP) colors along axes
	* addNetColors_Seaborn.py - Generates a Seaborn heatmap figure with The Cole-Anticevic Brain-wide Network Partition (CAB-NP) colors along axes
	* map_to_surface.py - Maps 2D matrix data onto a dscalar surface file (64k vertices); uses Glasser et al. 2016 ROI parcellation
	* max_r.py - Permutation testing to control for FWE (as in Nichols & Holmes, 2002 max-t); individual difference correlations (r)
	* max_t.py - Permutation testing to control for FWE (as in Nichols & Holmes, 2002); t-test variants (t)
	* regression.py - Compute multiple linear regression (with L2 regularization option)


