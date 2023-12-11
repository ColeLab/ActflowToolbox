# ActflowToolbox
## The Brain Activity Flow ("Actflow") Toolbox
A toolbox to facilitate discovery of how cognition & behavior are generated via brain network interactions

## Version 0.3.1

## Visit [https://colelab.github.io/ActflowToolbox/](https://colelab.github.io/ActflowToolbox/) for more information

### Version info:
* Version 0.3.1: Minor update, fixing a bug in the map_to_surface.py function [2023-12-06]
* Version 0.3.0: Added Glasso FC to the set of connectivity methods (new recommended best practice for activity flow mapping); see updated HCP_example Jupyter notebook for a demo [2023-09-24]
* Version 0.2.6: Added combinedFC to the set of connectivity methods (current recommended best practice for activity flow mapping); see updated HCP_example Jupyter notebook for a demo
* Version 0.2.5: Fixed minor bug related to applying parcel level non-circular code to subcortical data.
* Version 0.2.4: Updated the non-circular code to be more efficient. Also created an easier and faster version of the non-circular approach that is at the parcel level (excluding all parcels within 10mm of the target parcel).

### Cite as:
1) Cole MW, Ito T, Bassett DS, Schultz DH (2016). "Activity flow over resting-state networks shapes cognitive task activations". Nature Neuroscience. 19:1718–1726.http://dx.doi.org/10.1038/nn.4406
2) https://github.com/ColeLab/ActflowToolbox/
3) The article that describes the specific toolbox functions being used in most detail

## How to install

<p><i>Option 1:</i>
    <br><i>Within an Anaconda environment:</i> conda install -c conda-forge actflow
  </p>
  <p><i>Option 2:</i>
    <br>pip install actflow
  </p>
  <p><i>Option 3:</i>
    <br>git clone --recurse-submodules https://github.com/ColeLab/ActflowToolbox.git
  </p>

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
	* Those new to Git should go through a tutorial for branching, etc.: https://www.youtube.com/watch?v=oFYyTZwMyAg and https://guides.github.com/activities/hello-world/ and https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging
	* Use branching for adding new features, making sure code isn't broken by changes
	* Considering using unit tests and Travis CI (https://travis-ci.org) in future
* Style specifications:
	* PEP8 style as general guidelines (loosely applied for now): https://www.python.org/dev/peps/pep-0008/
	* Soft tabs (4 spaces) for indentations [ideally set "soft tabs" setting in editor, so pressing tab key produces 4 spaces]
	* Use intuitive variable and function names
	* Add detailed comments to explain what code does (especially when not obvious)
   
* How to update PIP when gitHub code gets updated? <br>
Steps:
	* Clone/Pull the ActflowToolbox gitHub repository which has been updated with new code.
	* Create a setup.py file/use the one already present and place it at the location parallel to the ActflowToolbox folder that was just cloned.
	* Increment version number in setup.py with latest version.
	* setup.py file would have dependency on setuptools.
Run :  pip3 install setuptools.
	* Test code locally using command “pip3 install .” in the directory where the setup.py file is present. (using Terminal). If build is successful you are good to go ahead with deployment on the official PyPI channel. If not you might need script correction for the setup.py file.
	* python3 setup.py sdist bdist_wheel (using Terminal). This creates a dist folder which we need to upload on PyPI official website.
	* pip3 install twine (using Terminal) 
	* twine upload dist/* (using Terminal)
    		* when prompted to credentials use username = __token__ & password = API token
    		* API Token can be extracted from Pypi account (Ask Prof. Mike for token in case you don't have access to Pypi account)
Reference - ​​ [Publish Your Own Python Package](https://www.youtube.com/watch?v=tEFkHEKypLI&t=436s&ab_channel=NeuralNine)

* How to update Conda when gitHub code gets updated? <br>
Steps:
	* Once the package is deployed with PyPI you don’t need to do anything else.

	* To check if deployment is successful follow this ➖
		* The feedstock (repository where the conda-forge package is built) will receive an automated PR shortly after (within a couple hours most of the time) the publication on PyPI.
		* Once that happens, follow the PR description and review that the recipe is up-to-date (e.g. check if the runtime dependencies changed) and that the CI is passing using the following PR link.
		* Reference - 
		https://github.com/conda-forge/actflow-feedstock (Feedstock for Actflow). Using this link check if build succeeded for updated code.
		* Go to the PULL REQUEST tab and follow point 2. Validate and tick all the checklist items.
		If everything's in place merge the Pull Request.
	* Pull request link (When PR was requested first time) - https://github.com/conda-forge/staged-recipes/pull/24411
	
	
	* Gitter Channel - https://app.gitter.im/#/room/#conda-forge:matrix.org (Community chat for any issues related to CONDA published packages)
	
	* Here is some additional information regarding PR - 
	
	
		* Feel free to push to the bot's branch to update this PR if needed. The bot will almost always only open one PR per version. 
		* The bot will stop issuing PRs if more than 3 version bump PRs generated by the bot are open. If you don't want to package a particular version please close the PR. 
		* If you want these PRs to be merged automatically, make an issue with @conda-forge-admin,please add bot automerge in the title and merge the resulting PR. 
		* This command will add our bot automerge feature to your feedstock. If this PR was opened in error or needs to be updated please add the bot-rerun label to this PR.
		* The bot will close this PR and schedule another one. 
		* If you do not have permissions to add this label, you can use the phrase @conda-forge-admin, please rerun bot in a PR comment to have the conda-forge-admin add it for you.

 	* Example build status - Succeded
    	
		![Screenshot 2023-12-08 at 10 11 37 AM](https://github.com/SammedAdmuthe/ActflowToolbox/assets/36372399/e9fbe888-a7e5-4b75-a670-325c1c18a296)



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
