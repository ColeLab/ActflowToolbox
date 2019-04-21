# ActflowToolbox
## The Brain Activity Flow ("Actflow") Toolbox

## Overview
The purpose of this software package is to facilitate use of neuroscience methods linking connectivity with cognitive/behavioral functions and task-evoked activity. The primary focus is on _activity flow mapping_ (http://rdcu.be/kOJq) and related methods such as _information transfer mapping_ (http://rdcu.be/wQ1M).

Other included methods that can be used along with activity flow mapping (or not) include advanced versions of task-state functional connectivity, resting-state functional connectivity, and general linear modeling (multiple regression). Supporting methods such as preprocessing and simulations for validation are also included.

### Included connectivity-activity mapping methods
* Activity flow mapping (http://rdcu.be/kOJq)
* Information transfer mapping (http://rdcu.be/wQ1M)

### Included connectivity mapping methods
* Resting-state functional connectivity (correlation and multiple regression)
* Task-state functional connectivity (correlation and multiple regression)
	* With mean task-evoked response removal (http://www.colelab.org/pubs/ColeEtAl2019NeuroImage.pdf)

## Development guidelines
* Primary language: Python 3
* Secondary language (for select functions, minimally maintained/updated): MATLAB
* Primary (default) brain parcellation: CAB-NP (https://github.com/ColeLab/ColeAnticevicNetPartition), which uses the Glasser2016 parcellation for cortex (https://balsa.wustl.edu/study/show/RVVG)
* Versioning guidelines: Semantic Versioning 2.0.0 (https://semver.org/); used loosely prior to v1.0, strictly after

## Content
* _Directory_: actflowcalc - Calculating activity flow mapping
* _Directory_: connectivity_estimation - Connectivity estimation methods
* _Directory_: dependencies - Other packages Actflow Toolbox depends on
* _Directory_: infotransfermapping - Calculating information transfer mapping
* _Directory_: matlab_code - Limited functions for activity flow mapping in MATLAB
* _Directory_: network_definitions - Data supporting parcel/region sets and network definitions
* _Directory_: pipelines - Example pipelines for data analyses
* _Directory_: preprocessing - Functions for preprocessing (after "minimal" preprocessing)
* _Directory_: simulations - Simulations used for validating methods
* _Directory_: tools - Miscellaneous tools
