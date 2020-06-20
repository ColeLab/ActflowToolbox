
import sys
sys.path.insert(0, '/projects/f_mc1689_1/TaskFCActflow/docs/scripts/final_analyses/')
sys.path.insert(0, '/projects/f_mc1689_1/TaskFCActflow/docs/scripts/final_analyses/ActflowToolbox/network_definitions/')

import ActflowToolbox as actflow
from calc_removal_coords import *

#First run dilateParcels.py, to dilate all of the parcels in the CIFTI network partition
#dilateParcels()

#Second, run calc_removal_coords.py to calculate which grayordinates to remove for each target region
#This second step speeds up the non-circular code by preventing the need to load all of the dilated masks one-at-a-time
calc_removal_coords(cortex_only=False, verbose=True)
