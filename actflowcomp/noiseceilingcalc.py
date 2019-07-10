
import numpy as np
import scipy.stats

def noiseceilingcalc(actvect_group, full_report=False, print_report=True, reliability_type='crosscondition_avgthencomp'):
    """
    Function to calculate the repeat reliability of the data in various ways. This is equivalent to calculating the "noise ceiling" for predictive models (such as encoding models like activity flow models). 
    
    actVect_group: node x condition x subject matrix with activation values
    fcMat_group: node x node x condition x subject matrix (or node x node x subject matrix) with connectiivty values
    separate_activations_bytarget: indicates if the input actVect_group matrix has a separate activation vector for each target (to-be-predicted) node (e.g., for the locally-non-circular approach)
    """