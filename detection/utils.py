import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "")

from simulation.generation.icd import ICD 


def get_ICD_with_Major():
    """Returns a list of all ICD codes with the first 3 digits of the code.
    
    Returns:
        np.array: Array of ICD codes with first digit.
    """
    icd_list = get_ICD_with_Minor()
    split_codes = np.char.split(icd_list.astype(str), sep=".")
    modified_codes = np.array(
        [code[0] if len(code) > 0 else "" for code in split_codes]
    )
    # modified_codes = np.append(modified_codes, "NaN")
    return np.unique(modified_codes)


def get_ICD_with_Minor():
    """Returns a list of all ICD codes.
    
    Returns:
        np.array: Array of ICD codes with first 4 digits.
    """

    icd_list = ICD().get_sec_icd_list()
    icd_list.extend(["n/A", "U12.5", "U12.8", "U12.9", "U11.9", "U99.2", "U07.7", "U13.1", "Z37.8"]) # manually adding codes that should not occur in the validation data as they are not valid in 2021 but still do
    return np.array(icd_list)

