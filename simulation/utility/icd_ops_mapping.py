import sys

sys.path.insert(0, "")

import simulation.generation.drg as drgo
from simulation.utility.utils import flatten

import json
import marisa_trie
import pandas as pd


class ICD_OPS_Mapping:
    """This class is used to map ICD codes to OPS codes. It uses the OPS codes from the ICD-OPS mapping.

    Attributes:
        ops (simulation.generation.ops.OPS): the OPS object
    """

    def __init__(self, ops):
        self.file = open("data/OPS/icd_ops_mapping.json", "r")
        self.icd_ops = json.load(self.file)
        self.file.close()
        self.keys = self.icd_ops.keys()
        self.ops = ops

    def get_operations_ops(self, icd):
        """
        Get the 5-er OPS codes for a given ICD code.

        Args:
            icd (str): the ICD code

        Returns:
            list of strings: the list of possible OPS codes
        """
        icd = icd.split(".")[0]
        ops = []
        if icd in self.keys:
            o = self.icd_ops[icd]
        elif icd[0] in self.keys:
            o = self.icd_ops[icd[0]]
        else:
            return None

        ops = self.ops.get_ops_list()
        ops = [s[:5] for s in ops]
        ops = list(set(ops))  # getting unique values
        trie = marisa_trie.Trie(ops)

        final = []

        for i in o:
            final.append(trie.keys(i))

        final = flatten(final)
        return final




