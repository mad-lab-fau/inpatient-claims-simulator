import sys

sys.path.insert(0, "")

import pandas as pd

import simulation.generation.drg as drg

"""
This file is used to calculate the claim for the most common DRGs in 2016, using the DRG rules from 2021. Commented out DRGs are not in the 2021 DRG list.
"""

drgo = drg.DRG()

dict_2016 = {
    "P67D": 323939,
    "G67C": 210028,
    "F62B": 152796,
    "G24B": 150983,
    "I68D": 137892,
    #'E77I': 136960,
    "F71B": 136433,
    "F49G": 123319,
    "B80Z": 119473,
    "G67B": 116162,
    "V60B": 86597,
    "E71C": 85355,
    #'F73Z': 84490,
    "F58B": 84126,
    "L20C": 79504,
    "E63B": 79416,
    "E65C": 75815,
    #'J65Z': 75615,
    "J64B": 74866,
    "I47B": 71863,
    #'F67D': 70482,
    "F74Z": 69835,
    "G26B": 66994,
    "L64A": 64472,
    "F52B": 63471,
    "G60B": 59913,
    "L64C": 55470,
    "K62B": 54597,
    "H41D": 53855,
    "H08B": 53523,
    #'B76G':52727,
    "G72B": 51447,
    #'L63F':49843,
    "I29B": 49421,
    "D30B": 49222,
    "I44B": 49113,
    "F59D": 48023,
    "G71Z": 47312,
    #'D62Z':46826,
    "D06C": 45952,
    "H62B": 45221,
    "M02B": 44660,
    "T60E": 43243,
    "B71D": 42779,
    "X62Z": 42295,
    "G67A": 41734,
    "K60F": 41318,
    # 'D61B':41016,
    "L20B": 39279,
    #'I30Z' :37791
}

drg_list = drgo.get_DRG_list_details()


claim = 0
for d in dict_2016:
    deets = drg_list.loc[drg_list["DRG"] == d]
    if len(deets) > 0:
        claim += (
            drgo.calculate_claim(
                d, deets["Mittlere Verweildauer"].item(), "2021-05-01", "Bayern"
            )
            * dict_2016[d]
        )
    else:
        print("SHIT " + d)

print(claim)
print(sum(dict_2016.values()))
print(claim / sum(dict_2016.values()))
