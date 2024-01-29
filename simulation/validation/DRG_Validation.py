from scipy.stats import kendalltau, spearmanr

"""
This file is used to calculate the similarity between the generated DRGs and the validation set and the statistics.
"""


def jaccard_similarity(list1, list2):
    """Calculates the Jaccard similarity between two lists."""
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1).union(set(list2)))
    return intersection / union


def dice_coefficient(list1, list2):
    """Calculates the Dice coefficient between two lists."""
    intersection = len(set(list1).intersection(set(list2)))
    total = len(list1) + len(list2)
    return (2 * intersection) / total


generated = [
    "P67D",
    "P67E",
    "U63Z",
    "E79C",
    "I69B",
    "F71B",
    "G67C",
    "I68D",
    "G67B",
    "P67B",
    "H64Z",
    "B70F",
    "G71Z",
    "I77Z",
    "L64D",
    "F65B",
    "F62D",
    "D65Z",
    "G74Z",
    "O01F",
    "A07E",
    "E71D",
    "O01E",
    "L64B",
    "V60B",
    "F62C",
    "G60B",
    "G67A",
    "F58B",
    "O60C",
    "U66Z",
    "K62C",
    "801D",
    "O65B",
    "I74C",
    "B78A",
    "D61Z",
    "I75B",
    "J62B",
    "F67C",
    "F60B",
    "D63B",
    "F75C",
    "L62C",
    "N62A",
    "B81B",
    "G72B",
    "F73B",
    "B69D",
    "U64Z",
]

validation = [
    "O60D",
    "F62C",
    "G67C",
    "E79C",
    "G67B",
    "I68D",
    "F71B",
    "P67E",
    "B80Z",
    "F67C",
    "L63E",
    "K62C",
    "F49G",
    "I47C",
    "I44C",
    "O01F",
    "F73B",
    "H08C",
    "G24C",
    "D61Z",
    "O01E",
    "E71D",
    "L64B",
    "E65C",
    "G72B",
    "O60C",
    "V60B",
    "F49F",
    "F58B",
    "O65B",
    "E69C",
    "B76E",
    "F52B",
    "I10E",
    "I21Z",
    "L20C",
    "F74Z",
    "B77Z",
    "T64C",
    "E65A",
    "G26B",
    "G23B",
    "J64B",
    "G60B",
    "G71Z",
    "O65A",
    "L64D",
    "D63B",
    "B81B",
    "K60F",
]

statistics = [
    "P67E",
    "O60D",
    "F62C",
    "E79C",
    "F71B",
    "I68D",
    "B80Z",
    "F49G",
    "F67C",
    "I47C",
    "E69C",
    "G24C",
    "K62C",
    "H08C",
    "I44C",
    "E71D",
    "O01E",
    "O01F",
    "D61Z",
    "F58B",
    "F49F",
    "I21Z",
    "E65C",
    "O65B",
    "F52B",
    "O60C",
    "G25Z",
    "G23B",
    "B81B",
    "I10E",
    "L20B",
    "V60B",
    "O65A",
    "B77Z",
    "J64B",
    "I41Z",
    "D63B",
    "K60F",
]

statistics.extend([""] * 12)
print(len(statistics))

print(
    "Jaccard Similarity (Generated - Validation): "
    + str(jaccard_similarity(generated, validation))
)
print(
    "Jaccard Similarity (Generated - Statistics): "
    + str(jaccard_similarity(generated, statistics))
)
print(
    "Jaccard Similarity (Statistics - Validation): "
    + str(jaccard_similarity(validation, statistics))
)

print("Kendall Tau (Generated - Validation): " + str(kendalltau(generated, validation)))
print("Kendall Tau (Generated - Statistics): " + str(kendalltau(generated, statistics)))
print(
    "Kendall Tau (Statistics - Validation): " + str(kendalltau(validation, statistics))
)

print("Spearman R (Generated - Validation): " + str(spearmanr(generated, validation)))
print("Spearman R (Generated - Statistics): " + str(spearmanr(generated, statistics)))
print("Spearman R (Statistics - Validation): " + str(spearmanr(validation, statistics)))

print(
    "Dice Coefficient (Generated - Validation): "
    + str(dice_coefficient(generated, validation))
)
print(
    "Dice Coefficient (Generated - Statistics): "
    + str(dice_coefficient(generated, statistics))
)
print(
    "Dice Coefficient (Statistics - Validation): "
    + str(dice_coefficient(validation, statistics))
)
