import json
import sys
from matplotlib import pyplot as plt

import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import sys

sys.path.insert(0, "")
from simulation.generation.ops import OPS
from simulation.generation.drg import DRG
import simulation.utility.utils as utils

sys.path.insert(0, "")

import pandas as pd
import seaborn as sns

import simulation.utility.utils as utils

"""This file is used to check the OPS distribution of the generated data and the validation set."""


def check_ops():
    data = pd.read_csv("data/generated data/claims.csv")

    ops_columns = [
        "OPS_1",
        "OPS_2",
        "OPS_3",
        "OPS_4",
        "OPS_5",
        "OPS_6",
        "OPS_7",
        "OPS_8",
        "OPS_9",
        "OPS_10",
        "OPS_11",
        "OPS_12",
        "OPS_13",
        "OPS_14",
        "OPS_15",
        "OPS_16",
        "OPS_17",
        "OPS_18",
        "OPS_19",
        "OPS_20",
    ]

    data = data[ops_columns]
    data.replace("n/A", "", inplace=True)
    data = data.applymap(lambda x: x[:5])
    ops = []

    for c in ops_columns:
        ops.extend(data[c].unique())

    ops = list(set(ops))  # remove duplicates

    dist = data.stack().value_counts()  # count occurences of each OPS
    dist = dist[dist.index != ""]  # remove empty OPS
    dist = dist.rename("COUNT")
    dist1 = dist[dist.index.str.startswith("1-")].sum() / len(data)
    dist3 = dist[dist.index.str.startswith("3-")].sum() / len(data)
    dist5 = dist[dist.index.str.startswith("5-")].sum() / len(data)
    dist8 = dist[dist.index.str.startswith("8-")].sum() / len(data)
    dist9 = dist[dist.index.str.startswith("9-")].sum() / len(data)
    print("1-OPS in data set: {}".format(dist1))
    print("3-OPS in data set: {}".format(dist3))
    print("5-OPS in data set: {}".format(dist5))
    print("8-OPS in data set: {}".format(dist8))
    print("9-OPS in data set: {}".format(dist9))
    d = dist.sum() / len(data)  # average number of OPS per claim
    dist = dist / len(data)  # normalize distribution

    original = pd.read_csv("data/OPS/generated/ops_probs_2.csv")
    ages = utils.get_age_list()
    genders = utils.get_gender_list()
    groups = []
    for a in ages:
        for g in genders:
            group = g + "_" + a
            groups.extend([group])

    # drop original["Insgesamt_Insgesamt"]
    original = original.drop(columns=["Insgesamt_Insgesamt"])
    original = original[
        original["OPS_Code_ "].str.len() >= 5
    ]  # remove OPS with less than 5 characters
    original["AVG"] = original[groups].mean(axis=1)
    original = original.iloc[:-2]  # remove last two rows

    relevant = original[["OPS_Code_ ", "AVG"]]  # remove unnecessary columns

    original1 = original[original["OPS_Code_ "].str.startswith("1-")]["AVG"].sum()
    original3 = original[original["OPS_Code_ "].str.startswith("3-")]["AVG"].sum()
    original5 = original[original["OPS_Code_ "].str.startswith("5-")]["AVG"].sum()
    original8 = original[original["OPS_Code_ "].str.startswith("8-")]["AVG"].sum()
    original9 = original[original["OPS_Code_ "].str.startswith("9-")]["AVG"].sum()

    print("1-OPS in original distribution: {}".format(original1))
    print("3-OPS in original distribution: {}".format(original3))
    print("5-OPS in original distribution: {}".format(original5))
    print("8-OPS in original distribution: {}".format(original8))
    print("9-OPS in original distribution: {}".format(original9))

    print("1-OPS difference: {}".format(original1 - dist1))
    print("3-OPS difference: {}".format(original3 - dist3))
    print("5-OPS difference: {}".format(original5 - dist5))
    print("8-OPS difference: {}".format(original8 - dist8))
    print("9-OPS difference: {}".format(original9 - dist9))

    print(
        "Total deviation: {}".format(
            (original1 - dist1)
            + (original3 - dist3)
            + (original5 - dist5)
            + (original8 - dist8)
            + (original9 - dist9)
        )
    )

    new = relevant.join(
        dist, on="OPS_Code_ ", how="left"
    )  # join original and new distribution
    new["COUNT"] = new["COUNT"].fillna(0)  # replace NaN with 0
    new["DIFF"] = (
        new["AVG"] - new["COUNT"]
    )  # calculate difference between original and new distribution
    print(
        "Difference 1-OPS: {}".format(
            new[new["OPS_Code_ "].str.startswith("1-")]["DIFF"].sum()
        )
    )
    print(
        "Difference 3-OPS: {}".format(
            new[new["OPS_Code_ "].str.startswith("3-")]["DIFF"].sum()
        )
    )
    print(
        "Difference 5-OPS: {}".format(
            new[new["OPS_Code_ "].str.startswith("5-")]["DIFF"].sum()
        )
    )
    print(
        "Difference 8-OPS: {}".format(
            new[new["OPS_Code_ "].str.startswith("8-")]["DIFF"].sum()
        )
    )
    print(
        "Difference 9-OPS: {}".format(
            new[new["OPS_Code_ "].str.startswith("9-")]["DIFF"].sum()
        )
    )
    print("Total difference: {}".format(new["DIFF"].sum()))

    h = new[["OPS_Code_ ", "DIFF"]]
    # h = h[h["OPS_Code_ "].str.startswith("3-")]

    sorted = h.sort_values(by=["DIFF"], ascending=True)
    print(sorted)
    print(
        "Cumulated Error 5er OPS: {}".format(
            sorted[sorted["OPS_Code_ "].str.startswith("5-")]["DIFF"].sum()
        )
    )
    print("Cumulated Error: {}".format(sorted["DIFF"].sum()))
    print(
        "Average number of OPS per claim: {}".format(d)
    )  # average number of OPS per claim
    print(
        "Average # of OPS per case in original data distribution: {}".format(
            original["AVG"].sum()
        )
    )
    """piv = pd.pivot_table(h, values='DIFF', index=['OPS_Code_ '])
    ax = sns.heatmap(piv, annot=True, square=True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()"""

    # h.plot(x="OPS_Code_ ", y="DIFF", kind="line", rot=90, figsize=(20, 10))
    # plot OPS difference for all OPS
    """plt.plot(h["OPS_Code_ "], h["DIFF"], marker="o", linestyle="None")
    labels = h["OPS_Code_ "].tolist()[14::15]
    plt.xticks(ticks=labels, rotation=90)
    plt.yticks([-0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
    plt.show()

    # plot OPS difference for chapters
    h["CHAPTER"] = h["OPS_Code_ "].str[:1]
    h = h.groupby("CHAPTER").sum()
    h.plot(y="DIFF", kind="bar", rot=0, figsize=(20, 10))
    plt.show()"""

    di = h.set_index("OPS_Code_ ")["DIFF"].to_dict()

    with open("data/OPS/generated/ops_diff.json", "r") as f:
        js = json.load(f)
    for key, value in di.items():
        if key in js:
            js[key] += value
        else:
            js[key] = value
    with open("data/OPS/generated/ops_diff.json", "w") as f:
        json.dump(js, f, indent=4)

    """with open ("data/OPS/generated/ops_diff.json", "w") as f:
        json.dump(di, f, indent=4)"""


ops = OPS(DRG())

ages = utils.get_age_list()
ages = ["Insgesamt_" + x for x in ages]
# print the INS row for all columns in ages

# print(number_of_claims[number_of_claims[ages]["OPS_Code_ "] == "INS"]["INS"])

stats = pd.read_csv("data/OPS/generated/ops_probs_2.csv")
stats.fillna(0, inplace=True)

data = pd.read_csv("data/generated data/claims.csv")

# get age distribution in data
ag = data["AGE"].value_counts().to_dict()
# print (ag)
age = {}
a = utils.get_age_list()
for i in ag.keys():
    g = utils.get_age_group(i)
    if g in age.keys():
        age[g] += ag[i]
    else:
        age[g] = ag[i]

o = ops.get_ops_stat_list()[:-2]
o = [o for o in o if len(o) == 5]

print(stats.loc[stats["OPS_Code_ "] == "1-20A", "Insgesamt_unter 1 Jahr"])

SOLL = {}
for i in age.keys():
    for j in o:
        try:
            value = stats.loc[stats["OPS_Code_ "] == j, "Insgesamt_" + i].values[0]
        except IndexError:
            value = 0
        if j in SOLL.keys():
            SOLL[j] += value * age[i]
        else:
            SOLL[j] = value * age[i]

SOLL = {k: v / (len(data)) for k, v in SOLL.items()}
SOLL = pd.DataFrame.from_dict(SOLL, orient="index")
print(SOLL)

ops_columns = [
    "OPS_1",
    "OPS_2",
    "OPS_3",
    "OPS_4",
    "OPS_5",
    "OPS_6",
    "OPS_7",
    "OPS_8",
    "OPS_9",
    "OPS_10",
    "OPS_11",
    "OPS_12",
    "OPS_13",
    "OPS_14",
    "OPS_15",
    "OPS_16",
    "OPS_17",
    "OPS_18",
    "OPS_19",
    "OPS_20",
]


data = data[ops_columns]
data.replace("n/A", "", inplace=True)
data = data.applymap(lambda x: x[:5])
ops = []

for c in ops_columns:
    ops.extend(data[c].unique())

ops = list(set(ops))  # remove duplicates

dist = data.stack().value_counts()  # count occurences of each OPS
dist = dist[dist.index != ""]  # remove empty OPS
dist = dist.rename("COUNT")
dist = dist / len(data)
print(dist)

merged = pd.merge(SOLL, dist, left_index=True, right_index=True)
merged.columns = ["SOLL", "IST"]
merged["DIFF"] = merged["SOLL"] - merged["IST"]
print(merged)

diff1 = merged.loc[merged.index.str.startswith("1-"), "DIFF"].sum()
diff3 = merged.loc[merged.index.str.startswith("3-"), "DIFF"].sum()
diff5 = merged.loc[merged.index.str.startswith("5-"), "DIFF"].sum()
diff6 = merged.loc[merged.index.str.startswith("6-"), "DIFF"].sum()
diff8 = merged.loc[merged.index.str.startswith("8-"), "DIFF"].sum()
diff9 = merged.loc[merged.index.str.startswith("9-"), "DIFF"].sum()

print(
    "1-OPS difference: "
    + str(merged.loc[merged.index.str.startswith("1-"), "DIFF"].sum())
)
print(
    "3-OPS difference: "
    + str(merged.loc[merged.index.str.startswith("3-"), "DIFF"].sum())
)
print(
    "5-OPS difference: "
    + str(merged.loc[merged.index.str.startswith("5-"), "DIFF"].sum())
)
print(
    "6-OPS difference: "
    + str(merged.loc[merged.index.str.startswith("6-"), "DIFF"].sum())
)
print(
    "8-OPS difference: "
    + str(merged.loc[merged.index.str.startswith("8-"), "DIFF"].sum())
)
print(
    "9-OPS difference: "
    + str(merged.loc[merged.index.str.startswith("9-"), "DIFF"].sum())
)

# plot diff1 - 9 in one bar plot
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(
    x - width / 2, [diff1, diff3, diff5, diff6, diff8, diff9], width, label="Difference"
)

ax.set_ylabel("Difference", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(["1", "3", "5", "6", "8", "9"], fontsize=14)
ax.set_xlabel("OPS Chapter", fontsize=16)
# ax.legend()

# minimize space between bars

# change fontsize for y ticks
plt.yticks(fontsize=14)

fig.tight_layout()

plt.show()

# check_ops()
