import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate

"""
This file is used to plot the density of weight for all cases and potential fraud cases.
"""

claims = pd.read_csv("data/generated data/claims_final.csv")

claims = claims[claims["WEIGHT"] != "n/A"]
claims["WEIGHT"] = pd.to_numeric(claims["WEIGHT"], errors="coerce")

# get number of cases per weight where POTENTIAL_FRAUD = 1
weight_f = claims[claims["POTENTIAL_FRAUD"] == 1]["WEIGHT"].value_counts()
# sort by index
weight_f = weight_f.sort_index()
print(weight_f)
print(weight_f[2918])
weight_c = claims[claims["POTENTIAL_FRAUD"] == 0]["WEIGHT"].value_counts()
# sort by index
weight_c = weight_c.sort_index()

weight_all = claims["WEIGHT"].value_counts()
# sort by index
weight_all = weight_all.sort_index()
# plot both graphs in one figure
plt.figure(figsize=(10, 6))

# draw vertical lines at 2499 and 1999
plt.axvline(x=2498, color="black", linestyle="--", label="2499 g Threshold")
plt.axvline(x=1998, color="blue", linestyle="--", label="1999 g Threshold")
plt.axvline(x=1498, color="green", linestyle="--", label="1499 g Threshold")
# sns.kdeplot(weight_all,  color='blue', label='All cases', linewidth=3, bw_adjust=1)
# sns.kdeplot(weight_f, linestyle="-.",color='red', label='Fraudulent cases', linewidth=1.5, bw_adjust=1)
# sns.kdeplot(weight_c,linestyle=":", color='green', label='Non-fraudulent cases', linewidth=3,bw_adjust=.02)
# plt.plot(weight_all.index, weight_all.values, label='All cases', color='blue', alpha=0.5)
# plt.plot(weight_f.index, weight_f.values, label='Fraudulent cases', color='red', alpha=0.5)
# sns.kdeplot(weight_all, label='All cases', color='blue', alpha=0.5)
# sns.kdeplot(weight_f, label='Fraudulent cases', color='red', alpha=0.5)

df_f = pd.DataFrame({"Weight": weight_f.index, "Density": weight_f.values})
df_c = pd.DataFrame({"Weight": weight_c.index, "Density": weight_c.values})
df_all = pd.DataFrame({"Weight": weight_all.index, "Density": weight_all.values})

# Plotting the kernel density estimators
sns.kdeplot(
    data=df_f,
    x="Weight",
    weights="Density",
    color="red",
    label="Fraud Cases",
    linestyle=":",
    linewidth=3,
    bw_adjust=0.02,
)
"""sns.kdeplot(
    data=df_c,
    x="Weight",
    weights="Density",
    color="blue",
    label="Non-Fraud Cases",
    linestyle=":",
    linewidth=3,
    bw_adjust=0.02,
)"""
sns.kdeplot(
    data=df_all,
    x="Weight",
    weights="Density",
    color="green",
    label="All Cases",
    linewidth=3,
    bw_adjust=0.02,
)

plt.xlabel("Weight", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.title('Density of weight for all cases and potential fraud cases')
plt.rcParams.update({"font.size": 20})
plt.legend()
plt.show()
