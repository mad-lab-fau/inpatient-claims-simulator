import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

"""
This file is used to plot the the number of claims for each ventilation type and mark the fraudulent claims red.
"""

claims = pd.read_csv("data/generated data/claims_final.csv")

# keep only claims where VENTILATION > 0
claims = claims[claims['VENTILATION'] > 0]

# keep only claims where VENTILATION > 39 and <71
claims = claims[claims['VENTILATION'] < 71]
claims = claims[claims['VENTILATION'] > 39]

# VENTILATION where POTENTIAL_FRAUD = 1
vent_f = claims[claims['POTENTIAL_FRAUD'] == 1]['VENTILATION'].value_counts()


# VENTILATION where POTENTIAL_FRAUD = 0
vent_nf = claims[claims['POTENTIAL_FRAUD'] == 0]['VENTILATION'].value_counts()

# fill missing values from vent_nf in vent_f with 0
for index in vent_nf.index:
    if index not in vent_f.index:
        vent_f[index] = 0

vent_nf = vent_nf.sort_index()
vent_f = vent_f.sort_index()

# fill missing values from vent_f in vent_nf with 0
for index in vent_f.index:
    if index not in vent_nf.index:
        vent_nf[index] = 0


# plot a bar chart summing up the number of claims for each ventilation type, adding vent_f ontop of vent_nf
plt.bar(vent_nf.index, vent_nf.values)
plt.bar(vent_f.index, vent_f.values, bottom=vent_nf.values, color='red')

# draw horizontal line at y=50
plt.axhline(y=50, color='black', linewidth=2)

# add legend
green_patch = mpatches.Patch(label='Non-Fraudulent Claims')
red_patch = mpatches.Patch(color='red', label='Fraudulent Claims')
plt.legend(handles=[green_patch, red_patch])

# add labels
plt.xlabel('Ventilation Hours', fontsize=20)
plt.ylabel('Number of Claims', fontsize=20)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()