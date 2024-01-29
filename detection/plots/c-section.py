import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

"""
This file is used to plot the the ratio of c-sections to vaginal deliveries per hospital and mark the fraudulent hospitals red.
"""

claims = pd.read_csv("data/generated data/claims_final.csv")
fraudulent_hospitals = [i for i in range(0, 300, 3)]

# get ratio of "O82" to "O80" in column "PRIMARY_ICD" per hospital
claims["O82"] = claims["PRIMARY_ICD"].apply(lambda x: 1 if x == "O82" else 0)
claims["O80"] = claims["PRIMARY_ICD"].apply(lambda x: 1 if x == "O80" else 0)
claims = claims.groupby("HOSPITAL_ID").sum()
claims["RATIO"] = claims["O82"] / claims["O80"]

# reset the index to make "HOSPITAL_ID" a column again
claims = claims.reset_index()

# sort the DataFrame by 'RATIO'
claims_sorted = claims.sort_values(by='RATIO', ascending=False)

# create a new variable for the order of hospitals
claims_sorted['HOSPITAL_ORDER'] = range(1, len(claims_sorted) + 1)

# change the color of the fraudulent hospitals
claims_sorted['COLOR'] = claims_sorted['HOSPITAL_ID'].apply(lambda x: 'red' if x in fraudulent_hospitals else 'blue')

# plot using 'HOSPITAL_ORDER' for x-axis
plt.bar(claims_sorted['HOSPITAL_ORDER'], claims_sorted['RATIO'], color=claims_sorted['COLOR'])

# add a horizontal line at 0.4
plt.axhline(y=0.4, color='black', linestyle='-')

threshold_patch = mpatches.Patch(color='black', label='Threshold')
fraud_patch = mpatches.Patch(color='red', label='Fraudulent hospital')
non_fraud_patch = mpatches.Patch(color='blue', label='Non-fraudulent hospital')

# add legend with custom patches
plt.legend(handles=[threshold_patch, fraud_patch, non_fraud_patch], fontsize=20)

# add labels
plt.xlabel('Hospital', fontsize=20)
plt.ylabel('Ratio of sectio cesarean to vaginal delivery', fontsize=20)
# remove the ticks on the x-axis
plt.xticks([])
plt.yticks(fontsize=18)
# change font size for labels
plt.rcParams.update({'font.size': 20})

plt.show()

