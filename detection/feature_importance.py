import joblib
import pandas as pd
import numpy as np

"""
This file is used to get the feature importance of the selected model.
"""

clf = joblib.load("data/models/rf_cat.pkl") # load model
validation = pd.read_csv("data/generated data/preprocessing/validation.csv") # load validation set
validation = validation.drop(columns=["POTENTIAL_FRAUD"]) # drop columns that are not needed for classification # "RATIO_SECTIO"
feature_names = list(validation.columns) # get feature names from validation set

importances = clf._final_estimator.feature_importances_ # get feature importances from model
indices = np.argsort(importances)[::-1] # sort feature importances in descending order

# Print each of the 10 most important features
for i in range(10):
    index = indices[i]
    print(f"{feature_names[index]}: {importances[index]}")