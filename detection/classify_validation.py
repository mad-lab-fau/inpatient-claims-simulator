import joblib
import pandas as pd
import numpy as np

"""
This file is used to classify the validation set and calculate the fraud recall, precision and f1 score.
"""

clf = joblib.load("data/models/dt.pkl") # load model
validation = pd.read_csv("data/generated data/preprocessing/validation.csv") # load validation set
validation = validation.drop(columns=["POTENTIAL_FRAUD"]) # "RATIO_SECTIO" drop columns that are not needed for classification; drop "RATIO_SECTIO" only if model is trained without ratio
fraud_ids = pd.read_csv("data/generated data/preprocessing/ID_Fraud_Mapping.csv") # load fraud ids if needed

result = clf.predict(validation) # predict validation set

fraud_ids = fraud_ids[fraud_ids["ID"].isin(validation["ID"])] #keep only fraud_ids where ID is in validation set
#print(len(fraud_ids))

fraud_ids["FRAUD"] = result # add column with predicted fraud

#print(len(fraud_ids[fraud_ids["FRAUD_ID"] == 3]))
#print(fraud_ids[fraud_ids["FRAUD_ID"] == 3]["FRAUD"].sum())
# for each fraud id get ratio of FRAUD = 1
fraud_recall = fraud_ids.groupby("FRAUD_ID").mean()

print("\nFraud Recall")
print(fraud_recall)

#f = len(fraud_ids[fraud_ids["FRAUD_ID"] != 0])

fraud_precision = fraud_ids.groupby("FRAUD_ID").sum()/sum(result) #f 
print("\nFraud Precision")
print(fraud_precision)

fraud_f1 = 2 * (fraud_precision * fraud_recall) / (fraud_precision + fraud_recall)
fraud_f1 = fraud_f1.fillna(0)
print("\nFraud F1")
print(fraud_f1)
#print(fraud_ids["FRAUD"]/sum(result))