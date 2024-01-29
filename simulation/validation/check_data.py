import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This file is used to calculate the statistics of the generated data.
"""

data = pd.read_csv("data/generated data/claims_final.csv")

anlage_5 = pd.read_csv("data/DRG/Anlage_5.csv", sep=";")
anlage_5["Betrag"] = anlage_5["Betrag"].str.replace(" €", "")
anlage_5["Betrag"] = anlage_5["Betrag"].str.replace(".", "")
anlage_5["Betrag"] = anlage_5["Betrag"].str.replace("siehe Anlage 2", "-1")
anlage_5["Betrag"] = anlage_5["Betrag"].str.replace(",", ".")  # .astype(float)
anlage_5["Betrag"] = pd.to_numeric(anlage_5["Betrag"], errors="coerce")

anlage_7 = pd.read_csv("data/DRG/Anlage_7.csv", sep=";")


def check_ZE(icd: str = None, ops: str = None):
    if icd is None and ops is None:
        return 0
    elif icd is not None:
        if icd in anlage_7["ICD-Kode"].values:
            return 1
    elif ops is not None:
        if ops in anlage_5["OPS-Kode"].values:
            return 1

    return 0


def check_row(row):
    if (
        check_ZE(row["PRIMARY_ICD"])
        or check_ZE(icd=row["ICD_1"])
        or check_ZE(icd=row["ICD_2"])
        or check_ZE(icd=row["ICD_3"])
        or check_ZE(icd=row["ICD_4"])
        or check_ZE(icd=row["ICD_5"])
        or check_ZE(icd=row["ICD_6"])
        or check_ZE(icd=row["ICD_7"])
        or check_ZE(icd=row["ICD_8"])
        or check_ZE(icd=row["ICD_9"])
        or check_ZE(icd=row["ICD_10"])
        or check_ZE(icd=row["ICD_11"])
        or check_ZE(icd=row["ICD_12"])
        or check_ZE(icd=row["ICD_13"])
        or check_ZE(icd=row["ICD_14"])
        or check_ZE(icd=row["ICD_15"])
        or check_ZE(icd=row["ICD_16"])
        or check_ZE(icd=row["ICD_17"])
        or check_ZE(icd=row["ICD_18"])
        or check_ZE(icd=row["ICD_19"])
        or check_ZE(icd=row["ICD_20"])
        or check_ZE(ops=row["OPS_1"])
        or check_ZE(ops=row["OPS_2"])
        or check_ZE(ops=row["OPS_3"])
        or check_ZE(ops=row["OPS_4"])
    ):
        return True
    else:
        return False


print("=== DEMOGRAPHIC INFORMATION ===")

print("Number of cases: " + str(len(data)))
print("Number of patients: " + str(data["PATIENT_ID"].nunique()))
print("Number of hospitals: " + str(data["HOSPITAL_ID"].nunique()))
print("Number of cases with fraud: " + str(data["POTENTIAL_FRAUD"].sum()))
print("Ratio of cases with fraud: " + str(data["POTENTIAL_FRAUD"].sum() / len(data)))

print("\nAverage age: " + str(data["AGE"].mean()))
print(
    "age >= 65: "
    + str(data[data["AGE"] >= 65].value_counts().value_counts().item() / len(data))
)  # goal: ca. 42% of population
print(
    "age < 65 and >= 45: "
    + str(
        data[(data["AGE"] < 65) & (data["AGE"] >= 45)]
        .value_counts()
        .value_counts()
        .item()
        / len(data)
    )
)  # goal: ca. 18 % of population
print(
    "age < 45 and >=15: "
    + str(
        data[(data["AGE"] < 45) & (data["AGE"] >= 15)]
        .value_counts()
        .value_counts()
        .item()
        / len(data)
    )
)  # goal: ca. 12 % of population
print(
    "age < 15: "
    + str(data[data["AGE"] < 15].value_counts().value_counts().item() / len(data))
)  # goal: ca. 14% of population

print("\nGender distribution:")
print(data["GENDER"].value_counts(normalize=True))


print("\n=== STAY DURATION and NUMBER OF STAYS ===")
print(
    "Average duration: "
    + str(data["DURATION"].mean())
    + "  Deviation from real world: "
    + str(round(7.2 - data["DURATION"].mean(), 2))
)  # should be around 7 days
duration = data[(data["DURATION"] >= 1) & (data["DURATION"] <= 3)].shape[0] / len(data)
print(
    "Percentage of stays between 1 and 3 days: "
    + str(duration)
    + "   Deviation from real world: "
    + str(round(0.4 - duration, 2))
)  # should be around 40 %
print("Longest stay: " + str(data["DURATION"].max().item()) + " days")
print("Standard deviation of duration: " + str(data["DURATION"].std().item()))
print("Number of cases per patient")
print(data["PATIENT_ID"].value_counts().value_counts())

# print number of FRAUD_ID
print("\n=== FRAUD TYPES ===")
print("Number of cases with fraud: " + str(data["POTENTIAL_FRAUD"].sum()))
print(
    "Number of cases with Increased Ventilation   :"
    + str(data[data["FRAUD_ID"] == 1].shape[0])
)
print(
    "Number of cases with Changed to Caesarean    :"
    + str(data[data["FRAUD_ID"] == 2].shape[0])
)
print(
    "Number of cases with Adjusted Weight         :"
    + str(data[data["FRAUD_ID"] == 3].shape[0])
)
print(
    "Number of cases with Additional Personal Care:"
    + str(data[data["FRAUD_ID"] == 4].shape[0])
)
print(
    "Number of cases with Bloody Release          :"
    + str(data[data["FRAUD_ID"] == 5].shape[0])
)
print(
    "Number of cases with Changed ICD Order       :"
    + str(data[data["FRAUD_ID"] == 6].shape[0])
)


print("\n=== OVERLAPPING CASES ===")
data = data.sort_values(by=["PATIENT_ID", "ADMISSION_DATE"])
data["Overlap"] = (
    data["ADMISSION_DATE"] < data.groupby("PATIENT_ID")["DISCHARGE_DATE"].shift(1)
).astype(int)
print(
    "Number of cases overlapping at one patient: " + str(data["Overlap"].sum())
)  # TODO: DELETE AND REPLACE or simply delete?
print("Ratio of overlapping cases: " + str(data["Overlap"].sum() / len(data)))

print("\n=== ICD, OPS, Ventilation ===")
print("Most common primary ICDs:")
print(data["PRIMARY_ICD"].value_counts())

print("Number of cases with ventilation hours: ")

bins = [
    1,
    24,
    48,
    59,
    72,
    95,
    120,
    179,
    180,
    240,
    249,
    263,
    275,
    320,
    479,
    480,
    499,
    599,
    899,
    999,
    1799,
    np.inf,
]
print(pd.cut(data["VENTILATION"], bins=bins).value_counts().sort_index())
print(
    "Ratio of ventilation cases: "
    + str(data[data["VENTILATION"] > 0].shape[0] / len(data))
    + " (should be around 0.02)"
)
"""plt.subplot(2, 2, 3)
plt.title("Ventilation hours")
#plt.hist(data["VENTILATION"], bins=bins)
plt.bar(data["VENTILATION"].unique(), data["VENTILATION"].value_counts())"""


print("\n=== DRG ===")
print("Distribution of DRG codes: " + str(data["DRG"].value_counts()))
print("50 most common DRG codes: " + str(data["DRG"].value_counts().head(50)))

print("Number of cases with fraud: " + str(data["POTENTIAL_FRAUD"].sum()))

print("Total claim: " + str(round(np.sum(data["CLAIM"]), 2)))
print(
    "Average claim: " + str(round(data[data["CLAIM"] != 0]["CLAIM"].mean(), 2))
)  # should be around 5.000 € according to https://www.destatis.de/DE/Presse/Pressemitteilungen/2021/04/PD21_194_231.html
print(
    "Standard deviation of claims: "
    + str(round(data[data["CLAIM"] != 0]["CLAIM"].std(), 2))
)
print(
    "Number of cases with individual contracts: "
    + str(data[data["CLAIM"] == 27089.18].shape[0])
)
print(
    "Number of cases with individual contracts and fraud: "
    + str(data[(data["CLAIM"] == 27089.18) & (data["POTENTIAL_FRAUD"] == 1)].shape[0])
)

print(
    "Average claim withouth individual contracts: "
    + str(round(data[data["CLAIM"] != 27089.18]["CLAIM"].mean(), 2))
)

data["EXTRA"] = data.apply(lambda x: check_row(x), axis=1)
ex = data["EXTRA"].sum()
print("Potential applicability of extra charges: " + str(ex))
print(
    "Average claim without fraud: "
    + str(round(data[data["POTENTIAL_FRAUD"] == 0]["CLAIM"].mean(), 2))
)
print(
    "Average claim with fraud: "
    + str(round(data[data["POTENTIAL_FRAUD"] == 1]["CLAIM"].mean(), 2))
)
print(
    "Average claim with average extra charges: "
    + str((ex * 696.69 + round(np.sum(data["CLAIM"]), 2)) / len(data))
)
print(
    "Average claim without fraud: "
    + str(round(data[data["POTENTIAL_FRAUD"] == 0]["CLAIM"].mean(), 2))
)
print(
    "Average claim with fraud: "
    + str(round(data[data["POTENTIAL_FRAUD"] == 1]["CLAIM"].mean(), 2))
)

print("Number of bad groupings: " + str(data[data["CLAIM"] == 0].shape[0]))
print(
    "Number of bad groupings with fraud: "
    + str(str(data[data["CLAIM"] == 0]["POTENTIAL_FRAUD"].sum()))
)
print(
    "Percentage of bad groupings: " + str(data[data["CLAIM"] == 0].shape[0] / len(data))
)
print("Average age of bad groupings: " + str(data[data["CLAIM"] == 0]["AGE"].mean()))
print(
    "Most common primary ICDs in bad groupings: "
    + str(data[data["CLAIM"] == 0]["PRIMARY_ICD"].value_counts())
    + "\n"
)
print(
    "DRG of bad groupings: "
    + str(data[data["CLAIM"] == 0]["DRG"].value_counts())
    + "\n"
)


"""print("\n=== Z11-diagnostics ===")
print("Ratio of Z11 diagnoses: " + str(data[data["PRIMARY_ICD"] == "Z11"].shape[0]/len(data)))
print("Ratio of gender in Z11:" + str(data[data["PRIMARY_ICD"] == "Z11"]["GENDER"].value_counts(normalize=True))) 
print("Average age of Z11: " + str(data[data["PRIMARY_ICD"] == "Z11"]["AGE"].mean()))
print("Ratio of age groups in Z11: " + str(data[data["PRIMARY_ICD"] == "Z11"]["AGE"].value_counts(normalize=True)))
print("Ratio of Z11 with fraud: " + str(data[data["PRIMARY_ICD"] == "Z11"]["POTENTIAL_FRAUD"].sum()/data[data["PRIMARY_ICD"] == "Z11"].shape[0]))
print("Number of cases with fraudulent Z11 primary ICD " + str(data[data["PRIMARY_ICD"] == "Z11"]["POTENTIAL_FRAUD"].sum()))
print("Ratio of Z11 cases with fraud: " + str(data[data["PRIMARY_ICD"] == "Z11"]["POTENTIAL_FRAUD"].sum()/data[data["POTENTIAL_FRAUD"] == 1].shape[0]))
print("")
print("Number of cases with Z-DRG: " + str(data[data["DRG"].str.startswith("Z")].shape[0]))
print("ICD and OPS codes of Z-DRG cases: " + str(data[data["DRG"].str.startswith("Z")]["PRIMARY_ICD"].value_counts()) + "\n")
print("DRGs of Z11 primary ICD: " + str(data[data["PRIMARY_ICD"] == "Z11"]["DRG"].value_counts())+"\n")
print("Average claim without Z-DRG: " + str(round(data[data["DRG"].str.startswith("Z") == False]["CLAIM"].mean(), 2)))
print("Average claim of Z-DRG: " + str(round(data[data["DRG"].str.startswith("Z")]["CLAIM"].mean(), 2)))
print("Average claim of Z64C DRG: " + str(round(data[data["DRG"] == "Z64C"]["CLAIM"].mean(), 2)))
print("Average duration of stay of Z64C DRG: " + str(round(data[data["DRG"] == "Z64C"]["DURATION"].mean(), 2)))
print("Ratio of Z64C DRG on fraud: "+ str(data[data["DRG"] == "Z64C"]["POTENTIAL_FRAUD"].sum()/data[data["DRG"] == "Z64C"].shape[0]))
print("Ratio of Z64C DRG on fraud: "+ str(data[data["DRG"] == "Z64C"]["POTENTIAL_FRAUD"].sum()/data[data["POTENTIAL_FRAUD"] == 1].shape[0]))"""

"""plt.subplot(2, 2, 1)
plt.bar(data["DURATION"].unique(), data["DURATION"].value_counts())
plt.title("Length of Stay")
"""
plt.subplot(2, 2, 1)
plt.title("Average claim per hospital")
data1 = data.dropna(subset=["HOSPITAL_ID"])
data1 = data1[
    data1["HOSPITAL_ID"].isin(data1["HOSPITAL_ID"].value_counts().head(300).index)
]
data1 = data1.groupby("HOSPITAL_ID")["CLAIM"].mean().to_frame()
data1 = data1.sort_values(by="CLAIM", ascending=False)
plt.bar(data1.index, data1["CLAIM"])
# print(temp["WEIGHT"])
plt.subplot(2, 2, 2)
temp = data[data["WEIGHT"] != "n/A"]
sns.kdeplot(temp["WEIGHT"].values.astype(float))
plt.title("Weight Distribution (KDE)")
plt.xlabel("Weight (g)")

# plot the number of cases per hospital in a bar chart sorted descending
plt.subplot(2, 2, 3)
counts = data["HOSPITAL_ID"].value_counts().sort_values(ascending=False)
plt.bar(counts.index, counts.values)
plt.title("Number of Cases per Hospital")
plt.xlabel("Hospital ID")


plt.subplot(2, 2, 4)
sns.kdeplot(temp[temp["POTENTIAL_FRAUD"] == 0]["WEIGHT"].values.astype(float))
plt.title("Weight Distribution (KDE) - No Fraud")
plt.xlabel("Weight (g)")

plt.show()
