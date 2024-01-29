import pandas as pd
import numpy as np
import warnings
import utils
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import sys

sys.path.insert(0, "")
import py.simulation.generation.drg as drg

"""
This file is used to preprocess the claims_final.csv file. It splits the ICD and OPS codes into Chapter, Major and Minor codes.
It also creates new features dependent on hospitals.

Outputs are:
- preprocessed.csv: training set
- validation.csv: validation set
- ID_Fraud_Mapping.csv: mapping of ID and FRAUD_ID for easier debugging
"""

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

claims = pd.read_csv("data/generated data/claims_final.csv")  # path to claims

# defining column names for ICD and OPS codes
icd_columns = [
    "PRIMARY_ICD",
    "ICD_1",
    "ICD_2",
    "ICD_3",
    "ICD_4",
    "ICD_5",
    "ICD_6",
    "ICD_7",
    "ICD_8",
    "ICD_9",
    "ICD_10",
    "ICD_11",
    "ICD_12",
    "ICD_13",
    "ICD_14",
    "ICD_15",
    "ICD_16",
    "ICD_17",
    "ICD_18",
    "ICD_19",
    "ICD_20",
]

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


def split_icd(column):
    """Splitting ICD codes into Chapter, Major and Minor codes.

    Args:
        column (str): column name of ICD code"""
    claims[column] = claims[column].astype(str)
    claims[[f"{column}_Chapter", f"{column}_Major", f"{column}_Minor"]] = claims[
        column
    ].str.extract(r"([A-Z])(\d\d)\.?(\d+)?")
    # claims.drop(column, axis=1, inplace=True)


def split_ops(column):
    """Splitting OPS codes into Chapter, Major, Minor, and Localization codes.

    Args:
        column (str): column name of OPS code"""
    claims[column] = claims[column].astype(str)
    claims[
        [
            f"{column}_Chapter",
            f"{column}_Major",
            f"{column}_Minor",
            f"{column}_Localization",
        ]
    ] = claims[column].str.extract(r"(\d)-(\d{1,3})(?:\.(\d{1,2}))?(?:-(\w{1,2}))?")
    # claims.drop(column, axis=1, inplace=True)


# defining new features dependent on hospitals
claims["AVG_VENTILATION"] = claims.groupby("HOSPITAL_ID")["VENTILATION"].transform(
    "mean"
)
claims["RATIO_SECTIO"] = claims.groupby("HOSPITAL_ID")["PRIMARY_ICD"].transform(
    lambda x: (x == "O82").sum() / (x == "O80").sum()
)
claims["AVG_CLAIM"] = claims.groupby("HOSPITAL_ID")["CLAIM"].transform("mean")
claims["WEIGHT"] = pd.to_numeric(claims["WEIGHT"], errors="coerce")
claims["AVG_WEIGHT"] = claims.groupby("HOSPITAL_ID")["WEIGHT"].transform("mean")
claims["N_CASES"] = claims.groupby("HOSPITAL_ID")["CLAIM"].transform("count")
claims["AVG_DISTANCE"] = claims.groupby("HOSPITAL_ID")["DISTANCE"].transform("mean")


# drop all rows with XXX as PRIMARY_ICD
claims.drop(claims[claims["PRIMARY_ICD"] == "XXX"].index, inplace=True)

# execute split_icd for all ICD columns
for column in icd_columns:
    split_icd(column)

# initialize and fit encoders for ICD codes Chapter + Major
le_icd_major = OrdinalEncoder()
le_icd_major.fit(utils.get_ICD_with_Major().reshape(-1, 1))

# initialize and fit encoders for ICD codes Chapter + Major + Minor
le_icd_minor = OrdinalEncoder()
le_icd_minor.fit(utils.get_ICD_with_Minor().reshape(-1, 1))

# apply one hot encoding for ICD codes Chapter
for column in icd_columns:
    c_column = column + "_Chapter"
    onehot_icd = OneHotEncoder(handle_unknown="ignore")
    # le_df = pd.DataFrame(le_icd.fit_transform(claims[[c_column]]).toarray())
    alphabet = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    alphabet_2d = np.array(alphabet).reshape(-1, 1)  # reshape alphabet to 2d array
    onehot_icd.fit(alphabet_2d)  # fit one hot encoder to alphabet
    column_2d = claims[c_column].values.reshape(-1, 1)  # reshape column to 2d array
    encoded_icd = onehot_icd.transform(
        column_2d
    )  # transform column with fitted one hot encoder
    encoded_icd_dense = encoded_icd.toarray()  # convert sparse matrix to dense matrix
    encoded_icd_df = pd.DataFrame(
        encoded_icd_dense, columns=onehot_icd.categories_[0]
    )  # convert dense matrix to dataframe
    encoded_icd_df.columns = [
        column + "_" + col for col in encoded_icd_df.columns
    ]  # rename columns
    claims = pd.concat(
        [claims, encoded_icd_df], axis=1
    )  # concat claims and encoded_icd_df

    ma_column = column + "_Major"  # add _Major to column name
    claims[ma_column] = claims[c_column] + claims[ma_column].astype(
        str
    )  # concat Chapter and Major to one column
    claims[ma_column].fillna("n/A", inplace=True)  # fill NaN values with n/A
    claims[ma_column] = le_icd_major.transform(
        claims[ma_column].values.reshape(-1, 1)
    )  # transform column with fitted encoder

    mi_column = column + "_Minor"  # add _Minor to column name

    claims[column + "_other"] = claims[mi_column].apply(
        lambda x: 1 if str(x).startswith("8") else 0
    )  # create new column with 1 if Minor code starts with 8 (indicating an ICD code not specified by the other codes)
    claims[column + "_undefined"] = claims[mi_column].apply(
        lambda x: 1 if str(x).startswith("9") else 0
    )  # create new column with 1 if Minor code starts with 9 (indicating an undefined ICD code)
    claims[mi_column] = claims[
        column
    ]  # use the initial columns as columns with Chapter, Major and Minor
    claims[mi_column].fillna("n/A", inplace=True)  # fill NaN values with n/A
    claims[mi_column] = le_icd_minor.transform(
        claims[mi_column].values.reshape(-1, 1)
    )  # transform column with fitted encoder

    claims.drop(
        [c_column, column], axis=1, inplace=True
    )  # drop Chapter (now one hot encoded) and initial column (now named with _Minor)

# execute split_ops for all OPS columns
for column in ops_columns:
    split_ops(column)

for column in ops_columns:
    c_column = column + "_Chapter"  # add _Chapter to column name
    le_ops = OneHotEncoder(handle_unknown="ignore")  # initialize one hot encoder
    alphabet = ["1", "3", "5", "6", "8", "9"]  # define valid alphabet for OPS chapters
    alphabet_2d = np.array(alphabet).reshape(-1, 1)  # reshape alphabet to 2d array
    le_ops.fit(alphabet_2d)  # fit one hot encoder to alphabet
    column_2d = claims[c_column].values.reshape(-1, 1)  # reshape column to 2d array
    encoded_ops = le_ops.transform(
        column_2d
    )  # transform column with fitted one hot encoder
    encoded_ops_dense = encoded_ops.toarray()  # convert sparse matrix to dense matrix
    encoded_ops_df = pd.DataFrame(
        encoded_ops_dense, columns=le_ops.categories_[0]
    )  # convert dense matrix to dataframe
    encoded_ops_df.columns = [
        column + "_" + col for col in encoded_ops_df.columns
    ]  # rename columns
    claims = pd.concat(
        [claims, encoded_ops_df], axis=1
    )  # concat claims and encoded_ops_df

    ma_column = column + "_Major"  # add _Major to column name
    claims[ma_column].fillna("", inplace=True)  # fill NaN values with ""
    claims[ma_column] = claims[c_column].astype(str) + claims[ma_column].astype(
        str
    )  # concat Chapter and Major to one column
    claims[ma_column] = pd.to_numeric(
        claims[ma_column], errors="coerce"
    )  # convert column to numeric

    mi_column = column + "_Minor"  # add _Minor to column name
    claims[mi_column].fillna("", inplace=True)  # fill NaN values with ""
    claims[mi_column] = claims[ma_column].astype(str) + claims[mi_column].astype(
        str
    )  # concat Chapter and Major to one column
    claims[mi_column] = pd.to_numeric(
        claims[mi_column], errors="coerce"
    )  # convert column to numeric

    claims.drop(
        [column, c_column], axis=1, inplace=True
    )  # drop Chapter (now one hot encoded) and initial column (now named with _Minor)

claims.replace("n/A", 0, inplace=True)  # replace n/A with 0

le_drg = OrdinalEncoder()  # initialize encoder for DRG codes
drg = drg.DRG().get_DRG_list_details()  # get DRG codes from DRG class
drg = drg["DRG"].to_list()  # convert DRG codes to list
drg.extend(
    [
        "Y01Z",
        "B76A",
        "U43Z",
        "E76A",
        "D23Z",
        "B11Z",
        "K01Z",
        "B61B",
        "A16A",
        "D01A",
        "Y61Z",
        "F29Z",
        "E41Z",
        "A04A",
        "I96Z",
        "I40Z",
        "U41Z",
        "B13Z",
        "B49Z",
        "L90A",
        "L90C",
        "F37Z",
        "A43Z",
        "G51Z",
        "A90A",
        "K43Z",
        "A90B",
        "B46Z",
        "W40Z",
        "U01Z",
        "Z43Z",
        "Z02Z",
        "W01A",
        "L90B",
        "B43Z",
        "Z42Z",
        "U42A",
        "F45Z",
        "W05Z",
    ]
)  # manually adding codes from individual contracts
drg = np.array(drg).reshape(-1, 1)  # reshape DRG codes to 2d array
le_drg.fit(drg)  # fit encoder to DRG codes
claims["DRG"] = le_drg.transform(
    claims["DRG"].values.reshape(-1, 1)
)  # transform DRG codes with fitted encoder

claims.replace(0, None, inplace=True)  # replace 0 with None
claims.fillna(0, inplace=True)  # fill NaN values with 0
claims["GENDER"].replace(
    ["M", "F", "D"], [1, 2, 3], inplace=True
)  # encode the gender numerically


claims.drop(
    ["ADMISSION_DATE", "DISCHARGE_DATE", "PLZ_PAT"], axis=1, inplace=True
)  # drop columns with dates (duration is sufficient) and PLZ_PAT (does not have any information in this model)
claims = claims.drop(claims[claims["CLAIM"] == 0].index)  # drop claims with 0 as CLAIM

print(claims.info())
print(claims)

claims[["ID", "FRAUD_ID"]].to_csv(
    "data/generated data/preprocessing/ID_Fraud_Mapping.csv", index=False
)  # csv output for easier debugging
claims.drop("FRAUD_ID", axis=1, inplace=True)  # drop FRAUD_ID column

# save claims with HOSPITAL_ID between 270 and 299 as validation set
claims_299 = claims[claims["HOSPITAL_ID"].between(270, 299)]  # define validation set
claims_299.drop(
    ["HOSPITAL_ID", "PLZ_HOSP"], axis=1, inplace=True
)  # drop HOSPITAL_ID and PLZ_HOSP columns as identifiers
claims_299.to_csv(
    "data/generated data/preprocessing/validation.csv", index=False
)  # validation set
claims.drop(
    claims[claims["HOSPITAL_ID"].between(270, 299)].index, inplace=True
)  # drop validation set from claims
claims.drop(
    ["HOSPITAL_ID", "PLZ_HOSP"], axis=1, inplace=True
)  # drop HOSPITAL_ID and PLZ_HOSP columns as identifiers
claims.to_csv(
    "data/generated data/preprocessing/preprocessed.csv", index=False
)  # training set
