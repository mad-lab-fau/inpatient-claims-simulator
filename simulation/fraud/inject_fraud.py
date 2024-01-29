import sys

from simulation.generation.adjust_coding import Adjustments
import simulation.generation.drg as drg

sys.path.insert(0, "")

import random
import pandas as pd
import numpy as np

import simulation.utility.grouper_wrapping as grouper_wrapping
import simulation.utility.utils as utils
import simulation.utility.config as config
import simulation.fraud.create_fraud as create_fraud

fraud_prob = (
    config.fraud_probability
)  # change for lower probability, value should be between 0 and 1
grouper = grouper_wrapping.Grouper()
fraud = create_fraud.FRAUD(grouper)
drgo = drg.DRG()


def inject(hospital_ids: list, random_state: int = None) -> None:
    """Injecting fraudulent cases in existing claims data.

    Iterating over all claims cases, two steps have to be taken for conversion to fraud:
        1. a random value is smaller than fraud_prob;
        2. the condition for one of the fraud options is fulfilled

    Args:
        hospital_id (list): the hospitals' IDs that are acting fraudulent
        random_state (int, optional): random state for reproducibility. Defaults to None.
    """

    adjustments = Adjustments(random_state=random_state)

    # open('data/generated data/temporary_combinations.csv', 'w').close() # clear temporary_combinations.csv before each run
    open(
        "data/generated data/grouper data/input.txt", "w"
    ).close()  # clear input.txt before each run

    temp_change = pd.DataFrame()
    temp_bloody = pd.DataFrame()
    ids = []
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    claims = pd.read_csv("data/generated data/claims.csv")
    rel_claims = claims[claims["HOSPITAL_ID"].isin(hospital_ids)]

    for index, row in rel_claims.iterrows():  # looping over all relevant claims
        if np.random.rand() < fraud_prob:  # check if claim should be changed to fraud
            changed_to_fraud = False
            fraud_id = 0

            # breathing fraud
            if row["VENTILATION"] != 0:  # check if claim has ventilation
                temp = fraud.increase_ventilation(int(row["VENTILATION"]))

                if temp != row["VENTILATION"]:  # check if ventilation was changed
                    claims.loc[index, "VENTILATION"] = temp
                    changed_to_fraud = True
                    fraud_id = 1

            # caesarean
            if row["PRIMARY_ICD"] == "O80":  # check if claim is a c-section
                claims.loc[index, "PRIMARY_ICD"] = fraud.change_to_caesarean(
                    row["PRIMARY_ICD"]
                )
                changed_to_fraud = True
                fraud_id = 2

            if row["WEIGHT"] != "n/A":  # check if claim has weight
                temp = fraud.decrease_weight(row["WEIGHT"])
                if temp != row["WEIGHT"]:  # check if weight was changed
                    claims.loc[index, "WEIGHT"] = temp
                    changed_to_fraud = True
                    fraud_id = 3

            # newborn personal care
            if (
                row["AGE"] < 1
            ) and changed_to_fraud == False:  # check if claim is for a newborn
                secondary = row[
                    [
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
                ].to_list()
                secondary = utils.strip_nA(secondary)
                secondary = fraud.newborn_add_personal_care(row["AGE"], secondary)
                secondary.extend(
                    ["n/A"] * (20 - len(secondary)) if len(secondary) < 20 else []
                )  # extend list to length 20
                claims.loc[
                    index,
                    [
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
                    ],
                ] = secondary

                changed_to_fraud = True
                fraud_id = 4

            if (
                not changed_to_fraud
            ):  # check if claim was changed to fraud in the previous steps
                if (
                    np.random.uniform(0, 1) <= config.ratio_change_bloody
                ):  # check if claim should be changed to ICD changes
                    secondary = row[
                        [
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
                    ].to_list()
                    secondary = utils.strip_nA(secondary)

                    # primary, secondary =
                    # fraud.change_ICD_order(id = row["ID"], primary=row["PRIMARY_ICD"], secondary=secondary)
                    temp_change = fraud.change_ICD_order(
                        id=row["ID"],
                        primary=row["PRIMARY_ICD"],
                        secondary=secondary,
                        temp_comb=temp_change,
                    )
                else:
                    ids = fraud.adjust_duration(
                        id=row["ID"], days=row["DURATION"], ids=ids
                    )  # check if claim should be changed to bloody releases

            if changed_to_fraud:
                claims.loc[index, "POTENTIAL_FRAUD"] = 1
                claims.loc[index, "FRAUD_ID"] = fraud_id

            primary = claims.loc[index, "PRIMARY_ICD"]
            secondary = utils.strip_nA(
                claims.loc[
                    index,
                    [
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
                    ],
                ].to_list()
            )
            operations = utils.strip_nA(
                claims.loc[
                    index,
                    [
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
                    ],
                ].to_list()
            )
            ventilation = claims.loc[index, "VENTILATION"] != 0

            (
                primary,
                secondary,
                operations,
                gender,
            ) = adjustments.adjust_coding(  # adjust coding according to coding guidelines
                age=row["AGE"],
                gender=row["GENDER"],
                primary=primary,
                secondary=secondary,
                operations=operations,
                zip=row["PLZ_PAT"],
                pump=ventilation,
            )
            claims.loc[index, "PRIMARY_ICD"] = primary

            if len(secondary) > 20:
                secondary = secondary[:20]

            if len(operations) > 20:
                operations = operations[:20]

            secondary.extend(
                ["n/A"] * (20 - len(secondary)) if len(secondary) < 20 else []
            )
            operations.extend(
                ["n/A"] * (20 - len(operations)) if len(operations) < 20 else []
            )

            claims.loc[
                index,
                [
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
                ],
            ] = secondary
            claims.loc[
                index,
                [
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
                ],
            ] = operations

    claims = select_ICD_order(claims=claims, combinations=temp_change)
    claims = adjust_duration(claims, ids)
    claims.to_csv("data/generated data/claims_with_fraud.csv", index=False)
    # print(claims["POTENTIAL_FRAUD"])


def adjust_duration(claims: pd.DataFrame, ids: list) -> pd.DataFrame:
    """Adjusts the duration of the claims.

    For each claim ID the method sets the new duration according to the fraud grouping.

    Args:
        claims (pd.DataFrame): the claims DataFrame
        temp_bloody (pd.DataFrame): the bloody DataFrame of all tried combinations

    Returns:
        pd.DataFrame: the claims DataFrame with the new duration
    """
    # print(temp_bloody)
    grouper.execute_interim_grouping()
    temp_bloody = pd.read_csv("data/generated data/claims_with_drg.csv")

    for id in ids:
        duration = select_duration(id, temp_bloody[temp_bloody["ID"] == id])

        if duration != claims.loc[claims["ID"] == id, "DURATION"].item():
            claims.loc[id, "DURATION"] = duration
            claims.loc[id, "POTENTIAL_FRAUD"] = 1
            claims.loc[id, "FRAUD_ID"] = 5

    open(
        "data/generated data/grouper data/input.txt", "w"
    ).close()  # clear input.csv after each run
    return claims


def select_duration(id: int, temp_bloody: pd.DataFrame) -> int:
    """Selects the new duration for a given claim ID.

    The method uses the fraud grouping result to extract the new duration for a given claim ID according to the effective weight.

    Args:
        id (int): the claim ID
        temp_bloody (pd.DataFrame): the bloody DataFrame

    Returns:
        int: the new duration
    """
    if len(temp_bloody) != 1:
        raise ValueError("More than one duration for one claim ID")
    else:
        drg = temp_bloody["DRG"].item()
        ugv = drgo.get_ugv(drg)
        days = temp_bloody["DURATION"].item()
        if ugv > 0 and ugv < days:
            diff = days - ugv
        else:
            diff = 0

        if days < 3 and ugv <= days - 1:
            return days - 1
        else:
            diff = round(diff * 0.25)
            if diff > 5:
                diff = 5  # limit the maximum increase to 5 days
            return days - round(diff * 0.25)


def select_ICD_order(claims: pd.DataFrame, combinations: pd.DataFrame) -> pd.DataFrame:
    """Selects the ICD order for the claim.

    For each claim ID the method sets the new primary ICD and the new secondary ICDs according to the fraud grouping.

    Args:
        claims (pd.DataFrame): the claims DataFrame
        combinations (pd.DataFrame): the combinations DataFrame of all tried combinations

    Returns:
        pd.DataFrame: the claims DataFrame with the new ICD order
    """
    result = grouper.execute_upcoding_grouping()  # execute upcoding grouping
    if not result.empty:
        ids = list(result["ID"].unique())

        for id in ids:
            primary, secondary = extract_primary_secondary(
                id, result, combinations
            )  # extract the best combination of primary and secondary ICDs for a given claim ID

            if (
                primary is not None
                and claims.loc[claims["ID"] == id, "PRIMARY_ICD"].item() != primary
            ):  # check if primary ICD was changed
                secondary.extend(
                    ["n/A"] * (20 - len(secondary)) if len(secondary) < 20 else []
                )

                claims.loc[id, "PRIMARY_ICD"] = primary
                claims.loc[
                    id,
                    [
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
                    ],
                ] = secondary
                claims.loc[id, "POTENTIAL_FRAUD"] = 1
                claims.loc[id, "FRAUD_ID"] = 6
        open(
            "data/generated data/grouper data/input.txt", "w"
        ).close()  # clear input.csv after each run
    return claims


def extract_primary_secondary(
    id: int, result: pd.DataFrame, combinations: pd.DataFrame
):
    """Extracts the best combination of primary and secondary ICDs for a given claim ID.

    The method uses the fraud grouping result to extract the best combination of primary and secondary ICDs for a given claim ID according to the effective weight.

    Args:
        id (int): the claim ID
        result (pd.DataFrame): the fraud grouping result
        combinations (pd.DataFrame): the combinations DataFrame of all tried combinations

    Returns:
        str: the new primary ICD
        list: the new secondary ICDs
    """
    max_index = result.loc[
        result["ID"] == id, "EFF_WEIGHT"
    ].idxmax()  # get index of the best combination
    try:
        rel_comb = combinations.iloc[[max_index]]  # get the best combination
    except:
        print(max_index)
        raise
    secondary = rel_comb["ICD"].tolist()[
        0
    ]  # get the secondary ICDs of the best combination
    primary = secondary.pop(
        0
    )  # get the primary ICD of the best combination as the first element of the secondary ICDs
    counter_max = len(
        result.loc[result["ID"] == id]
    )  # set maximum number of iterations to the number of tried combinations
    iteration = 0
    while (
        primary.startswith("Z") and random.uniform(0, 1) > 0.0001
    ):  # check if primary ICD is a Z code and if the random value is smaller than 0.0001; Z codes lead to higher claim but do not occur usually in the data
        max_index = result.loc[result["ID"] == id, "EFF_WEIGHT"].nlargest(2).index[-1]
        rel_comb = combinations.iloc[[max_index]]
        secondary = rel_comb["ICD"].tolist()[0]
        if len(secondary) > 0:
            primary = secondary.pop(0)
        else:
            return None, None

        iteration += 1
        if iteration > counter_max:
            return None, None
        # print("new primary: " + primary)

    return primary, secondary
