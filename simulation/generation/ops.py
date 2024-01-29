import json
import sys

sys.path.insert(0, "")

import random
import numpy as np
import pandas as pd
import marisa_trie

import simulation.utility.utils as utils
import simulation.utility.icd_ops_mapping as icd_ops_mapping

import simulation.generation.icd as icd
import simulation.generation.drg as drg


class OPS:
    """OPS is used to create codes for procedures and operations.

    When initiallizing it with a constructor, a DRG instance is necessary.

    Attributes:
        ops_csv: a list of all code-able OPS codes
        ops_stat: a DataFrame with total number of occurances per group of an OPS code
        probs_csv: a DataFrame with the probability of an OPS code within the group (sums up to 1)
        probs_csv_2: a DataFrame with the probability of an OPS code within the group in relation to the number of inpatient stays
        drg: an instance of `DRG`
    """

    def __init__(self, drg, random_state: int = None):
        """Creating an instance of OPS.

        Initializing all important values with empty states. This is replaced with the first access to the variable.
        Variables here used to hold them in memory without the need to access the same information twice from files.

        Args:
            drg (:obj: `DRG`): instance of DRG for matching ICDs on OPS
            random_state (int, optional): random state for reproducibility. Defaults to None.
        """
        self.ops_csv = []
        self.ops_stat = pd.DataFrame()
        self.probs_csv = pd.DataFrame()
        self.probs_csv_2 = pd.DataFrame()
        self.drg = drg
        self.mapping = icd_ops_mapping.ICD_OPS_Mapping(self)
        self.negative = pd.read_csv("data/OPS/generated/negative_OPS_list.csv")

        file = open("data/OPS/generated/ops_diff.json", "r")
        self.ops_diff = json.load(file)
        file.close()

        self.random_state = random_state
        random.seed(random_state)

    def get_ops_list(self) -> list:
        """Get the list of all possible OPS codes.

        All not code-able and not defined codes are filtered out from a current list of OPS codes.

        Returns:
            List of OPS codes.
        """
        ops_csv = self.ops_csv
        if len(ops_csv) == 0:
            ops_csv = pd.read_csv(
                "data/OPS/original/ops2021syst_kodes.csv", sep=";", header=None
            )
            # filtering only for terminal key numbers (kodierbarer Endpunkt)
            ops_csv = ops_csv[ops_csv.loc[:, 1] != "N"]
            # filtering only for operations with defined names (no 'Nicht belegte Schlüsselnummer')
            ops_csv = ops_csv[
                ~ops_csv.loc[:, 9].str.contains("Nicht belegte Schlüsselnummer")
            ]
            ops_csv = ops_csv.iloc[:, 6]
            ops_csv = ops_csv.to_numpy()
            ops_csv = ops_csv.tolist()
            self.ops_csv = ops_csv
        return ops_csv

    def get_random_ops(self) -> str:
        """Get a random OPS code given the Standard-OPS-list.

        Args:
            random_state (int): None if not reproducable, int if reproducable. Defaults to None.

        Returns:
            String of a random OPS code.

        """
        ops = random.choice(self.get_ops_list())
        return ops

    def get_random_ops_from_major(self, age: int, gender: str, major: str) -> str:
        """Get a random OPS code given the Standard-OPS-list from an OPS Code major group.

        The major group is defined by the first n digits of the OPS code. The probability is based on the age and gender given.
        It works with any length of OPS code, as long as it matches the official OPS code list.

        Args:
            age (int): the age of the patient
            gender (str): the gender of the patient
            major (str): the major group of the OPS code

        Returns:
            String of a random OPS code."""

        probs_csv = self.probs_csv
        if probs_csv.empty:
            probs_csv = pd.read_csv("data/OPS/generated/ops_probs.csv")
            self.probs_csv = probs_csv

        age_group = utils.get_age_group(age)
        gender_group = utils.get_gender_group(gender)
        group = gender_group + "_" + age_group

        full_majors = marisa_trie.Trie(probs_csv["OPS_Code_ "].to_list())
        mask = probs_csv["OPS_Code_ "].isin(full_majors.keys(major))
        result = probs_csv[mask]

        probs = np.array(result[["OPS_Code_ ", group]][group].to_list())
        r_major = np.random.choice(result["OPS_Code_ "], p=(probs / sum(probs)))

        final = self.get_ops_list()
        trie = marisa_trie.Trie(final)
        final_selection = trie.keys(r_major)
        f_ops = np.random.choice(final_selection)

        return f_ops

    def read_ops_statistics(self):
        """Reading the total numbers of cases an OPS code is used in hospital claims.

        When reading the file, information on year (always 2021), and description (irrelevant for further use) are dropped along with other unnecessary data.
        The string "OPS-" is replaced for interoperability with other lists.
        Remaining nA-values are referring to sums and therefore filled with "INS" to be accessed later.
        To remove any long texts that do not belong to the necessary parts, only OPS codes with less than 6 chars are kept.

        Returns:
            a `Pandas.DataFrame()` instance
        """
        print("Reading OPS statistics")
        probs = self.ops_stat
        if probs.empty:
            probs = pd.read_csv(
                "data/OPS/original/ops_primary.csv",
                sep=";",
                encoding="utf-8",
                header=None,
                low_memory=False,
            )

            new_header = probs.iloc[0] + "_" + probs.iloc[1]
            probs.columns = new_header
            probs = probs.iloc[2:]
            probs = probs.drop(["Jahr_ ", "Beschreibung_ "], axis=1)

            probs["OPS_Code_ "] = probs["OPS_Code_ "].str.replace("OPS-", "")
            probs["OPS_Code_ "] = probs["OPS_Code_ "].fillna("INS")
            probs = probs[probs["OPS_Code_ "].str.len() <= 5]

            probs = probs.replace("-", 0)
            self.ops_stat = probs
        return probs

    def get_ops_stat_list(self) -> list:
        """Get a list of unique OPS values used in the statistics file.

        Returns:
            List of OPS codes
        """
        probs = self.read_ops_statistics()
        return list(probs["OPS_Code_ "].unique())

    # run only in case it is necessary, took me 10 hours to run once (therefore the csv output)
    def calculate_ops_probabilities_v2(self):
        """Calculate the ratio of the number of OPS codes compared to the total number of hospital treatments.
        (according to the number of primary ICD)

        Attention: run only in case it is necessary, took me 10 hours to run once (therefore the csv output)
        """
        print("read primary ops")
        probs = self.read_ops_statistics()
        print("primary ops read")
        ages = utils.get_age_list()
        genders = utils.get_gender_list()
        opss = self.get_ops_stat_list()
        print("start")
        for i in opss:
            print(i)
            for a in ages:
                for g in genders:
                    group = g + "_" + a
                    # probs.loc[(probs["OPS_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()
                    # print(group + "  "  + i + "  " + str((int(probs.loc[(probs["OPS_Code_ "] == i), group].item()))/(int(probs.loc[(probs["OPS_Code_ "] == 'INS'), group].item()))))
                    probs.loc[(probs["OPS_Code_ "] == i), group] = (
                        int(probs.loc[(probs["OPS_Code_ "] == i), group].item())
                    ) / (icd.get_case_number_sum(group))
                    # print(probs.loc[(probs["OPS_Code_ "] == i), group])
        probs["OPS_Code_ "] = probs["OPS_Code_ "].str.lower()
        probs.to_csv("data/OPS/generated/ops_probs_2.csv", index=False)
        print("THE END")

    # run only in case it is necessary, took me 2.5 hours to run once (therefore the csv output)
    def calculate_ops_probabilities(self):
        """Calculate the ratio of the number of OPS codes compared to the total number of OPS (the difference to calculate_ops_probabilities_2())

        Attention: run only in case it is necessary, took me 2.5 hours to run once (therefore the csv output)
        """
        probs = self.read_ops_statistics()
        ages = utils.get_age_list()
        genders = utils.get_gender_list()
        opss = self.get_ops_stat_list()

        for i in opss:
            for a in ages:
                for g in genders:
                    group = g + "_" + a
                    # probs.loc[(probs["OPS_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()
                    # print(group + "  "  + i + "  " + str((int(probs.loc[(probs["OPS_Code_ "] == i), group].item()))/(int(probs.loc[(probs["OPS_Code_ "] == 'INS'), group].item()))))
                    probs.loc[(probs["OPS_Code_ "] == i), group] = (
                        int(probs.loc[(probs["OPS_Code_ "] == i), group].item())
                    ) / (int(probs.loc[(probs["OPS_Code_ "] == "INS"), group].item()))
        probs["OPS_Code_ "] = probs["OPS_Code_ "].str.lower()
        probs.to_csv("data/OPS/generated/ops_probs.csv", index=False)
        print("THE END")

    def adjust_probs(self, row, group):
        """Adjust the probabilities of OPS codes to the error calculated in ops_diff.json.

        Args:
            row (Pandas.Series): a row of the DataFrame

        Returns:
            float: the adjusted probability
        """
        # print(row)
        ops = row["OPS_Code_ "]
        diff = self.ops_diff[ops]
        """if diff < 0:
            factor = (1+diff)
        else:
            factor = 1 + diff ** 2"""
        ret = row[group] + diff
        return ret  # row[group] + diff

    def get_random_ops_list_v2(
        self, age: int, gender: str, primary: str, secondary: list
    ) -> list:
        """Get a list of random OPS codes depending on age, gender, and diagnoses.

        Getting the list of possible OPS codes depending on diagnoses by the mapping provided in `DRG` and storing in a trie (for performance reasons).
        Then mapping the probabilities accodring to age and gender on all OPS codes and using the probabilities of all relevant codes by scanning
        through the trie with prefixes() (to match different levels of OPS codes).
        Probabilities are normalized as only a fraction of the original codes is used for each case. According to those probabilities one OPS code is drawn.
        The number of OPS codes returned is calculated by a poisson distribution given the accumulated probabilities of all OPS codes per group.
        In the final selection, the higher-level OPS code will be replaced with one of the codeable codes with the same beginning.

        Args:
            age (int): the age of the patient
            gender (str): "M" for male, "F" for female, "D" for anything else
            primary (str): the primary ICD code
            secondary (list of str): the list of secondary ICD codes


        Returns:
            list of strings or empty list (if no matching OPS could be found for diagnoses)

        """
        probs_csv = self.probs_csv
        if probs_csv.empty:
            probs_csv = pd.read_csv("data/OPS/generated/ops_probs.csv")
            self.probs_csv = probs_csv

        probs_csv_2 = self.probs_csv_2
        if probs_csv_2.empty:
            probs_csv_2 = pd.read_csv("data/OPS/generated/ops_probs_2.csv")
            self.probs_csv_2 = probs_csv_2

        age_group = utils.get_age_group(age)
        gender_group = utils.get_gender_group(gender)
        group = gender_group + "_" + age_group

        ops_list_5_digits, ops_list = self.drg.get_OPS_list_on_ICD(
            primary=primary, secondary=secondary
        )  # getting the OPS codes from the mapping via DRGs without 5-er OPS codes
        ex = self.mapping.get_operations_ops(
            primary
        )  # getting the 5-er OPS codes from the mapping for the primary ICD
        if ex is not None:
            ops_list_5_digits.extend(
                ex
            )  # adding the OPS codes from the mapping for the primary ICD (=cause for hospitalization)
        
        result = probs_csv[["OPS_Code_ ", group]]
        ops_trie = marisa_trie.Trie(ops_list_5_digits)
        mask = result["OPS_Code_ "].apply(lambda x: any(ops_trie.prefixes(x)))
        result = result[mask]

        n_ops = utils.poisson_greater_than_zero(
            (probs_csv_2.loc[(probs_csv_2["OPS_Code_ "] == "ins"), group].item()),
            random_state=self.random_state,
        )  # drawing the number of OPS codes
        result.apply(
            self.adjust_probs, args=([group]), axis=1
        )  # adjusting the probabilities according to the error calculated in ops_diff.json
        result[group] = result[group] + np.abs(
            result[group].min()
        )  # shifting the probabilities to be positive
        result[group] = (
            result[group] / result[group].sum()
        )  # normalizing the probabilities
        ops = []
        if len(ops_list_5_digits) > 0 and not result["OPS_Code_ "].empty:
            for n in range(n_ops):
                ops.append(
                    np.random.choice(result["OPS_Code_ "], p=result[group])
                )  # drawing the OPS codes

            # final = self.get_ops_list()
            final = ops_list  # testing if results are more relialbe if using the OPS codes from the mapping and not the whole list
            f = self.get_ops_list()
            # remove all OPS codes where the first 5 digits are already in final
            first_5_digits_final = [x[:5] for x in final]
            f = [x for x in f if x[:5] not in first_5_digits_final]
            final.extend(f)
            trie = marisa_trie.Trie(final)
            r_operations = []
            for o in ops:
                final_selection = trie.keys(
                    o
                )  # as OPS Codes might be given in different lengths in different statistics, this is used to find all relevant codes
                final_selection = self.substract_negatives(primary, final_selection)
                if len(final_selection) == 0:
                    return []
                f_ops = np.random.choice(final_selection)
                if f_ops not in r_operations:
                    r_operations.append(f_ops)

            return r_operations
        return []

    def substract_negatives(self, primary, final_selection):
        """Substract all OPS codes that are not allowed for the given primary ICD.

        Args:
            primary (str): the primary ICD code
            final_selection (list of str): the list of OPS codes

        Returns:
            list of strings: the list of OPS codes without the OPS codes that are not allowed for the given primary ICD
        """

        neg = self.negative[self.negative["ICD"] == primary]
        neg = neg["OPS"].to_list()
        final_selection = list(
            set(final_selection) - set(neg)
        )  # removing all OPS codes that are not allowed for the given primary ICD
        return final_selection

