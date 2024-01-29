import sys

sys.path.insert(0, "")

import random
import numpy as np
import pandas as pd
import marisa_trie
import sys

import simulation.utility.plz as plzs
import simulation.utility.utils as utils

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ICD:
    """ICD is used to generate primary and secondary ICD codes, and getting a random age depending on primary ICDs (= hospital treatments).

    Attributes:
        icd_csv: a list of all code-able ICD codes
        probs_male: a list of probabilities for each age group for male patients
        probs_female: a list of probabilities for each age group for female patients
        probs_diverse: a list of probabilities for each age group for all genders
        prim_csv: a DataFrame with the total number of occurances per group of a primary ICD code
        sec_csv: a DataFrame with the total number of occurances per group of a secondary ICD code
        sec_prob_csv: a DataFrame with the probability of a secondary ICD code within the group in relation to the number of inpatient stays
        prim_prob_csv: a DataFrame with the probability of a primary ICD code within the group (sums up to 1)
        plz = a `PLZ` instance
    """

    def __init__(self, plz=None, random_state: int = None):
        """Creating an instance of ICD.

        Initializing all important values with empty states. This is replaced with the first access to the variable.
        Variables here used to hold them in memory without the need to access the same information twice from files.

        Args:
            plz (PLZ): a `PLZ` instance. Defaults to None.
            random_state (int): None if not reproducable, int if reproducable. Defaults to None.
        """
        self.prim_icd_csv = []  # list of all code-able primary ICD codes
        self.sec_icd_csv = []  # list of all code-able secondary ICD codes
        self.icd_desc = (
            pd.DataFrame()
        )  # DataFrame with all ICD codes and their description
        self.probs_male = []  # probabilities of being male dependent on age
        self.probs_female = []  # probabilities of being female dependent on age
        self.probs_diverse = []  # probabilities of being diverse dependent on age
        self.prim_csv = (
            pd.DataFrame()
        )  # DataFrame with the total number of occurances per group of a primary ICD code
        self.sec_csv = (
            pd.DataFrame()
        )  # DataFrame with the total number of occurances per group of a secondary ICD code
        self.sec_prob_csv = (
            pd.DataFrame()
        )  # DataFrame with the probability of a secondary ICD code within the group in relation to the number of inpatient stays
        self.prim_prob_csv = (
            pd.DataFrame()
        )  # DataFrame with the probability of a primary ICD code within the group (sums up to 1)

        if plz is None:
            self.plz = plzs.PLZ(random_state=random_state)
        else:
            self.plz = plz

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def get_prim_icd_list(self) -> list:
        """Get the list of all possible primary ICD codes.

        All not code-able and not defined codes are filtered out from a current list of ICD codes.

        Returns:
            List of ICD codes.
        """
        icd_csv = self.prim_icd_csv
        if len(icd_csv) == 0:
            icd_csv = pd.read_csv(
                "data/ICD/original/icd10gm2021syst_kodes.csv",
                sep=";",
                header=None,
            )
            # filtering only for terminal key numbers (kodierbarer Endpunkt)
            icd_csv = icd_csv[icd_csv.loc[:, 1] != "N"]
            # filtering only for diagnoses with defined diseases (no 'Nicht belegte Schlüsselnummer')
            icd_csv = icd_csv[icd_csv.loc[:, 25] != "N"]
            icd_csv = icd_csv.iloc[:, 5]
            icd_csv = icd_csv.to_numpy()
            icd_csv = icd_csv.tolist()
            icd_csv = [x for x in icd_csv if not x.endswith("!")]
            self.prim_icd_csv = icd_csv

        return icd_csv

    def get_sec_icd_list(self) -> list:
        """Get the list of all possible secondary ICD codes.

        All not code-able and not defined codes are filtered out from a current list of ICD codes.

        Returns:
            List of ICD codes.
        """
        icd_csv = self.sec_icd_csv
        if len(icd_csv) == 0:
            icd_csv = pd.read_csv(
                "data/ICD/original/icd10gm2021syst_kodes.csv",
                sep=";",
                header=None,
            )
            # filtering only for terminal key numbers (kodierbarer Endpunkt)
            icd_csv = icd_csv[icd_csv.loc[:, 1] != "N"]
            # filtering only for diagnosis with defined diseases (no 'Nicht belegte Schlüsselnummer')
            icd_csv = icd_csv[icd_csv.loc[:, 25] != "N"]
            icd_csv = icd_csv.iloc[:, 6]
            icd_csv = icd_csv.to_numpy()
            icd_csv = icd_csv.tolist()
            self.sec_icd_csv = icd_csv

        return icd_csv

    def check_gender_relevancy(self) -> pd.DataFrame():
        """Check all ICD codes to their relevancy for gender and note it in column "GENDER".

        1 = female,
        2 = male,
        0 = both/not relevant

        Returns:
            a `Pandas.DataFrame()` instance
        """
        icd_desc = self.icd_desc
        if icd_desc.empty:
            icd_desc = pd.read_csv(
                "data/ICD/original/icd10gm2021syst_kodes.csv", sep=";", header=None
            )
            # filtering only for terminal key numbers (kodierbarer Endpunkt)
            icd_desc = icd_desc[icd_desc.loc[:, 1] != "N"]
            # filtering only for diagnosis with defined diseases (no 'Nicht belegte Schlüsselnummer')
            icd_desc = icd_desc[icd_desc.loc[:, 25] != "N"]
            icd_desc = icd_desc.iloc[:, [6, 8]]
            icd_desc.columns = ["ICD", "DESCRIPTION"]

            def apply_gender(description):
                female = [
                    "vulva",
                    "vagina",
                    "uterus",
                    "weiblich",
                    "frau",
                    "uterin",
                    "schwanger",
                    "hysterektomie",
                    "endometrium",
                ]
                male = [
                    "prostata",
                    "penis",
                    "samen",
                    "männlich",
                    "mann",
                    "hoden",
                    "skrotum",
                    "hypospadie",
                    "hydrozele",
                    "priapismus",
                ]
                if any(x in description.lower() for x in female):
                    return 1
                elif any(x in description.lower() for x in male):
                    return 2
                else:
                    return 0

            icd_desc["GENDER"] = icd_desc["DESCRIPTION"].apply(apply_gender)
            # drop column DESCRIPTION
            icd_desc.drop("DESCRIPTION", axis=1, inplace=True)
            self.icd_desc = icd_desc
        return icd_desc

    def get_gender_relevancy(self, icd: str) -> int:
        """Get the gender relevancy for a given ICD-code.

        Args:
            icd(str): ICD-code in question

        Returns:
            an `int` value for the gender (1=female, 2=male, 0=both/not relevant)
        """
        icd_desc = self.check_gender_relevancy()
        return icd_desc[icd_desc["ICD"] == icd]["GENDER"].item()

    def read_primary_icd_statistics(self):
        """Reading the total numbers of cases a primary ICD code is used in hospital claims.

        When reading the file, information on year (always 2021), and description (irrelevant for further use) are dropped along with other unnecessary data.
        The string "ICD-" is replaced for interoperability with other lists.
        Remaining nA-values are referring to sums and therefore filled with "INS" to be accessed later.
        To remove any long texts that do not belong to the necessary parts, only OPS codes with less than 4 chars are kept.

        Returns:
            a `Pandas.DataFrame()` instance
        """
        prim_csv = self.prim_csv
        if prim_csv.empty:
            prim_csv = pd.read_csv(
                "data/ICD/original/icd_primary.csv",
                sep=";",
                encoding="utf-8",
                header=None,
                low_memory=False,
            )

            new_header = prim_csv.iloc[0] + "_" + prim_csv.iloc[1]
            prim_csv.columns = new_header
            prim_csv = prim_csv.iloc[2:]
            prim_csv = prim_csv.drop(["Jahr_ ", "Beschreibung_ "], axis=1)

            prim_csv["ICD_Code_ "] = prim_csv["ICD_Code_ "].str.replace("ICD10-", "")
            prim_csv["ICD_Code_ "] = prim_csv["ICD_Code_ "].fillna("INS")
            prim_csv = prim_csv[prim_csv["ICD_Code_ "].str.len() <= 3]

            prim_csv = prim_csv.replace("-", 0)
            self.prim_csv = prim_csv
        return prim_csv

    def get_case_number_sum(self, group: str) -> int:
        """Get the number of cases/hospital treatments per population group.

        Notice:
            this method should not be called manually

        Args:
            group (str): name of the population group in the form of "gender_age"

        Returns:
            an `int` representing the number of inpatient stays for a given group
        """
        probs = self.read_primary_icd_statistics()
        # group = gender + "_" + age
        state = self.plz.get_state_list()
        filtered_probs = probs[probs["ICD_Code_ "] == "INS"]
        number = 0
        # number += int(filtered_probs.loc[filtered_probs["Bundesland_ "].isin(state), group].sum())
        for s in state:
            number += int(
                filtered_probs.loc[filtered_probs["Bundesland_ "] == s, group].values[0]
            )
        # print(number)
        return number

    def get_age(self, gender: str) -> int:
        """Get a random age depending on a gender.

        Returning an random age value depending on a given gender  according to the age distribution in hospital cases for this gender.
        As age is only given in age-groups, a linear distribution is assumed within a group. The only exception is the group of oldest patients.
        There it is considered to be less likely the higher the age gets, with a maximum of 120 (considered as maximum biological age).

        Args:
            gender (str): "M" for male, "F" for female, "D" for anything else

        Returns:
            an `int` value for the age

        Raises:
            ValueError: if the age-group is not calculated properly
        """
        age_group = utils.get_age_list()
        gender_group = utils.get_gender_group(gender)

        if gender == "M":
            probs_male = self.probs_male
            if len(probs_male) == 0:
                for a in age_group:
                    group = gender_group + "_" + a
                    probs_male.append(self.get_case_number_sum(group=group))
                probs_male = probs_male / np.sum(probs_male)
                self.probs_male = probs_male

            age = np.random.choice(age_group, p=probs_male)
        elif gender == "F":
            probs_female = self.probs_female
            if len(probs_female) == 0:
                for a in age_group:
                    group = gender_group + "_" + a
                    probs_female.append(self.get_case_number_sum(group=group))
                probs_female = probs_female / np.sum(probs_female)
                self.probs_female = probs_female

            age = np.random.choice(age_group, p=probs_female)
        else:
            probs_diverse = self.probs_diverse
            if len(probs_diverse) == 0:
                for a in age_group:
                    group = gender_group + "_" + a
                    probs_diverse.append(self.get_case_number_sum(group=group))
                probs_diverse = probs_diverse / np.sum(probs_diverse)
                self.probs_diverse = probs_diverse

            age = np.random.choice(age_group, p=probs_diverse)

        if age == "unter 1 Jahr":
            return 0
        elif age == "1 bis unter 5 Jahre":
            return np.random.randint(1, 5)
        elif age == "5 bis unter 10 Jahre":
            return np.random.randint(5, 10)
        elif age == "10 bis unter 15 Jahre":
            return np.random.randint(10, 15)
        elif age == "15 bis unter 18 Jahre":
            return np.random.randint(15, 18)
        elif age == "18 bis unter 20 Jahre":
            return np.random.randint(18, 20)
        elif age == "20 bis unter 25 Jahre":
            return np.random.randint(20, 25)
        elif age == "25 bis unter 30 Jahre":
            return np.random.randint(25, 30)
        elif age == "30 bis unter 35 Jahre":
            return np.random.randint(30, 35)
        elif age == "35 bis unter 40 Jahre":
            return np.random.randint(35, 40)
        elif age == "40 bis unter 45 Jahre":
            return np.random.randint(40, 45)
        elif age == "45 bis unter 50 Jahre":
            return np.random.randint(45, 50)
        elif age == "50 bis unter 55 Jahre":
            return np.random.randint(50, 55)
        elif age == "55 bis unter 60 Jahre":
            return np.random.randint(55, 60)
        elif age == "60 bis unter 65 Jahre":
            return np.random.randint(60, 65)
        elif age == "65 bis unter 70 Jahre":
            return np.random.randint(65, 70)
        elif age == "70 bis unter 75 Jahre":
            return np.random.randint(70, 75)
        elif age == "75 bis unter 80 Jahre":
            return np.random.randint(75, 80)
        elif age == "80 bis unter 85 Jahre":
            return np.random.randint(80, 85)
        elif age == "85 bis unter 90 Jahre":
            return np.random.randint(85, 90)
        elif age == "90 bis unter 95 Jahre":
            return np.random.randint(90, 95)
        elif age == "95 Jahre und mehr":
            values = np.arange(95, 121)
            weights = 1 / values
            weights = weights / np.sum(weights)

            return np.random.choice(values, p=weights)
        else:
            raise ValueError("No age-group given")

    def get_primary_icd_list(self) -> list:
        """Get a list of ICD values in the statistics file.

        Returns:
            a `list` of unique ICD values
        """
        probs = self.read_primary_icd_statistics()
        return list(probs["ICD_Code_ "].unique())

    # run only in case it is necessary, took me 2.5 hours to run once (therefore the csv output)
    def calculate_primary_probabilities(self):
        """Calculate the ratio of the number of primary ICD codes compared to the total number of primary ICDs.

        Attention: run only in case it is necessary, took me 2.5 hours to run once (therefore the csv output)
        """
        probs = self.read_primary_icd_statistics()
        state = self.plz.get_state_list()
        ages = utils.get_age_list()
        genders = utils.get_gender_list()
        icds = self.get_primary_icd_list()
        for s in state:
            for i in icds:
                for a in ages:
                    for g in genders:
                        group = g + "_" + a
                        # probs.loc[(probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()
                        # print(s + "  " + group + "  "  + i + "  " + str((int(probs.loc[(probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()))/(int(probs.loc[(probs["ICD_Code_ "] == 'INS') & (probs["Bundesland_ "] == s), group].item()))))
                        probs.loc[
                            (probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s),
                            group,
                        ] = (
                            int(
                                probs.loc[
                                    (probs["ICD_Code_ "] == i)
                                    & (probs["Bundesland_ "] == s),
                                    group,
                                ].item()
                            )
                        ) / (
                            int(
                                probs.loc[
                                    (probs["ICD_Code_ "] == "INS")
                                    & (probs["Bundesland_ "] == s),
                                    group,
                                ].item()
                            )
                        )
        probs.to_csv("data/ICD/generated/primary_probs.csv", index=False)
        print("THE END")

    # --- secondary ICDs ---
    def read_secondary_icd_statistics(self) -> pd.DataFrame():
        """Reading the total numbers of cases a secondary ICD code is used in hospital claims.

        When reading the file, information on year (always 2021), and description (irrelevant for further use) are dropped along with other unnecessary data.
        The string "ICD-" is replaced for interoperability with other lists.
        Remaining nA-values are referring to sums and therefore filled with "INS" to be accessed later.
        To remove any long texts that do not belong to the necessary parts, only OPS codes with less than 4 chars are kept.

        Returns:
            a `Pandas.DataFrame()` instance
        """

        probs = self.sec_csv
        if probs.empty:
            probs = pd.read_csv(
                "data/ICD/original/icd_secondary.csv",
                sep=";",
                encoding="utf-8",
                header=None,
                low_memory=False,
            )
            self.sec_csv = probs
        new_header = probs.iloc[0] + "_" + probs.iloc[1]
        probs.columns = new_header
        probs = probs.iloc[2:]
        probs = probs.drop(["Jahr_ ", "Beschreibung_ "], axis=1)

        probs["ICD_Code_ "] = probs["ICD_Code_ "].str.replace("ICD10-", "")
        probs["ICD_Code_ "] = probs["ICD_Code_ "].fillna("INS")
        probs = probs[probs["ICD_Code_ "].str.len() <= 3]

        probs = probs.replace("-", 0)
        return probs

    def get_secondary_icd_list(self) -> list:
        """Get the list of all possible secondary ICD codes.

        All not code-able and not defined codes are filtered out from a current list of ICD codes.

        Returns:
            List of ICD codes.
        """
        probs = self.read_secondary_icd_statistics()
        return list(probs["ICD_Code_ "].unique())

    def get_n_secondary(self) -> None:
        """Calculate the ratio of the number of secondary ICD codes compared to the total number of primary ICDs

        Attention: run only in case it is necessary, took me 2.5 hours to run once (therefore the csv output)
        """
        primary_cases = self.read_primary_icd_statistics()
        secondary_cases = self.read_secondary_icd_statistics()
        result = secondary_cases.copy()

        icds = self.get_secondary_icd_list()
        state = self.plz.get_state_list()
        ages = utils.get_age_list()
        genders = utils.get_gender_list()
        for s in state:
            for i in icds:
                for a in ages:
                    for g in genders:
                        group = g + "_" + a
                        # probs.loc[(probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()
                        # print(s + "  " + group + "  "  + i + "  " + str((int(probs.loc[(probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()))/(int(probs.loc[(probs["ICD_Code_ "] == 'INS') & (probs["Bundesland_ "] == s), group].item()))))
                        result.loc[
                            (result["ICD_Code_ "] == i) & (result["Bundesland_ "] == s),
                            group,
                        ] = (
                            int(
                                secondary_cases.loc[
                                    (secondary_cases["ICD_Code_ "] == i)
                                    & (secondary_cases["Bundesland_ "] == s),
                                    group,
                                ].item()
                            )
                        ) / (
                            int(
                                primary_cases.loc[
                                    (primary_cases["ICD_Code_ "] == "INS")
                                    & (primary_cases["Bundesland_ "] == s),
                                    group,
                                ].item()
                            )
                        )
        result.to_csv("data/ICD/generated/secondary_probs_2.csv", index=False)
        # print(primary_cases.loc[(primary_cases["Bundesland_ "] == "Bayern")& (primary_cases["ICD_Code_ "] == "INS"), "männlich_95 Jahre und mehr"].item())
        # print(secondary_cases.loc[(secondary_cases["Bundesland_ "] == "Bayern")& (secondary_cases["ICD_Code_ "] == "U99"), "männlich_95 Jahre und mehr"].item())

    def get_secondary_list(self, gender: str, age: int, zip: int) -> list:
        """Get a list of secondary ICDs depending on gender, age, and location.

        Get a list of secondary ICDs by using the probabilities depending on number of hospital treatments by
        generating an array of random floats. If the random value is smaller than the probability, the ICD code is used.

        Args:
            gender (str): "M" for male, "F" for female, "D" for anything else
            age (int): age of the patient
            zip (int): plz of the patient

        Returns:
            a `list` of secondary ICDs
        """
        probs = self.sec_prob_csv
        if probs.empty:
            probs = pd.read_csv("data/ICD/generated/secondary_probs_2.csv")
            self.sec_prob_csv = probs
        state = self.plz.get_state_from_plz(zip)
        age_group = utils.get_age_group(age)
        gender_group = utils.get_gender_group(gender)
        group = gender_group + "_" + age_group
        result = probs.loc[probs["Bundesland_ "] == state, ["ICD_Code_ ", group]]
        result = result.drop(result.index[-1])
        secondary = []

        """for i, row in result.iterrows():
            # print(i)
            r = random.uniform(0, 1)
            if r <= row[group]:
                # print(str(r) + " <= " + str(row[group]))
                icd = row["ICD_Code_ "]
                final = self.get_icd_list()
                final_selection = [x for x in final if x.startswith(icd)]
                f_icd = random.choice(final_selection)
                secondary.append(f_icd)"""
        r = np.random.uniform(0, 1, len(result))
        mask = r <= result[group]
        selected_rows = result[mask]
        final = self.get_sec_icd_list()
        trie = marisa_trie.Trie(final)
        for icd in selected_rows["ICD_Code_ "]:
            final_selection = trie.keys(
                icd
            )  # get all ICD codes starting with the given suffix (for interoperability with other lists)
            # final_selection = [x for x in final if x.startswith(icd)]
            if gender == "M":
                final_selection = [
                    x for x in final_selection if self.get_gender_relevancy(x) != 1
                ]
            elif gender == "F":
                final_selection = [
                    x for x in final_selection if self.get_gender_relevancy(x) != 2
                ]
            """else:
                final_selection = [
                    x for x in final_selection if self.get_gender_relevancy(x) == 0
                ]"""
            if final_selection.__len__() != 0:
                f_icd = np.random.choice(final_selection)
                secondary.append(f_icd)

        # print(secondary)
        return secondary
        # print(probs.loc[(probs["Bundesland_ "] == "Berlin"), "männlich_unter 1 Jahr"].sum())

    # get_secondary_list("M", 0, 80339)

    # run only in case it is necessary, took me 2.5 hours to run once (therefore the csv output)
    def calculate_secondary_probabilities(self) -> None:
        """Calculate the ratio of the number of secondary ICD codes compared to the total number of secondary ICDs (sums up to 1)

        Attention: run only in case it is necessary, took me 2.5 hours to run once (therefore the csv output)
        """

        probs = self.read_secondary_icd_statistics()
        state = self.plz.get_state_list()
        ages = utils.get_age_list()
        genders = utils.get_gender_list()
        icds = self.get_secondary_icd_list()
        for s in state:
            for i in icds:
                for a in ages:
                    for g in genders:
                        group = g + "_" + a
                        # probs.loc[(probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()
                        # print(s + "  " + group + "  "  + i + "  " + str((int(probs.loc[(probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s), group].item()))/(int(probs.loc[(probs["ICD_Code_ "] == 'INS') & (probs["Bundesland_ "] == s), group].item()))))
                        probs.loc[
                            (probs["ICD_Code_ "] == i) & (probs["Bundesland_ "] == s),
                            group,
                        ] = (
                            int(
                                probs.loc[
                                    (probs["ICD_Code_ "] == i)
                                    & (probs["Bundesland_ "] == s),
                                    group,
                                ].item()
                            )
                        ) / (
                            int(
                                probs.loc[
                                    (probs["ICD_Code_ "] == "INS")
                                    & (probs["Bundesland_ "] == s),
                                    group,
                                ].item()
                            )
                        )
        probs.to_csv("data/ICD/generated/secondary_probs.csv", index=False)
        print("THE END")

    """def get_primary_probabilities(icd, gender, age, zip):
        stats = read_primary_icd_statistics()
        age_group = get_age_group(age)
        state = self.plz.get_state_from_plz(zip)
        gender_group = get_gender_group(gender)
        column_name = gender_group + "_" + age_group
        icd = icd[:3]"""

    def get_primary_probability(
        self, age: int, gender: str, zip: int, icd: str
    ) -> float:
        """Get the probability of a primary ICD code.

        Get the probability of a primary ICD code by using the calculated probabilities for primary ICDs.

        Args:
            age (int): age of the patient
            gender (str): gender of the patient
            zip (int): plz of the patient
            icd (str): primary ICD code

        Returns:
            a `float` with the probability of the primary ICD code"""

        probs = self.prim_prob_csv
        if probs.empty:
            probs = pd.read_csv("data/ICD/generated/primary_probs.csv")
            self.prim_prob_csv = probs

        state = self.plz.get_state_from_plz(zip)
        age_group = utils.get_age_group(age)
        gender_group = utils.get_gender_group(gender)
        group = gender_group + "_" + age_group

        result = probs.loc[
            (probs["Bundesland_ "] == state) & (probs["ICD_Code_ "] == icd), group
        ]
        return result.item()

    def get_random_primary_icd(self, age: int, gender: str, zip: int) -> str:
        """Get a randomly selected primary ICD code.

        Get a randomly selected primary ICD code by getting a higher level of code using the calculated probabilities for primary ICDs.
        The selected ICD is then compared with code-able ICD codes. The codes with the same start as the selected ones are used
        and a code-able value is selected with linear random selection.

        Args:
            age (int): age of the patient
            gender (str): "M" for male, "F" for female, "D" for anything else
            zip (int): plz of the patient

        Returns:
            a `str` with the selected primary ICD
        """

        probs = self.prim_prob_csv
        if probs.empty:
            probs = pd.read_csv("data/ICD/generated/primary_probs.csv")
            self.prim_prob_csv = probs
        state = self.plz.get_state_from_plz(zip)
        age_group = utils.get_age_group(age)
        gender_group = utils.get_gender_group(gender)
        group = gender_group + "_" + age_group

        result = probs.loc[probs["Bundesland_ "] == state, ["ICD_Code_ ", group]]
        result = result.drop(result.index[-1])
        result[group] = result[group] / np.sum(result[group])
        icd = np.random.choice(result["ICD_Code_ "], p=result[group])
        final = self.get_prim_icd_list()
        final_selection = [x for x in final if x.startswith(icd)]
        if gender == "M":
            final_selection = [
                x for x in final_selection if self.get_gender_relevancy(x) != 1
            ]
        elif gender == "F":
            final_selection = [
                x for x in final_selection if self.get_gender_relevancy(x) != 2
            ]
        """else:
            final_selection = [x for x in final_selection if self.get_gender_relevancy(x) == 0]"""

        if len(final_selection) != 0:
            f_icd = random.choice(final_selection)
        else:
            f_icd = self.get_random_primary_icd(age, gender, zip)
        return f_icd

    """
    def get_random_secondary_icd(self, age, gender, zip, random_state=0):
        # random.seed(random_state)
        state = self.plz.get_state_from_plz(zip)
        age_group = utils.get_age_group(age)
        gender_group = utils.get_gender_group(gender)
        group = gender_group + "_" + age_group
        probs = pd.read_csv("data/ICD/generated/secondary_probs.csv")
        result = probs.loc[probs["Bundesland_ "] == state, ["ICD_Code_ ", group]]
        result = result.drop(result.index[-1])
        result[group] = result[group] / np.sum(result[group])
        icd = np.random.choice(result["ICD_Code_ "], p=result[group])
        final = self.get_icd_list()
        final_selection = [x for x in final if x.startswith(icd)]
        f_icd = random.choice(final_selection)
        return f_icd
    """

    # --- correct according to coding guidelines --- not yet integrated and documented ---


# print(ICD().get_case_number_sum("männlich_unter 1 Jahr"))
# print(ICD().get_gender_relevancy("C61"))
"""icd = ICD()
for i in range(100):
    r = icd.get_random_primary_icd(78, "F", 21368)
    print(str(ICD().get_gender_relevancy(r)) + " " + r)"""
# print(ICD().get_gender_relevancy("D07.4"))
