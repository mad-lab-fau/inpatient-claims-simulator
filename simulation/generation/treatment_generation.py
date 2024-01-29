import sys

sys.path.insert(0, "")

import numpy as np
import pandas as pd
import random
from scipy.stats import powerlaw

import simulation.generation.icd as icds
import simulation.generation.ops as opss
import simulation.generation.adjust_coding as adjust_coding
import simulation.generation.drg as drgs

import simulation.utility.hospitals as hospitalss
import simulation.utility.utils as utils
import simulation.utility.plz as plzs
import simulation.utility.config as config

import cProfile
from pstats import Stats, SortKey


class GENERATION:
    """Class for generation of inpatient claims data. This class is controlling all sub-classes and produces the final output."""

    def __init__(self, plz=None, random_state: int = None) -> None:
        """Constructor of GENERATION class. It starts the entire process of inpatient claims generation by first starting the treatment generation,
        then entering this into the DRG Grouper and using this output to calculate the claims.

        Args:
            plz (PLZ, optional): a PLZ object. Defaults to None.
            random_state (int, optional): random state for reproducibility. Defaults to None.
        """

        if plz is None:
            self.plz = plzs.PLZ(random_state=random_state)
        else:
            self.plz = plz
        self.hospitals = hospitalss.Hospitals()
        self.adjustments = adjust_coding.Adjustments(random_state=random_state)
        self.drg = drgs.DRG(random_state=random_state)
        self.icd = icds.ICD(self.plz, random_state=random_state)
        self.ops = opss.OPS(drg=self.drg, random_state=random_state)
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def generate_patients(self, n_patients: int, state: str = None):
        """Generate all patients used in claims generation.

        Args:
            n_patients (int): number of patients to generate
            state (str): state (str): the name of a German state (e.g. "Bayern"). Defaults to None.

        Returns:
            a `Pandas.DataFrame()` containing all patients and relevant information (age, gender, location).
        """
        patient_id_properties = []

        # Generate patient properties from random distributions
        for p_id in range(n_patients):
            if state is None:
                zip = self.plz.get_random_plz()
            else:
                zip = self.plz.get_random_state_plz(state)
            gender = np.random.choice(
                ["M", "F", "D"], p=[0.479, 0.520, 0.001]
            )  # adjusted to rounded values according to data from 2021
            age = self.icd.get_age(gender=gender)
            patient_id_properties.append([p_id, zip, age, gender])

        patient_profiles_table = pd.DataFrame(
            patient_id_properties, columns=["PATIENT_ID", "PLZ", "AGE", "GENDER"]
        )

        return patient_profiles_table

    def generate_hospitals(self, n_hospitals: int, state: str = None):
        """Generate all hospitals used in claims generation.

        Args:
            n_hospitals (int): number of hospitals to generate
            state (str): state (str): the name of a German state (e.g. "Bayern"). Defaults to None.

        Returns:
            a `Pandas.DataFrame()` containing all hospital IDs and location.
        """

        hospitals_id_properties = []

        # Generate hospital properties from random distributions
        """for hospital_id in range(n_hospitals):
            if state is None:
                zip = self.plz.get_random_plz()
            else:
                zip = self.plz.get_random_state_plz(state)

            hospitals_id_properties.append([hospital_id, zip])"""

        # Generate hospitals from hospital list
        for hospital_id in range(n_hospitals):
            if state is None:
                zip = np.random.choice(
                    self.hospitals.get_hospitals()["Adresse_Postleitzahl_Standort"],
                    p=self.hospitals.get_hospitals()["INSG"],
                )
            else:
                zip = np.random.choice(
                    self.hospitals.get_hospitals_from_state(state)[
                        "Adresse_Postleitzahl_Standort"
                    ],
                    p=self.hospitals.get_hospitals_from_state(state)["INSG"],
                )
            hospitals_id_properties.append([hospital_id, zip])

        hospital_profiles_table = pd.DataFrame(
            hospitals_id_properties, columns=["HOSPITAL_ID", "PLZ"]
        )

        return hospital_profiles_table

    def match_hospital_on_patient(self, patient_plz: int, hospital_profile) -> int:
        """Find a hospital for the patient's treatment.

        Selecting randomly a hospital from the given DataFrame based on distance from the patient.

        Args:
            patient_plz (int): the plz of the patient
            hospital_profile (Pandas.DataFrame()): a table of all hospitals with their respective plz

        Returns:
            an ID of the selected hospital as `int`
        """
        y_hosp, x = hospital_profile.shape

        hospitals = []
        distance = []
        for i in range(y_hosp):
            hospitals.append(i)
            distance.append(
                self.plz.get_distance(patient_plz, hospital_profile["PLZ"][i])
            )

        distance = np.asarray(distance)

        zero_indices = np.where(distance == 0)
        distance[zero_indices] = 1e-10
        weights = 1 / distance
        weights[np.isinf(weights)] = 0

        weights = weights / np.sum(weights)
        # weights = np.nan_to_num(weights) # removing NaN because it occured once
        return np.random.choice(hospitals, p=weights)

    '''def check_overlap(self, arr, id: int, start_date):
        """Check if there is an overlap between the new entry and the existing entries in the DataFrame.

        Args:
            arr (np.array): a numpy array containing all existing entries
            id (int): the ID of the patient
            start_date (str): the start date of the new entry

        Returns:
            a new start date if there is an overlap, else the given start date
        """
        new_start = np.datetime64(start_date)
        df = pd.DataFrame(arr)
        # arr = np.array(arr)
        # if arr.size != 0:
        if not df.empty:
            while True:
                overlap = (
                    df[df.iloc[:, 1] == id]
                    .apply(
                        lambda x: x.iloc[3] <= new_start and x.iloc[4] >= new_start,
                        axis=1,
                    )
                    .any()
                ).any()
                # print(str(arr[0, 1]) + " " + str(arr[0, 3]) + " " + str(arr[0, 4]) + " " + str(new_start))
                # overlap = np.any((arr[0, 1] == id) & (arr[0, 3] <= new_start) & (arr[0, 4] >= new_start))
                # print(str(overlap) + " " + str(start_date) + "  " + str(id))
                # print(np.any(arr[0, 1] == id) & (arr[0, 3] <= new_start))
                if overlap:
                    # If there is an overlap, add 5 days to the start date of the new entry
                    new_start += pd.Timedelta(days=5)
                else:
                    # If there is no overlap, insert the new entry into the DataFrame
                    return new_start
        else:
            return start_date'''

    def generate_treatment_table(
        self,
        patients,
        hospitals,
        n_cases: int,
        start_date: str = "2021-01-01",
        end_date: str = "2021-12-31",
        id_list: list = None,
    ):
        """Generate all treatments for the given patients in the given hospitals.

        Using the patients and hospitals generating the number of treatment cases by calling the respective methods in
        ICD(), OPS(), and DRG() and evaluating the output.

        Args:
            patients (Pandas.DataFrame()): a table containing all relevant patient information
            hospitals (Pandas.DataFrame()): a table containing all relevant hospital information
            start_date (str):  a String indicating a day as starting point of the timeframe possible for treatment in the format YYYY-MM-DD. Defaults to "2021-01-01"
            end_date (str):  a String indicating a day as end point of the timeframe possible for treatment in the format YYYY-MM-DD. Defaults to "2021-12-31"
            n_cases (int): the number of inpatient claims to generate
            id_list (list): a list of treatment IDs to generate claims for. Defaults to None. (used for multiprocessing)

        Returns:
            a `Pandas.DataFrame()` containing all hospital treatments with relevant information (ICDs, OPS, claim, etc.).
        """

        patient_transactions = []

        if id_list is None:
            id_list = list(range(n_cases))

        for n in id_list:
            # y_hosp, x = hospitals.shape
            y_pat, x = patients.shape

            patient_id = np.random.randint(y_pat)
            hospital_id = self.match_hospital_on_patient(
                patient_plz=patients.loc[patient_id]["PLZ"],
                hospital_profile=hospitals,
            )

            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            admission_date = np.random.choice(
                date_range
            )  # self.check_overlap(patient_transactions, patient_id, np.random.choice(date_range))

            # n_icd = np.random.randint(1, 10)
            # n_ops = np.random.randint(1, 10)

            primary_icd = self.icd.get_random_primary_icd(
                patients.loc[patient_id]["AGE"],
                patients.loc[patient_id]["GENDER"],
                patients.loc[patient_id]["PLZ"],
            )

            arr_icd = self.icd.get_secondary_list(
                age=patients.loc[patient_id]["AGE"],
                gender=patients.loc[patient_id]["GENDER"],
                zip=patients.loc[patient_id]["PLZ"],
            )

            """arr_icd = []
            for x in range(0, n_icd):
                new = icd.get_random_secondary_icd(
                    patient_profile.loc[patient_id]["AGE"],
                    patient_profile.loc[patient_id]["GENDER"],
                    patient_profile.loc[patient_id]["PLZ"],
                )
                if new not in arr_icd and new != primary_icd:
                    arr_icd.append(new)"""

            arr_ops = self.ops.get_random_ops_list_v2(
                age=patients.loc[patient_id]["AGE"],
                gender=patients.loc[patient_id]["GENDER"],
                primary=primary_icd,
                secondary=arr_icd,
            )

            """
            arr_ops = []
            for x in range(0, n_ops):
                new = ops.get_random_ops_2(
                    patient_profile.loc[patient_id]["AGE"],
                    patient_profile.loc[patient_id]["GENDER"],
                    primary=primary_icd,
                    secondary=arr_icd
                )
                if new not in arr_ops:
                    arr_ops.append(new)
            """
            breathing_ops = [
                "1-717",
                "8-706",
                "8-71",
                "8-978",
                "8-98d",
                "8-98f",
                "9-501",
            ]
            arr_ops = [x for x in arr_ops if x is not None]
            b = [x for x in arr_ops if any(x.startswith(y) for y in breathing_ops)]
            breathing = not not b

            """if breathing == False:
                breathing_icd = ["T81", "J95", "J96", "P27.8", "Z99.1"]
                b = [x for x in arr_icd if any(x.startswith(y) for y in breathing_icd)]"""

            days = self.drg.get_DRG_params(
                primary=primary_icd,
                secondary=arr_icd,
                ops=arr_ops,
                breathing=breathing,
                date=admission_date,
                state=self.plz.get_state_from_plz(hospitals.loc[hospital_id]["PLZ"]),
            )

            primary_icd, arr_icd, arr_ops, gender = self.adjustments.adjust_coding(
                primary_icd,
                arr_icd,
                breathing,
                arr_ops,
                patients.loc[patient_id]["GENDER"],
                patients.loc[patient_id]["AGE"],
                patients.loc[patient_id]["PLZ"],
            )

            patients.loc[patient_id, "GENDER"] = gender

            if len(arr_icd) > 20:
                arr_icd = arr_icd[:20]

            if len(arr_ops) > 20:
                arr_ops = arr_ops[:20]
            # setting hours of breathing
            if (
                breathing == True and np.random.rand() < 0.6
            ):  # to reduce the number of ventilation cases according to validation data
                if np.random.rand() < 0.15:
                    breathing = random.randint(config.ventilation_to_linear + 1, 5000)
                else:
                    breathing = round(
                        powerlaw.rvs(a=0.35, loc=1, scale=config.ventilation_to_linear)
                    )
            else:
                breathing = 0

            if patients.loc[patient_id]["AGE"] < 1:
                weight = round(
                    random.normalvariate(3289, 500)
                )  # https://www.who.int/tools/child-growth-standards/standards/weight-for-age
            else:
                weight = "n/A"

            discharge_date = admission_date + pd.Timedelta(days=days)
            temp = []
            temp.append(
                [
                    n,
                    patient_id,
                    hospital_id,
                    admission_date,
                    discharge_date,
                    primary_icd,
                ]
            )
            if isinstance(arr_icd, list):
                temp.extend(arr_icd)
            else:
                temp.append(arr_icd)
            temp.extend(["n/A"] * (20 - len(arr_icd)) if len(arr_icd) < 20 else [])

            if isinstance(arr_ops, list):
                temp.extend(arr_ops)
            else:
                temp.append(arr_ops)
            temp.extend(["n/A"] * (20 - len(arr_ops)) if len(arr_ops) < 20 else [])

            temp.extend([days, weight, breathing])
            temp = utils.flatten(temp)
            patient_transactions.append(temp)

        patient_transactions = pd.DataFrame(
            patient_transactions,
            columns=[
                "ID",
                "PATIENT_ID",
                "HOSPITAL_ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
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
                "DURATION",
                "WEIGHT",
                "VENTILATION",
            ],
        )
        return patient_transactions

    def generate_data(
        self,
        n_patients: int,
        n_hospitals: int,
        n_cases: int,
        patients=None,
        hospitals=None,
        id_list: list = None,
        state: str = None,
    ):
        """Generate all relevant data tables.

        This method calls everything else.

        Args:
            n_patients (int): number of patients to generate.
            patients (Pandas.DataFrame()): a table containing all relevant patient information. Defaults to None. (used for multiprocessing)
            n_hospitals (int): number of hospitals to generate.
            hospitals (Pandas.DataFrame()): a table containing all relevant hospital information. Defaults to None. (used for multiprocessing)
            n_cases (int): number of treatments to generate.
            id_list (list): a list of treatment IDs to generate claims for. Defaults to None. (used for multiprocessing)
            state (str): the name of a German state (e.g. "Bayern"). Defaults to None.

        Returns:
            a `Pandas.DataFrame()` containing all relevant claims information.
        """
        if patients is None:
            patient_profiles_table = self.generate_patients(n_patients, state)
        else:
            patient_profiles_table = patients

        # n_hospitals = int(np.random.triangular(1, math.ceil(n_cases/8872), math.ceil(n_cases/8872)*10)) # 8872 is the average number of treatments per hospitals
        if hospitals is None:
            hospitals_table = self.generate_hospitals(n_hospitals, state)
        else:
            hospitals_table = hospitals

        treatment_table = self.generate_treatment_table(
            patient_profiles_table,
            hospitals_table,
            start_date="2021-01-01",
            n_cases=n_cases,
            id_list=id_list,
        )

        ptnt_x_trt_df = treatment_table.set_index("PATIENT_ID").join(
            patient_profiles_table, on="PATIENT_ID"
        )
        full_df = ptnt_x_trt_df.set_index("HOSPITAL_ID").join(
            hospitals_table, on="HOSPITAL_ID", lsuffix="_PAT", rsuffix="_HOSP"
        )

        full_df = full_df.drop_duplicates()
        full_df = full_df.reset_index(drop=True)

        distance = []

        for i in range(len(full_df)):
            distance.append(
                self.plz.get_distance(
                    full_df["PLZ_PAT"][i].item(), full_df["PLZ_HOSP"][i].item()
                )
            )

        full_df["DISTANCE"] = distance
        full_df["POTENTIAL_FRAUD"] = np.zeros(len(full_df), dtype=int)
        full_df["FRAUD_ID"] = np.zeros(len(full_df), dtype=int)
        if (
            patients is None and hospitals is None
        ):  # meaning there is no multiprocessing in progress
            full_df.to_csv("data/generated data/claims.csv", index=False)
        return full_df


"""    
with cProfile.Profile() as pr:
    GENERATION().generate_data(n_patients=1000, n_hospitals=5, n_cases=500, state="Bayern", random_state=5).to_csv('data/generated data/claims.csv', index=False)
with open('profiling/profiling_stats_1.txt', 'w') as stream:
    stats = Stats(pr, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.dump_stats('profiling/profile_8.prof')
    stats.print_stats()    
"""
