import sys

sys.path.insert(0, "")

import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import subprocess
import os

import simulation.utility.config as config
import simulation.utility.utils as utils


class Grouper:
    """Grouper is used for creating the necessary format for the DRG Grouper, executing the grouper, and merging the results in the existing table."""

    def __init__(self):
        """Constructor of the Grouper class."""
        try:
            self.claims = pd.read_csv("data/generated data/claims.csv")
        except EmptyDataError:
            self.claims = pd.DataFrame()
        self.comb = pd.DataFrame()

    def prepare_grouper(self):
        """This method is reading the generated claims file and creating the file format required by the grouper."""
        claims = pd.read_csv("data/generated data/claims_with_fraud.csv")

        columns = ["ICD_" + str(i) for i in range(1, 21)]

        results = claims[columns].values.tolist()
        results = [utils.strip_nA(x) for x in results]
        results = ["~".join(x) for x in results]
        claims["ICD"] = claims["PRIMARY_ICD"] + "~" + results
        claims = claims.drop(["PRIMARY_ICD"], axis=1)
        claims = claims.drop(columns, axis=1)

        columns = ["OPS_" + str(i) for i in range(1, 21)]
        results = claims[columns].values.tolist()
        results = [utils.strip_nA(x) for x in results]
        results = ["~".join(x) for x in results]
        claims["OPS"] = results
        claims = claims.drop(columns, axis=1)

        claims["ADMISSION_DATE"] = pd.to_datetime(claims["ADMISSION_DATE"])
        claims["BIRTHDAY"] = claims.apply(
            lambda x: x["ADMISSION_DATE"] - pd.DateOffset(years=x["AGE"]), axis=1
        )
        claims["BIRTHDAY"] = claims["BIRTHDAY"].dt.strftime("%Y%m%d")

        claims["ADMISSION_DATE"] = claims["ADMISSION_DATE"].dt.strftime("%Y%m%d")
        claims["DISCHARGE_DATE"] = pd.to_datetime(claims["DISCHARGE_DATE"]).dt.strftime(
            "%Y%m%d"
        )

        claims["ADMISSION_REASON"] = np.full(len(results), "E")
        claims["ADMISSION_CAUSE"] = np.full(len(results), "01")
        claims["DISCHARGE_CAUSE"] = np.full(len(results), "01")
        claims["AGE_D"] = np.zeros(len(results), dtype=int)
        claims["GENDER"] = claims["GENDER"].replace({"M": 1, "F": 2, "D": 3})
        claims["VACATION"] = np.zeros(len(results), dtype=int)
        claims["TYPE"] = np.full(len(results), 1)

        department = np.char.add(
            np.full(len(results), "0001"), np.full(len(results), "&")
        )
        department = np.char.add(department, claims["ADMISSION_DATE"])
        department = np.char.add(department, np.full(len(results), "&"))
        department = np.char.add(department, claims["DISCHARGE_DATE"])
        claims["DEPARTMENT"] = department

        claims = claims[
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DURATION",
                "TYPE",
            ]
        ]

        claims.to_csv(
            "data/generated data/grouper data/input.txt",
            sep="|",
            index=False,
            header=None,
        )

    def prepare_interim_grouper(self):
        """This method is reading the generated claims file and creating the file format required by the grouper."""
        claims = self.claims

        columns = ["ICD_" + str(i) for i in range(1, 21)]

        results = claims[columns].values.tolist()
        results = [utils.strip_nA(x) for x in results]
        results = ["~".join(x) for x in results]
        claims["ICD"] = claims["PRIMARY_ICD"] + "~" + results
        claims = claims.drop(["PRIMARY_ICD"], axis=1)
        claims = claims.drop(columns, axis=1)

        columns = ["OPS_" + str(i) for i in range(1, 21)]
        results = claims[columns].values.tolist()
        results = [utils.strip_nA(x) for x in results]
        results = ["~".join(x) for x in results]
        claims["OPS"] = results
        claims = claims.drop(columns, axis=1)

        claims["ADMISSION_DATE"] = pd.to_datetime(claims["ADMISSION_DATE"])
        claims["BIRTHDAY"] = claims.apply(
            lambda x: x["ADMISSION_DATE"] - pd.DateOffset(years=x["AGE"]), axis=1
        )
        claims["BIRTHDAY"] = claims["BIRTHDAY"].dt.strftime("%Y%m%d")

        claims["ADMISSION_DATE"] = claims["ADMISSION_DATE"].dt.strftime("%Y%m%d")
        claims["DISCHARGE_DATE"] = pd.to_datetime(claims["DISCHARGE_DATE"]).dt.strftime(
            "%Y%m%d"
        )

        claims["ADMISSION_REASON"] = np.full(len(results), "E")
        claims["ADMISSION_CAUSE"] = np.full(len(results), "01")
        claims["DISCHARGE_CAUSE"] = np.full(len(results), "01")
        claims["AGE_D"] = np.zeros(len(results), dtype=int)
        claims["GENDER"] = claims["GENDER"].replace({"M": 1, "F": 2, "D": 3})
        claims["VACATION"] = np.zeros(len(results), dtype=int)
        claims["TYPE"] = np.full(len(results), 1)

        department = np.char.add(
            np.full(len(results), "0001"), np.full(len(results), "&")
        )
        department = np.char.add(department, claims["ADMISSION_DATE"])
        department = np.char.add(department, np.full(len(results), "&"))
        department = np.char.add(department, claims["DISCHARGE_DATE"])
        claims["DEPARTMENT"] = department

        claims = claims[
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DURATION",
                "TYPE",
            ]
        ]

        claims.to_csv(
            "data/generated data/grouper data/input.txt",
            sep="|",
            index=False,
            header=None,
        )

    def prepare_negative_list_grouper(self, combinations):
        combinations["HOSPITAL_ID"] = np.full(len(combinations), 1)
        combinations["ID"] = np.full(len(combinations), 1)
        combinations["ADMISSION_DATE"] = np.full(len(combinations), 20210101)
        combinations["DISCHARGE_DATE"] = np.full(len(combinations), 20210127)
        combinations["ADMISSION_REASON"] = np.full(len(combinations), "E")
        combinations["ADMISSION_CAUSE"] = np.full(len(combinations), "01")
        combinations["DISCHARGE_CAUSE"] = np.full(len(combinations), "01")
        combinations["BIRTHDAY"] = np.full(len(combinations), 19500101)
        combinations["WEIGHT"] = np.full(len(combinations), 0)
        combinations["AGE"] = np.full(len(combinations), 70)
        combinations["AGE_D"] = np.full(len(combinations), 0)
        combinations["GENDER"] = np.full(len(combinations), 1)
        combinations["VENTILATION"] = np.full(len(combinations), 0)
        combinations["DEPARTMENT"] = np.full(
            len(combinations), "0001&20210101&20210101"
        )
        combinations["DURATION"] = np.full(len(combinations), 26)
        combinations["VACATION"] = np.full(len(combinations), 0)
        combinations["TYPE"] = np.full(len(combinations), 1)

        claims = combinations[
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DURATION",
                "TYPE",
            ]
        ]

        claims.to_csv(
            "data/generated data/grouper data/input.txt",
            sep="|",
            index=False,
            header=None,
        )

    def prepare_fraud_grouper(self, id: int, combinations: list):
        """This method is creating the combinations dataframe in the format required by the grouper and filling unnecessary columns with dummy values.

        Args:
            id (int): the id of the patient
            combinations (list): the list of combinations to be created"""

        # claims = pd.read_csv("data/generated data/claims.csv")
        claims = self.claims
        claims = claims[claims["ID"] == id]
        claims = pd.concat([claims] * len(combinations), ignore_index=True)
        # print(claims)
        columns = ["ICD_" + str(i) for i in range(1, 21)]

        results = combinations
        # results = [utils.strip_nA(x) for x in results]
        results = ["~".join(map(str, x)) for x in results]
        claims["ICD"] = results  # claims["PRIMARY_ICD"] + "~" + results
        claims = claims.drop(["PRIMARY_ICD"], axis=1)
        claims = claims.drop(columns, axis=1)

        columns = ["OPS_" + str(i) for i in range(1, 21)]
        results = claims[columns].values.tolist()
        results = [utils.strip_nA(x) for x in results]
        results = ["~".join(map(str, x)) for x in results]
        claims["OPS"] = results
        claims = claims.drop(columns, axis=1)

        claims["ADMISSION_DATE"] = pd.to_datetime(claims["ADMISSION_DATE"])
        claims["BIRTHDAY"] = claims.apply(
            lambda x: x["ADMISSION_DATE"] - pd.DateOffset(years=x["AGE"]), axis=1
        )
        claims["BIRTHDAY"] = claims["BIRTHDAY"].dt.strftime("%Y%m%d")

        claims["ADMISSION_DATE"] = claims["ADMISSION_DATE"].dt.strftime("%Y%m%d")
        claims["DISCHARGE_DATE"] = pd.to_datetime(claims["DISCHARGE_DATE"]).dt.strftime(
            "%Y%m%d"
        )

        claims["ADMISSION_REASON"] = np.full(len(results), "E")
        claims["ADMISSION_CAUSE"] = np.full(len(results), "01")
        claims["DISCHARGE_CAUSE"] = np.full(len(results), "01")
        claims["AGE_D"] = np.zeros(len(results), dtype=int)
        claims["GENDER"] = claims["GENDER"].replace({"M": 1, "F": 2, "D": 3})
        claims["VACATION"] = np.zeros(len(results), dtype=int)
        claims["TYPE"] = np.full(len(results), 1)

        department = np.char.add(
            np.full(len(results), "0001"), np.full(len(results), "&")
        )
        department = np.char.add(department, claims["ADMISSION_DATE"])
        department = np.char.add(department, np.full(len(results), "&"))
        department = np.char.add(department, claims["DISCHARGE_DATE"])
        claims["DEPARTMENT"] = department

        claims = claims[
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DURATION",
                "TYPE",
            ]
        ]
        # self.comb = self.comb.append(claims)
        self.comb = pd.concat([self.comb, claims], axis=0, ignore_index=True)
        # print(self.comb)
        # claims.to_csv("data/generated data/grouper data/input.txt", sep='|', mode="a", index=False, header=None)

    def run_grouper(self):
        """This method uses the created file and executes the grouper using powershell.

        - "powershell.exe", "&", config.grouper + "Batchgrouper.exe'" is executing the Batchgrouper.exe in the installation directory.
        - "DRG" sets the grouper to DRG (instead of PEPP)
        - "21/21" sets the year of data and the year of the grouping guidelines
        - config.base_path + "/data/generated data/grouper data/input.txt'" points to the created input file
        - config.base_path + "/data/generated data/grouper data/output.txt'" points to the newly created output file
        - "true" to calculate effective weights
        - value lowerings for certain ICDs has been ignored
        """
        # sleep(5) # inserted due to read/write-conflicts in grouper
        subprocess.run(
            [
                "powershell.exe",
                "&",
                config.grouper + "Batchgrouper.exe'",
                "DRG",
                "21/21",
                '"'
                + str(
                    os.path.abspath(
                        os.path.join(
                            os.getcwd(),
                            "data",
                            "generated data",
                            "grouper data",
                            "input.txt",
                        )
                    )
                )
                + '"',
                '"'
                + str(
                    os.path.abspath(
                        os.path.join(
                            os.getcwd(),
                            "data",
                            "generated data",
                            "grouper data",
                            "output.txt",
                        )
                    )
                )
                + '"',
                "true",
            ]
        )
        open(
            "data/generated data/grouper data/input.txt", "w"
        ).close()  # clear input.csv after each run

    def merge_data(self):
        """Assigning each treatment the according DRG."""
        # sleep(5) # inserted due to read/write-conflicts in grouper
        output = pd.read_csv(
            "data\generated data\grouper data\output.txt", sep="|", header=None
        )
        output = output.set_axis(
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DRG",
                "MDC",
                "PCCL",
                "STATUS",
                "ERROR_INFO",
                "REL_WEIGHT",
                "DEPARTMENT_ACT",
                "DAYS_SHORT",
                "DAYS_LONG",
                "DAYS_REDUCED",
                "EFF_WEIGHT",
            ],
            axis=1,
        )
        output = output[["ID", "DRG"]]
        # output = output.set_index("ID")

        claims = pd.read_csv("data\generated data\claims_with_fraud.csv")

        new = pd.merge(claims, output, on="ID")
        new.to_csv("data/generated data/claims_with_drg.csv", index=False)

    def merge_interim_data(self):
        """Assigning each treatment the according DRG."""
        # sleep(5) # inserted due to read/write-conflicts in grouper
        output = pd.read_csv(
            "data\generated data\grouper data\output.txt", sep="|", header=None
        )
        output = output.set_axis(
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DRG",
                "MDC",
                "PCCL",
                "STATUS",
                "ERROR_INFO",
                "REL_WEIGHT",
                "DEPARTMENT_ACT",
                "DAYS_SHORT",
                "DAYS_LONG",
                "DAYS_REDUCED",
                "EFF_WEIGHT",
            ],
            axis=1,
        )
        output = output[["ID", "DRG"]]
        # output = output.set_index("ID")

        claims = pd.read_csv("data\generated data\claims.csv")

        new = pd.merge(claims, output, on="ID")
        new.to_csv("data/generated data/claims_with_drg.csv", index=False)

    def merge_negative_list_data(self, combinations):
        output = pd.read_csv(
            "data\generated data\grouper data\output.txt",
            sep="|",
            header=None,
            low_memory=False,
        )
        output = output.set_axis(
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DRG",
                "MDC",
                "PCCL",
                "STATUS",
                "ERROR_INFO",
                "REL_WEIGHT",
                "DEPARTMENT_ACT",
                "DAYS_SHORT",
                "DAYS_LONG",
                "DAYS_REDUCED",
                "EFF_WEIGHT",
            ],
            axis=1,
        )

        # merge with combinations with concat
        output = output["DRG"]
        output.reset_index()
        combinations.reset_index(inplace=True)
        output = pd.concat([output, combinations], axis=1, ignore_index=True)
        return output

    def execute_grouping(self):
        """Executing preparation, execution, and merging in the correct order."""
        self.prepare_grouper()
        self.run_grouper()
        self.merge_data()

    def execute_interim_grouping(self):
        """Executing preparation, execution, and merging in the correct order for grouping the claims file interim-wise."""
        self.prepare_interim_grouper()
        self.run_grouper()
        self.merge_interim_data()

    def execute_upcoding_grouping(self):
        """Executing preparation, execution, and merging in the correct order."""
        # os.remove("data\generated data\grouper data\input.txt")
        # self.prepare_fraud_grouper(id, combinations)
        # self.claims = self.claims[["HOSPITAL_ID", "ID", "ADMISSION_DATE", "DISCHARGE_DATE", "ADMISSION_REASON", "ADMISSION_CAUSE", "DISCHARGE_CAUSE", "BIRTHDAY", "WEIGHT", "AGE", "AGE_D", "GENDER", "VENTILATION", "ICD", "OPS","DEPARTMENT", "DURATION", "VACATION", "DURATION", "TYPE"]]
        self.comb.to_csv(
            "data/generated data/grouper data/input.txt",
            sep="|",
            index=False,
            header=None,
        )

        self.run_grouper()
        try:
            output = pd.read_csv(
                "data/generated data/grouper data/output.txt", sep="|", header=None
            )
            output = output.set_axis(
                [
                    "HOSPITAL_ID",
                    "ID",
                    "ADMISSION_DATE",
                    "DISCHARGE_DATE",
                    "ADMISSION_REASON",
                    "ADMISSION_CAUSE",
                    "DISCHARGE_CAUSE",
                    "BIRTHDAY",
                    "WEIGHT",
                    "AGE",
                    "AGE_D",
                    "GENDER",
                    "VENTILATION",
                    "ICD",
                    "OPS",
                    "DEPARTMENT",
                    "DURATION",
                    "VACATION",
                    "DRG",
                    "MDC",
                    "PCCL",
                    "STATUS",
                    "ERROR_INFO",
                    "REL_WEIGHT",
                    "DEPARTMENT_ACT",
                    "DAYS_SHORT",
                    "DAYS_LONG",
                    "DAYS_REDUCED",
                    "EFF_WEIGHT",
                ],
                axis=1,
            )
            output = output[["ID", "DRG", "EFF_WEIGHT"]]
            output["EFF_WEIGHT"] = output["EFF_WEIGHT"].str.replace(",", ".")
            output["EFF_WEIGHT"] = pd.to_numeric(output["EFF_WEIGHT"])
        except pd.errors.EmptyDataError:
            print("No output file created")
            output = pd.DataFrame()

        # print (output)
        return output

    def execute_negative_list_grouping(self, combinations):
        self.prepare_negative_list_grouper(combinations)
        self.run_grouper()
        return self.merge_negative_list_data(combinations)

    def check_grouping(self):
        """Checking the grouper status for irregularities (!= 0)."""
        output = pd.read_csv(
            "data/generated data/grouper data/output.txt", sep="|", header=None
        )
        output = output.set_axis(
            [
                "HOSPITAL_ID",
                "ID",
                "ADMISSION_DATE",
                "DISCHARGE_DATE",
                "ADMISSION_REASON",
                "ADMISSION_CAUSE",
                "DISCHARGE_CAUSE",
                "BIRTHDAY",
                "WEIGHT",
                "AGE",
                "AGE_D",
                "GENDER",
                "VENTILATION",
                "ICD",
                "OPS",
                "DEPARTMENT",
                "DURATION",
                "VACATION",
                "DRG",
                "MDC",
                "PCCL",
                "STATUS",
                "ERROR_INFO",
                "REL_WEIGHT",
                "DEPARTMENT_ACT",
                "DAYS_SHORT",
                "DAYS_LONG",
                "DAYS_REDUCED",
                "EFF_WEIGHT",
            ],
            axis=1,
        )

        print(output[output["STATUS"] != 0])


# Grouper().prepare_fraud_grouper(1, [['I21.0', 'J18.9', 'I10.00', 'I50.13', 'A08.0'], ['J18.9', 'I21.0', 'I10.00', 'I50.13', 'A08.0'], ['I10.00', 'I21.0', 'J18.9', 'I50.13', 'A08.0'], ['I50.13', 'I21.0', 'J18.9', 'I10.00', 'A08.0'], ['A08.0', 'I21.0', 'J18.9', 'I10.00', 'I50.13']])
# print(Grouper().execute_fraud_grouping(1, ['I21.0', 'J18.9', 'I10.00', 'I50.13', 'A08.0']))
# print(str(os.path.abspath(os.path.join(os.getcwd(), "data", "generated data", "grouper data", "input.txt"))))
# Grouper().run_grouper()
