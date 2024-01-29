import sys

sys.path.insert(0, "")

import random

import numpy as np
import simulation.generation.icd as icd
import simulation.generation.ops as ops
import simulation.generation.drg as drg
import simulation.utility.config as config


class Adjustments:
    def __init__(self, random_state=None):
        """Adjustments is a class that contains all adjustments to the coding according to the coding guidelines."""
        self.icd = icd.ICD()
        self.ops = ops.OPS(drg.DRG())

        if random_state is not None:
            random.seed(random_state)

    def adjust_coding(
        self,
        primary: str,
        secondary: list,
        pump: bool,
        operations: list,
        gender: str,
        age: int,
        zip: int,
    ) -> tuple:
        """Adjust the coding according to the coding guidelines.

        Adjust the coding according to the coding guidelines. The coding guidelines are based on the OPS and ICD coding guidelines of the German Institute for Medical Documentation and Information (DIMDI). The coding guidelines are implemented as a series of if-statements.

        Args:
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes
            pump (bool): True if a pump is used, False otherwise
            operations (list): list of OPS codes
            gender (str): gender of the patient
            age (int): age of the patient
            zip (int): plz of the patient

        Returns:
            a `tuple` of the adjusted primary ICD code, the adjusted list of secondary ICD codes, and the adjusted list of OPS codes
        """
        primary, secondary = self.check_validity_HIV(primary, secondary)
        secondary, operations = self.check_zytotoxic_materials(secondary, operations)
        secondary = self.check_folic_acid(primary, secondary)
        secondary = self.add_probable_diabetic_comorbidities(primary, secondary)
        primary, secondary = self.no_metabolic_when_diabetes(
            primary=primary,
            secondary=secondary,
            gender=gender,
            age=age,
            zip=zip,
        )
        operations = self.add_ops_on_pump(pump=pump, operations=operations)
        operations = self.add_ops_on_revisioned_heart(operations=operations)
        secondary = self.add_relevant_icd_birth(primary=primary, secondary=secondary)
        secondary = self.add_relevant_icd_babies_hie(
            primary=primary, secondary=secondary
        )
        operations = self.correct_sectio(age, gender, primary, secondary, operations)
        primary = self.check_frustrane(primary)
        primary = self.replace_old_myocard_infarct(primary)
        primary = self.replace_fluid_surpluss(primary)
        primary = self.check_Z01_secondary(primary)
        primary = self.replace_unknown_examination(primary)
        primary = self.replace_senile_degeneration(primary)
        primary = self.replace_chemo(primary)
        primary = self.correct_lymphom(primary)
        gender = self.change_gender_for_pregnancy(gender, primary, secondary)
        primary = self.replace_caused_by_other_in_primary(primary, age, gender, zip)
        primary, secondary = self.adjust_carcinoma_to_gender(gender, primary, secondary)
        primary, secondary = self.adjust_kidney_secondary(gender, primary, secondary)
        primary = self.change_companion(primary)
        primary = self.change_fetus_abortus(primary)

        return primary, secondary, operations, gender

    def check_validity_HIV(self, primary: str, secondary: list) -> tuple:
        """Check the validity of HIV codes.

        Check the validity of HIV codes. If HIV is the primary diagnose, it is replaced by B24. If HIV is a secondary diagnose, it is removed.

        Args:
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes

        Returns:
            a `tuple` of the adjusted primary ICD code and the adjusted list of secondary ICD codes
        """
        if primary == "R75":
            primary = "B24"

        excluded_codes_HIV = ["R75", "Z21", "B230", "B20", "B21", "B22", "B23", "B24"]
        found_1 = False

        for d in secondary:
            if d in excluded_codes_HIV and found_1 == False:
                found_1 = True
            elif d in excluded_codes_HIV and found_1 == True:
                secondary.remove(d)

        return primary, secondary

    def check_zytotoxic_materials(self, secondary: str, operations: list):
        """Check the validity of zytotoxic materials.

        Check the validity of zytotoxic materials. If 8-541.4 is used, it is removed.

        Args:
            secondary (list): list of secondary ICD codes
            operations (list): list of OPS codes

        Returns:
            a `tuple` of the adjusted list of secondary ICD codes and the adjusted list of OPS codes
        """
        # 0212d
        found_1 = False
        for d in operations:
            if d == "8-541.4" and found_1 == False:
                found_1 = True
            elif d == "8-541.4" and found_1 == True:
                operations.remove(d)

        # 0212d
        for d in secondary:
            if d == "Z51.1":
                secondary.remove(d)

        return secondary, operations

    def check_folic_acid(self, primary: str, secondary: list) -> list:
        """Check the validity of folic acid.

        Check the validity of folic acid. If D52.0, D52.1, D52.8, or D52.9 is used, E53.8 is removed.

        Args:
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes

        Returns:
            a `list` of the adjusted list of secondary ICD codes
        """
        # 0305u
        excluded_codes_folic = ["D52.0", "D52.1", "D52.8", "D52.9"]
        found_1 = False
        if primary in excluded_codes_folic:
            found_1 = True
        else:
            for d in secondary:
                if d in excluded_codes_folic and found_1 == False:
                    found_1 = True

        if found_1 == True:
            try:
                secondary.remove("E53.8")
            except:
                pass

        return secondary

    def add_probable_diabetic_comorbidities(
        self, primary: str, secondary: list, rand_num=None
    ):
        """Add probable diabetic comorbidities.

        Add probable diabetic comorbidities. If diabetes is the primary diagnose or a comorbidity, a comorbidity is added with a probability of 0.4.

        Args:
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes
            rand_num (float): random number to be used for the probability of adding a comorbidity

        Returns:
            a `list` of the adjusted list of secondary ICD codes"""
        # 0401
        relevant_icd = [
            "E10.74",
            "E10.75",
            "E11.74",
            "E11.75",
            "E12.74",
            "E12.75",
            "E13.74",
            "E13.75",
            "E14.74",
            "E14.75",
        ]
        possible_icd = [
            "L02.4",
            "L03.02",
            "L03.11",
            "L89.01",
            "L89.02",
            "L89.03",
            "L89.10",
            "L89.11",
            "L89.22",
            "L89.20",
            "L89.21",
            "L89.30",
            "L89.31",
            "L89.90",
            "L97",
            "I70.20",
            "I70.21",
            "I70.22",
            "I70.23",
            "M20.1",
            "M20.2",
            "M20.5",
            "M21.27",
        ]  # not all possible comorbidities mentioned here

        has_diabetes = False
        if primary in relevant_icd:
            has_diabetes = True
        else:
            for d in secondary:
                if d in relevant_icd:
                    has_diabetes = True

        if rand_num is None:
            rand_num = random.random()

        if has_diabetes and rand_num < config.probability_diabetic_comorbidity:
            secondary.extend([random.choice(possible_icd)])

        return secondary

    def no_metabolic_when_diabetes(
        self, primary: str, secondary: list, gender: str, age: int, zip: int
    ) -> tuple:
        """Check the validity of metabolic.

        Check the validity of metabolic. If diabetes is the primary diagnose or a comorbidity, no metabolic is added.

        Args:
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes
            gender (str): gender of the patient
            age (int): age of the patient
            zip (int): plz of the patient

        Returns:
            a `tuple` of the adjusted primary ICD code and the adjusted list of secondary ICD codes
        """
        # 0403
        excluded_codes_diabetes = ["E16.0", "E16.1", "E16.2", "E16.8", "E16.9"]

        diabetes = False

        for d in secondary:
            if d.startswith(("E10", "E11", "E12", "E13", "E14")):
                diabetes = True

        if diabetes:
            if primary in excluded_codes_diabetes:
                primary = self.icd.get_random_primary_icd(
                    gender=gender, age=age, zip=zip
                )
                self.no_metabolic_when_diabetes(
                    primary, secondary, gender=gender, age=age, zip=zip
                )

        return primary, secondary

    def add_ops_on_pump(self, pump: bool, operations: list, rand_choice=None) -> list:
        """Add operations on pump.

        Add operations on pump. If a pump is used, "8-851.4" or "8-851.5" is added.

        Args:
            pump (bool): True if a pump is used, False otherwise
            operations (list): list of OPS codes
            rand_choice (str): random choice of OPS code to be used

        Returns:
            a `list` of the adjusted list of OPS codes
        """
        # 0908l
        if pump > 0 and not ("8-851.4" in operations or "8-851.5" in operations):
            if rand_choice is None:
                rand_choice = random.choice(["8-851.4", "8-851.5"])
            operations.extend([rand_choice])

        return operations

    def add_ops_on_revisioned_heart(self, operations: list) -> list:
        """Add operations on revisioned heart.

        Add operations on revisioned heart. If a heart is revisioned, 5-379.5 is added.

        Args:
            operations (list): list of OPS codes

        Returns:
            a `list` of the adjusted list of OPS codes
        """
        # 0909d
        if "5-352.00" in operations and not "5-379.5" in operations:
            operations.extend(["5-379.5"])
        return operations

    def add_relevant_icd_birth(self, primary: str, secondary: list) -> list:
        """Add relevant ICD codes for birth.

        Add relevant ICD codes for birth. If a birth is the primary diagnose, one of Z37.0-9 is added.

        Args:
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes

        Returns:
            a `list` of the adjusted list of secondary ICD codes
        """
        # 1507e
        pregnant = False
        duration = False

        if primary in ["O80", "O81", "O82"]:
            pregnant = True
            for d in secondary:
                if d.startswith("Z37"):
                    duration = True
        else:
            for d in secondary:
                if d in ["O80", "O81", "O82"]:
                    pregnant = True
                if d.startswith("Z37"):
                    duration = True

        if pregnant and not duration:
            secondary.extend(
                [
                    random.choice(
                        [
                            "Z37.0",
                            "Z37.1",
                            "Z37.2",
                            "Z37.3",
                            "Z37.4",
                            "Z37.5",
                            "Z37.5",
                            "Z37.6",
                            "Z37.8",
                            "Z37.9",
                        ]
                    )
                ]
            )

        return secondary

    def add_relevant_icd_babies_hie(self, primary: str, secondary: list) -> list:
        """Add relevant ICD codes for babies.

        Add relevant ICD codes for babies. If a baby is the primary diagnose, P90 and one of P21.0, P20.0, P20.1, and P20.9 is added.

        Args:
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes

        Returns:
            a `list` of the adjusted list of secondary ICD codes
        """
        # 1606e
        # assumption: P90 is always present
        if "P91.3" in secondary or "P91.3" == primary:
            secondary.extend([random.choice(["P21.0", "P20.0", "P20.1", "P20.9"])])
        elif "P91.4" in secondary or "P91.4" == primary:
            secondary.extend(
                ["P90", random.choice(["P21.0", "P20.0", "P20.1", "P20.9"])]
            )
        elif "P91.5" in secondary or "P91.5" == primary:
            secondary.extend(
                ["P90", random.choice(["P21.0", "P20.0", "P20.1", "P20.9"])]
            )

        return secondary

    def correct_sectio(
        self, age: int, gender: str, primary: str, secondary: list, operations: list
    ) -> list:
        """Correct the coding of a sectio.

        Correct the coding of a sectio. If a sectio is the primary or secondary diagnose, an OPS code of the 5-74 family is added.

        Args:
            age (int): age of the patient
            gender(str): gender of the patient
            primary (str): primary ICD code
            secondary (list): list of secondary ICD codes
            operations (list): list of OPS codes

        Returns:
            a `list` of the adjusted list of OPS codes"""

        # 1525j
        if (primary == "O82" or "O82" in secondary) and not any(
            value.startswith("5-74") for value in operations
        ):
            # print("adjusting: " + str(operations))
            operations.append(self.ops.get_random_ops_from_major(age, gender, "5-74"))
            # print("adjusted: " + str(operations))

        return operations

    def check_frustrane(self, primary: str) -> str:
        """Check the validity of frustrane contractions.

        Check the validity. If frustrane contractions is the primary diagnose, it is replaced by O47.0 or O47.1.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code
        """
        # not found
        if primary == "O47.9":
            primary = random.choice(
                ["O47.0", "O47.1"]
            )  # .9 means not specified, therefore unqualified for primary diagnose; .0 less than 37 weeks, .1 more than 37 weeks
        return primary

    def replace_old_myocard_infarct(self, primary: str) -> str:
        """Replace old myocard infarct with other form of ischemic heart disease.

        Replace old myocard infarct with other form of ischemic heart disease. If I25.2 is the primary diagnose, it is replaced by I25.8.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""

        # 0901f
        if primary.startswith("I25.2"):
            primary = "I25.8"

        return primary

    def replace_fluid_surpluss(self, primary: str) -> str:
        """Replace fluid surpluss with other form of fluid disorder.

        Replace fluid surpluss with other form of fluid disorder. If E87.7 is the primary diagnose, it is replaced by E87.0, E87.1, E87.2, E87.3, E87.4, E87.5, E87.6, or E87.8.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code
        """
        if primary == "E87.7":
            primary = random.choice(
                ["E87.0", "E87.1", "E87.2", "E87.3", "E87.4", "E87.5", "E87.6", "E87.8"]
            )
        return primary

    def check_Z01_secondary(self, primary: str) -> str:
        """Check the validity of Z01 secondary.

        Check the validity of Z01 secondary. If Z01.6, Z01.7, or Z01.9 is the primary diagnose, it is replaced by Z01.1, Z01.2, Z01.3, Z01.4, Z01.5, Z01.80, Z01.81, or Z01.88.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""

        if primary in ["Z01.6", "Z01.7", "Z01.9"]:
            primary = random.choice(
                [
                    "Z01.1",
                    "Z01.2",
                    "Z01.3",
                    "Z01.4",
                    "Z01.5",
                    "Z01.80",
                    "Z01.81",
                    "Z01.88",
                ]
            )
        return primary

    def replace_unknown_examination(self, primary: str) -> str:
        """Replace unknown examination with other form of examination.

        Replace unknown examination with other form of examination. If Z04.9 is the primary diagnose, it is replaced by Z04.1, Z04.2, Z04.3, Z04.5, or Z04.8.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""
        if primary == "Z04.9":
            primary = random.choice(["Z04.1", "Z04.2", "Z04.3", "Z04.5", "Z04.8"])

        return primary

    def replace_senile_degeneration(self, primary: str) -> str:
        """Replace senile degeneration with other form of senile degeneration.

        Replace senile degeneration with other form of senile degeneration. If G31.1 is the primary diagnose, it is replaced by G31.9.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""

        if primary == "G31.1":
            primary = "G31.9"

        return primary

    def replace_chemo(self, primary: str) -> str:
        """Replace chemo with other form of chemo.

        Replace chemo with other form of chemo. If Z51.1 is the primary diagnose, it is replaced by Z51.0, Z51.2, Z51.3, Z51.4, Z51.5, Z51.6, Z51.7, Z51.8, or Z51.9.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""

        # 0212d (partly)
        if primary.startswith("Z51") and primary not in [
            "Z51.4",
            "Z51.6",
            "Z51.83",
            "Z51.88",
            "Z51.9",
        ]:
            primary = random.choice(["Z51.4", "Z51.6", "Z51.83", "Z51.88", "Z51.9"])
        return primary

    def correct_lymphom(self, primary: str) -> str:
        """Correct lymphom with other form of lymphom.

        Correct lymphom with other form of lymphom. If C79.6 is the primary diagnose, it is replaced by C79.0, C79.1, C79.2, C79.3, C79.4, C79.5, C79.7, C79.81, C79.82, C79.83, C79.84, C79.85, C79.86, C79.88, or C79.9.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""

        # 0215q (partly)

        if primary == "C79.6":
            primary = random.choice(
                [
                    "C79.0",
                    "C79.1",
                    "C79.2",
                    "C79.3",
                    "C79.4",
                    "C79.5",
                    "C79.7",
                    "C79.81",
                    "C79.82",
                    "C79.83",
                    "C79.84",
                    "C79.85",
                    "C79.86",
                    "C79.88",
                    "C79.9",
                ]
            )

        return primary

    def change_gender_for_pregnancy(
        self, gender: str, primary: str, secondary: list
    ) -> str:
        """If patient is pregnant, change gender to female (otherwise grouping will fail).

        Args:
            - gender (str): gender of the patient
            - primary (str): primary ICD code
            - secondary (list): list of secondary ICD codes

        Returns:
            a `str` of the adjusted gender"""

        if (
            primary.startswith("O") or any(x.startswith("O") for x in secondary)
        ) and gender != "F":
            gender = "F"
        return gender

    def replace_caused_by_other_in_primary(
        self, primary: str, age: int, gender: str, plz: int
    ) -> str:
        """Replace ICD indicating disease caused by other disease in primary.

        If the primary diagnose is caused by another diagnose, it is replaced by the other diagnose.

        Args:
            primary (str): primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""

        # 0212d
        while primary.startswith(("B9", "E64", "E68", "G09", "I69", "O94", "T9")):
            primary = self.icd.get_random_primary_icd(age, gender, plz)
        return primary

    def adjust_carcinoma_to_gender(
        self, gender: str, primary: str, secondary: list
    ) -> tuple:
        """Adjust carcinmoa ICDs to the according gender.

        Can't filter in ICD for ovary cancer as ovar is also part of "Biovar cholerae"

        Args:
            gender (str): the gender of the patient
            primary (str): the primary ICD code
            secondary (list): the list of secondary ICD codes

        Returns:
            a `tuple` of the adjusted primary ICD code and the adjusted list of secondary ICD codes
        """
        if primary.startswith("D07"):
            p = True
        else:
            p = False

        result = [s for s in secondary if s.startswith("D07")]

        if p:
            if (
                int(primary.split(".")[1]) <= 3 and gender == "M"
            ):  # minors <= 3 are carcinomes for females only
                primary = random.choice(["D07.4", "D07.5", "D07.6"])
            if (
                int(primary.split(".")[1]) >= 4 and gender == "F"
            ):  # minors >= 4 are carcinomes for males only
                primary = random.choice(["D07.0", "D07.1", "D07.2", "D07.3"])

        if len(result) > 0:
            for r in result:
                if (
                    int(r.split(".")[1]) <= 3 and gender == "M"
                ):  # minors <= 3 are carcinomes for females only
                    secondary = [
                        x if x != r else random.choice(["D07.4", "D07.5", "D07.6"])
                        for x in secondary
                    ]
                if (
                    int(r.split(".")[1]) >= 4 and gender == "F"
                ):  # minors >= 4 are carcinomes for males only
                    secondary = [
                        x
                        if x != r
                        else random.choice(["D07.0", "D07.1", "D07.2", "D07.3"])
                        for x in secondary
                    ]

        return primary, secondary

    def adjust_kidney_secondary(self, gender: str, primary: str, secondary: list):
        """Adjust possible secondary in the kidney region to gender.

        Args:
            gender (str): the gender of the patient
            primary (str): the primary ICD code
            secondary (list): the list of secondary ICD codes

        Returns:
            a `tuple` of the adjusted primary ICD code and the adjusted list of secondary ICD codes
        """
        if gender == "M":
            if primary in ["S37.4", "S37.5", "S37.6"]:
                primary = random.choice(["S37.82", "S37.83", "S37.84"])
            for i, x in enumerate(secondary):
                if x in ["S37.4", "S37.5", "S37.6"]:
                    secondary[i] = random.choice(["S37.82", "S37.83", "S37.84"])

        elif gender == "F":
            if primary in ["S37.82", "S37.83", "S37.84"]:
                primary = random.choice(["S37.4", "S37.5", "S37.6"])
            for i, x in enumerate(secondary):
                if x in ["S37.82", "S37.83", "S37.84"]:
                    secondary[i] = random.choice(["S37.4", "S37.5", "S37.6"])

        return primary, secondary

    def change_companion(self, primary: str) -> str:
        """Change companion ICDs to other ICDs.

        Args:
            primary (str): the primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""
        if primary == "Z76.3":
            primary = random.choice(
                ["Z76.0", "Z76.1", "Z76.2", "Z76.4", "Z76.8", "Z76.9"]
            )
        return primary

    def change_fetus_abortus(self, primary: str) -> str:
        """Change fetus abort ICDs to other ICDs.

        Args:
            primary (str): the primary ICD code

        Returns:
            a `str` of the adjusted primary ICD code"""

        if primary == "P96.4":
            primary = random.choice(
                ["P96.0", "P96.1", "P96.2", "P96.3", "P96.5", "P96.8", "P96.9"]
            )

        return primary
