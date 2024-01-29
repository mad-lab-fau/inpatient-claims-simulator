import sys

sys.path.insert(0, "")

import numpy as np
import pandas as pd

import simulation.utility.grouper_wrapping as grouper_wrapping
import simulation.utility.config as config
import simulation.utility.utils as utils


class FRAUD:
    def increase_ventilation(self, b_hours: int) -> int:
        """Increasing the number of ventilation hours to be over the next threshold.

        Args:
            b_hours (int): the current number of ventilation hours

        Returns:
            an `int` value for the new (fraudulent) number of breathing hours


        Source:
            Salomon, F. (2010).
            Ökonomie und Ethik im Klinikalltag - Der Arzt im Spannungsfeld zwischen Patientenwohl und Wirtschaftlichkeit
            [Economy and ethics in daily hospital routine - physicians in conflict between the well-being of patients and profitability].
            Anasthesiologie, Intensivmedizin, Notfallmedizin, Schmerztherapie : AINS, 45(2), 128–131.
            https://doi.org/10.1055/s-0030-1248148

            Busse, R., Geissler, A., Aaviksoo, A., Cots, F., Häkkinen, U., Kobel, C., Mateus, C., Or, Z., O'Reilly, J., Serdén, L., Street, A., Tan, S. S., & Quentin, W. (2013).
            Diagnosis related groups in Europe: Moving towards transparency, efficiency, and quality in hospitals?
            BMJ (Clinical Research Ed.), 346, f3197.
            https://doi.org/10.1136/bmj.f3197

            https://reimbursement.institute/glossar/beatmungsdauer/
            Beatmungsdauer
            > 24 Stunden
            > 48 Stunden
            > 59 Stunden
            > 72 Stunden
            > 95 Stunden
            > 120 Stunden
            > 179 Stunden
            > 180 Stunden
            > 240 Stunden
            > 249 Stunden
            > 263 Stunden
            > 275 Stunden
            > 320 Stunden
            > 479 Stunden
            > 480 Stunden
            < 481 Stunden
            > 499 Stunden
            > 599 Stunden
            > 899 Stunden
            > 999 Stunden
            > 1.799 Stunden
        """
        fraud_coefficient = (
            config.fraud_coefficient_ventilation
        )  # increase for more fraudulent behavior, decrease for less
        thresholds = [
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
        ]

        for t in thresholds:
            rel_hours = np.ceil(t - t * fraud_coefficient)
            if b_hours <= t and b_hours > rel_hours:
                return t + 1

        return b_hours

    def decrease_weight(self, weight: int) -> int:
        """Decreasing the weight of newborns to be under the next threshold.

        Args:
            weight (int): the current weight of a newborn
            fraud_coefficient (float, optional): the coefficient for the decrease of the weight. Defaults to 0.12.

        Returns:
            an `int` value for the new (fraudulent) weight


        Source:
            Jürges, H., & Köberlein, J.
            First do no harm. Then do not cheat: DRG upcoding in German neonatology.
            DIW Discussion Papers, 2013(No. 1314).
            http://hdl.handle.net/10419/79257
        """
        fraud_coefficient = (
            config.fraud_coefficient_weight
        )  # increase for more fraudulent behavior, decrease for less
        thresholds = [600, 749, 874, 999, 1249, 1499, 1999, 2499]
        for t in thresholds:
            rel_weight = np.ceil(t * (1 + fraud_coefficient))
            if int(weight) >= t and int(weight) < rel_weight:
                return t - 1

        return weight

    def change_to_caesarean(self, primary: str) -> str:
        """Changing a normal birth to a Caesarean birth.

        If a birth is coded, change it to a Caesarean.

        Args:
            primary (str): the primary ICD

        Returns:
            a String for Caesarean


        Source:
            Salomon, F. (2010).
            Ökonomie und Ethik im Klinikalltag - Der Arzt im Spannungsfeld zwischen Patientenwohl und Wirtschaftlichkeit
            [Economy and ethics in daily hospital routine - physicians in conflict between the well-being of patients and profitability].
            Anasthesiologie, Intensivmedizin, Notfallmedizin, Schmerztherapie : AINS, 45(2), 128–131.
            https://doi.org/10.1055/s-0030-1248148
        """
        if primary == "O80":
            return "O82"

    def change_ICD_order(
        self, id: int, primary: str, secondary: list, temp_comb
    ):  
        """Changing the order of given ICDs to get a better DRG.

        Trying to change the order of ICDs by changing current secondary ICDs to primary ICD and checking with the DRG grouper.

        Args:
            primary (str): the current primary ICD
            secondary (list): the current list of secondary ICDs
            days (int): the number of days stayed in hospital
            claim (float): the current claim ammount
            temp_comb (pd.DataFrame): a temporary DataFrame to store the combinations of ICDs

        Returns:
            a `pd.DataFrame` with the combinations of ICDs

        Source:
            van Herwaarden, S., Wallenburg, I., Messelink, J., & Bal, R. (2020).
            Opening the black box of diagnosis-related groups (DRGs): Unpacking the technical remuneration structure of the Dutch DRG system.
            Health Economics, Policy and Law, 15(2), 196–209.
            https://doi.org/10.1017/S1744133118000324
        """

        combinations = []

        # Generate combinations for "X" and Y
        combination1 = [primary] + secondary
        combinations.append(utils.strip_nA(combination1))

        # Generate combinations for each element in Y and [X] + other elements in Y
        for i in range(len(secondary)):
            combination = [secondary[i], primary] + list(
                secondary[:i] + secondary[i + 1 :]
            )
            if combination not in combinations:
                combinations.append(combination)
        self.grouper.prepare_fraud_grouper(
            id, combinations
        )  # adding the combinations to the grouper preparation

        # try:
        #    temp_comb = pd.read_csv("data/generated data/temporary_combinations.csv")
        # except pd.errors.EmptyDataError:
        #    temp_comb = pd.DataFrame()

        temp = pd.DataFrame()
        temp["ID"] = np.full(len(combinations), id)
        temp["ICD"] = combinations

        temp_comb = pd.concat([temp_comb, temp], axis=0)
        # temp_comb.to_csv("data/generated data/temporary_combinations.csv", index=False)
        return temp_comb


    ## newborns with a secondary diagnosis of “need for assistance with personal care” (ICD-10:Z74.1)
    def newborn_add_personal_care(self, age: int, secondary: list) -> list:
        """Adds the necessity of personal care for newborns.

        Args:
            age (int): the age of the person (must be 0)
            secondary (list): the list of secondary ICDs

        Returns:
            a `list` of secondary ICDs extended with the ICD for personal care


        Source:

        Busse, R., Geissler, A., Aaviksoo, A., Cots, F., Häkkinen, U., Kobel, C., Mateus, C., Or, Z., O'Reilly, J., Serdén, L., Street, A., Tan, S. S., & Quentin, W. (2013).
        Diagnosis related groups in Europe: Moving towards transparency, efficiency, and quality in hospitals?
        BMJ (Clinical Research Ed.), 346, f3197.
        https://doi.org/10.1136/bmj.f3197
        """
        if age < 1 and "Z74.1" not in secondary:
            secondary.append("Z74.1")
            return secondary
        return secondary

    def adjust_duration(self, id: int, days: int, ids: list) -> list:
        temp_params = pd.DataFrame()

        if days > 1:
            ids.append(id)

        return ids

    def __init__(self, grouper) -> None:
        self.grouper = grouper


# change_ICD_order(1, "I21.0", ["J18.9", "I10.00", "I50.13", "A08.0"])
