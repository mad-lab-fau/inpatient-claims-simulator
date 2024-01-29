import numpy as np
import pandas as pd
import random

import cProfile
from pstats import Stats, SortKey
import sys

sys.path.insert(0, "")
from simulation.utility.grouper_wrapping import Grouper


class DRG:
    """DRG is used to estimate the duration of a inpatient stay, mapping OPS codes on ICD codes and calculating claims from case ratios.

    Attributes:
        overview: a DataFrame containing general information on DRGs
        primary: a DataFrame containing the mapping of primary ICDs on DRGs
        secondary: a DataFrame containing the mapping of secondary ICDs on DRGs
        ops: a DataFrame containing the mapping of OPS on DRGs
        lbfw: a DataFrame containing the values of LBFW
        extrabudget: a DataFrame containing the names of extrabudgetary DRGs
    """

    def __init__(self, random_state: int = None):
        """Creating an instance of DRG.

        Initializing all important values with empty states. This is replaced with the first access to the variable.
        Variables here are used to hold them in memory without the need to access the same information twice from files.

        Args:
            random_state (int, optional): a random state for reproducibility. Defaults to None.
        """
        self.overview = pd.DataFrame()
        self.primary = pd.DataFrame()
        self.secondary = pd.DataFrame()
        self.ops = pd.DataFrame()
        self.lbfw = pd.DataFrame()
        self.extrabudget = pd.DataFrame()

        if random_state is not None:
            random.seed(random_state)

    def get_DRG_list_details(self) -> pd.DataFrame:
        """Reading DRG information and filtering out irrelevant information.

        The relevant information contains columns for the DRG, base ratio, mean duration of stay, lower boundary of stay,
        upper boundary of stay, and base ratio for care work.

        Returns:
            a `Pandas.DataFrame()` containing relevant information on DRGs.
        """
        drg = self.overview
        if drg.empty:
            # print("drg")
            drg = pd.read_csv(
                "data/DRG/Fallpauschalenkatalog_2021_20201112.csv",
                sep=";",  # , index_col=False
            )
            drg = drg[
                [
                    "DRG",
                    "Bewertungsrelation bei Hauptabteilung",
                    "Mittlere Verweildauer",
                    "Untere Grenzverweildauer",
                    "Obere Grenzverweildauer",
                    "Pflegeerlös Bewertungsrelation pro Tag",
                    "Untere Grenzverweildauer: Bewertungsrelation pro Tag",
                    "Obere Grenzverweildauer: Bewertungsrelation pro Tag",
                ]
            ]
            drg["DRG"] = drg["DRG"].fillna("")
            drg = drg[~drg["DRG"].str.contains("MDC")]

            """drg["Untere Grenzverweildauer"] = drg[
                "Untere Grenzverweildauer"
            ].str.replace("-", "0")"""
            drg["Untere Grenzverweildauer"] = drg["Untere Grenzverweildauer"].fillna(
                "0"
            )
            drg["Mittlere Verweildauer"] = drg["Mittlere Verweildauer"].str.replace(
                "-", ""
            )
            """drg["Obere Grenzverweildauer"] = drg["Obere Grenzverweildauer"].str.replace(
                "-", ""
            )
            drg["Pflegeerlös Bewertungsrelation pro Tag"] = drg[
                "Pflegeerlös Bewertungsrelation pro Tag"
            ].str.replace("-", "0")"""
            drg["Obere Grenzverweildauer"] = drg["Obere Grenzverweildauer"].fillna("0")
            drg["Bewertungsrelation bei Hauptabteilung"] = drg[
                "Bewertungsrelation bei Hauptabteilung"
            ].str.replace("-", "0")

            drg["Mittlere Verweildauer"] = drg["Mittlere Verweildauer"].str.replace(
                ",", "."
            )

            drg["Mittlere Verweildauer"] = pd.to_numeric(
                drg["Mittlere Verweildauer"], errors="coerce"
            )

            """drg["Untere Grenzverweildauer"] = drg[
                "Untere Grenzverweildauer"
            ].str.replace(",", ".")"""

            drg["Untere Grenzverweildauer"] = pd.to_numeric(
                drg["Untere Grenzverweildauer"]
            )
            """drg["Obere Grenzverweildauer"] = drg["Obere Grenzverweildauer"].str.replace(
                ",", "."
            )"""

            drg["Obere Grenzverweildauer"] = pd.to_numeric(
                drg["Obere Grenzverweildauer"], errors="coerce"
            )

            drg["Bewertungsrelation bei Hauptabteilung"] = drg[
                "Bewertungsrelation bei Hauptabteilung"
            ].str.replace(",", ".")
            drg["Bewertungsrelation bei Hauptabteilung"] = pd.to_numeric(
                drg["Bewertungsrelation bei Hauptabteilung"]
            )

            drg["Untere Grenzverweildauer: Bewertungsrelation pro Tag"] = drg[
                "Untere Grenzverweildauer: Bewertungsrelation pro Tag"
            ].str.replace(",", ".")
            drg["Untere Grenzverweildauer: Bewertungsrelation pro Tag"] = pd.to_numeric(
                drg["Untere Grenzverweildauer: Bewertungsrelation pro Tag"]
            )

            drg["Obere Grenzverweildauer: Bewertungsrelation pro Tag"] = drg[
                "Obere Grenzverweildauer: Bewertungsrelation pro Tag"
            ].str.replace(",", ".")
            drg["Obere Grenzverweildauer: Bewertungsrelation pro Tag"] = pd.to_numeric(
                drg["Obere Grenzverweildauer: Bewertungsrelation pro Tag"]
            )

            drg["Pflegeerlös Bewertungsrelation pro Tag"] = drg[
                "Pflegeerlös Bewertungsrelation pro Tag"
            ].str.replace(",", ".")
            drg["Pflegeerlös Bewertungsrelation pro Tag"] = pd.to_numeric(
                drg["Pflegeerlös Bewertungsrelation pro Tag"]
            )

            self.overview = drg

        return drg

    def get_DRG_list_primary(self) -> pd.DataFrame:
        """Reading DRG information on primary ICDs filtering out irrelevant information.

        The relevant information contains columns for the DRG, ICD code, and a percentage (for relevancy).

        Returns:
            a `Pandas.DataFrame()` containing a mapping of primary ICDs on DRGs.
        """
        drg = self.primary
        if drg.empty:
            drg = pd.read_csv(
                "data/DRG/RepBrDrg_HA_21_Hauptdiagnose.csv",
                sep=";",
            )
            drg = drg[["DRG", "Code", "Prozent"]]
            drg["DRG"] = drg["DRG"].fillna("")
            drg = drg[~drg["DRG"].str.startswith("80")]
            drg = drg[
                ~drg["DRG"].str.startswith("A")
            ]  # A-DRGs are excluded as the point to transplantations and ventilation and interfere correct mappings
            self.primary = drg
        return drg

    def get_DRG_list_secondary(self) -> pd.DataFrame:
        """Reading DRG information on secondary ICDs filtering out irrelevant information.

        The relevant information contains columns for the DRG, ICD code, and a percentage (for relevancy).

        Returns:
            a `Pandas.DataFrame()` containing a mapping of secondary ICDs on DRGs.
        """
        drg = self.secondary
        if drg.empty:
            drg = pd.read_csv(
                "data/DRG/RepBrDrg_HA_21_Nebendiagnosen.csv",
                sep=";",
            )
            drg = drg[["DRG", "Code", "ProzentN"]]
            drg["DRG"] = drg["DRG"].fillna("")
            drg = drg[~drg["DRG"].str.startswith("80")]
            drg = drg[
                ~drg["DRG"].str.startswith("A")
            ]  # A-DRGs are excluded as the point to transplantations and ventilation and interfere correct mappings
            drg.set_index("Code", inplace=True)
            self.secondary = drg
        return drg

    def get_DRG_list_procedures(self) -> pd.DataFrame:
        """Reading DRG information on OPS filtering out irrelevant information.

        The relevant information contains columns for the DRG, OPS code, and a percentage (for relevancy).

        Returns:
            a `Pandas.DataFrame()` containing a mapping of OPS on DRGs.
        """
        drg = self.ops
        if drg.empty:
            drg = pd.read_csv(
                "data/DRG/RepBrDrg_HA_21_Prozeduren.csv",
                sep=";",
            )
            drg = drg[["DRG", "Code", "ProzentO"]]
            drg["DRG"] = drg["DRG"].fillna("")
            drg = drg[
                ~drg["DRG"].str.startswith("80")
            ]  # 80er DRGs offer too many opportunities for mapping as the main intent of those DRGs are to point on ventilation
            drg = drg[
                ~drg["DRG"].str.startswith("A")
            ]  # A-DRGs are excluded as the point to transplantations and ventilation and interfere correct mappings
            # drg = drg[~drg["Code"].str.startswith("5-")] # 5er OPS codes are included by manual mapping (see line 339 in opy)
            drg.set_index("DRG", inplace=True)
            self.ops = drg
        return drg

    def get_OPS_list_on_ICD(self, primary: str, secondary: list) -> list:
        """Get a list of relevant OPS codes given primary and secondary diagnoses.

        Get a list of mapped OPS codes by comparing relevant DRGs for given diagnoses and filtering the OPS-DRG mapping accordingly.
        The creation of one large DataFrame mapping every primary and secondary ICD as well as OPS on DRG failed due to a limit in RAM (32 GB).

        Args:
            primary (str): String of a primary ICD code (e.g. "I10.00")
            secondary: List of Strings of secondary ICD codes (e.g. ["K65.09", "S72.11"])

        Returns:
            two `list`s of Strings of possible OPS codes, the first shortened to 5 digits, the other till the final digit
        """
        primary_list = self.get_DRG_list_primary()
        # secondary_list = self.get_DRG_list_secondary()

        drg_list_1 = primary_list[primary_list["Code"] == primary]

        drg_list_2 = pd.DataFrame()
        """try:
            
            drg_list_2 = secondary_list.loc[
                secondary
            ]  # using index according to https://stackoverflow.com/questions/23945493/a-faster-alternative-to-pandas-isin-function
        except KeyError:
            pass"""
        # drg_list_2 = secondary_list[secondary_list["Code"].isin(secondary)]
        drg = pd.concat([drg_list_1, drg_list_2], axis=0)
        drg = drg["DRG"].drop_duplicates()
        drg = drg.to_list()

        drg_list = self.get_DRG_list_procedures()

        try:
            drg_list = drg_list.loc[drg]
        except KeyError:
            pass
        # drg_list = drg_list[drg_list["DRG"].isin(drg)]
        ops_list = drg_list["Code"].drop_duplicates()
        # print(ops_list)
        ops_list = ops_list.to_list()
        # ops_list = [s for s in ops_list if s.startswith("5-")==False] # should not occur anymore

        # if one of the ops_list elements starts with 5- print list
        # if any("5-" in s for s in ops_list):
        #    print("HEIHO!!")

        # ops_list = [s for s in ops_list if s.startswith("8-8")==False] # 8-8er OPS codes are way too many
        ops_list_5_digits = [s[:5] for s in ops_list]
        ops_list = list(set(ops_list))
        return ops_list_5_digits, ops_list

    def get_DRG_overview(
        self, primary: str, secondary: list, ops: list, breathing: int
    ) -> pd.DataFrame:
        """Get relevant information on DRGs given primary and secondary ICDs, OPS codes and the number of breathing hours.

        Merging the tables for mapping of primary ICD, secondary ICD, and OPS on DRG with the relevant information of DRGs.
        This merge is only carried out on relevant DRGs.

        Args:
            primary (str): String of a primary ICD code (e.g. "I10.00")
            secondary (list): List of Strings of secondary ICD codes (e.g. ["K65.09", "S72.11"])
            ops (list): List of Strings of OPS codes
            breathing (int): number of hours for ventilation

        Returns:
            a `Pandas.DataFrame()` containing the relevant information on DRGs given the input values.
        """
        drg_details = self.get_DRG_list_details()
        icd_primary = self.get_DRG_list_primary()
        # icd_secondary = self.get_DRG_list_secondary()
        # ops_list = self.get_DRG_list_procedures()

        icd_primary = icd_primary[icd_primary["Code"] == primary]
        # icd_secondary = icd_secondary[icd_secondary["Code"].isin(secondary)]
        # ops_list = ops_list[ops_list["Code"].isin(ops)]

        drg_details.set_index("DRG")
        icd_primary = icd_primary.set_index("DRG")
        details_x_primary = drg_details.join(icd_primary, on="DRG")
        # icd_secondary = icd_secondary.set_index("DRG")
        # details_x_primary_x_secondary = details_x_primary.join(
        #    icd_secondary, on="DRG", rsuffix="_SEC"
        # )
        # ops_list = ops_list.set_index("DRG")
        # drg_overview = details_x_primary_x_secondary.join(
        #    ops_list, on="DRG", rsuffix="_OPS", lsuffix="_PRI"
        # )
        drg_overview = details_x_primary
        drg_overview = drg_overview.drop_duplicates()
        drg_overview = drg_overview.reset_index(drop=True)

        drg_overview = drg_overview.iloc[:-6]

        if breathing == 0:
            breathing_drg = ["A06", "A07", "A09", "A11", "A13"]
            mask = drg_overview["DRG"].apply(
                lambda x: any([x.startswith(i) for i in breathing_drg])
            )
            drg_overview = drg_overview[~mask]
        return drg_overview

    def get_LBFW(self, state: str) -> float:
        """Get the individual LBFW for each German state.

        Reading the file for LBFWs and filtering out the relevant value for the given state

        Args:
            state (str): the name of a German state (e.g. "Bayern")

        Retuns:
            a `float` value indicating an Euro-value for each base ratio point
        """
        lbfw = self.lbfw
        if lbfw.empty:
            lbfw = pd.read_csv(
                "data/DRG/landesbasisfallwerte.csv",
            )
            self.lbfw = lbfw
        ret = lbfw[lbfw["Bundesland"] == state]
        return ret["Zahlbetrags-LBFW 2021"].item()

    def get_care_value(self, date: str) -> float:
        """Get the value of a care ratio point depending on the date of care.

        Args:
            date (str): a String indicating a day in the format YYYY-MM-DD

        Returns:
            a `float` value indicating an Euro-value for each care ratio point

        Raises:
            ValueError: if the date is before 2021 (the use of care ratio points was introduced 01.01.2021)
        """
        if (date >= pd.to_datetime("2021-01-01")) and (
            date <= pd.to_datetime("2022-06-30")
        ):
            return 163.09
        elif (date >= pd.to_datetime("2022-07-01")) and (
            date <= pd.to_datetime("2022-12-31")
        ):
            return 200.00
        elif date >= pd.to_datetime("2023-01-01"):
            return 230.00
        else:
            raise ValueError(
                str(date) + " before 2021 and therefore no aDRG calculation possible"
            )

    def get_claim_ammount(
        self, points: float, date: str, days: int, state: str, care_relation: float
    ) -> float:
        """Calculate the claim ammount based on treatment information.

        The claim value is calculated by adding the care value (= care_value * days of stay * care relation) and the
        treatment value (= base ratio points * LBFW).

        Args:
            points (float): base ratio points for the according DRG
            date (str): a String indicating a day in the format YYYY-MM-DD
            days (int): number of days for inpatient stay
            state (str): the Name of a German state (e.g. "Bayern")
            care_relation (float): care ratio points for the according DRG

        Returns:
            a `float` value rounded to 2 decimals representing the claims value
        """
        care_value = round(self.get_care_value(date=date) * days * care_relation, 2)
        # print(points * float(self.get_LBFW(state)))
        treatment = round(points * float(self.get_LBFW(state)), 2)

        return round(care_value + treatment, 2)

    def generate_random_days(self, lower: float, upper: float, mean: float) -> int:
        """Get a random number of inpatient stay duration in days using a Gaussian distribution.

        Note:
            If no upper value is given, 3 * mean is assumed.
            If no lower value is given, 0 is assumed.

        Args:
            lower (float): lower boundary of stay duration in days
            upper (float): upper boundary of stay duration in days
            mean (float): mean duration of stay in days

        Returns:
            an `int` number of days for the given hospital treatment
        """
        if np.isnan(upper):
            upper = mean * 3
        if np.isnan(lower):
            lower = 0

        x = 0
        t = 0
        mu = mean - 1
        sigma = (upper - (lower)) / 4
        while (
            x < 1 and t < 10
        ):  # avoiding values < 1 while still using gaussian distribution (0 days would be another type of hospital stay, negative values are nonsense)
            x = random.gauss(mu, sigma)
            t += 1
            if t == 10:
                x = mean
        # print(str(lower) + " - " + str(mean) + " - " + str(upper) + str(type(upper)))
        return round(x)

    def get_DRG_params(
        self,
        primary: str,
        secondary: list,
        ops: list,
        breathing: int,
        state: str,
        date: str,
    ) -> int:
        """Get the values for duration of stay and claims ammount given diagnoses and operations.

        Args:
            primary (str): String of a primary ICD code (e.g. "I10.00")
            secondary (list): List of Strings of secondary ICD codes (e.g. ["K65.09", "S72.11"])
            ops (list): List of Strings of OPS codes
            breathing (int): number of hours for ventilation
            state (str): the Name of a German state (e.g. "Bayern")
            date (str): a String indicating a day in the format YYYY-MM-DD

        Returns:
            an `int` number of days for the given hospital treatment
        """

        drg = self.get_DRG_overview(primary, secondary, ops, breathing)
        # print(drg)
        if drg.dropna() is not drg.empty:
            drg.dropna()

        drg["Prozent"] = drg["Prozent"].str.replace(",", ".")
        drg["Prozent"] = pd.to_numeric(drg["Prozent"])

        # drg["ProzentN"] = drg["ProzentN"].str.replace(",", ".")
        # drg["ProzentN"] = pd.to_numeric(drg["ProzentN"])

        # drg["ProzentO"] = drg["ProzentO"].str.replace(",", ".")
        # drg["ProzentO"] = pd.to_numeric(drg["ProzentO"])

        # mean = drg["Mittlere Verweildauer"].astype("float").mean()
        # lower = drg["Untere Grenzverweildauer"].astype("float").mean()
        # upper = drg["Obere Grenzverweildauer"].astype("float").mean()
        # points = drg["Bewertungsrelation bei Hauptabteilung"].astype("float").mean()

        max_percentage = drg[["Prozent"]].max(axis=1)
        max_drg = max_percentage.idxmax()
        if not np.isnan(max_drg):
            mean = float(drg.loc[max_drg, "Mittlere Verweildauer"])
            lower = float(drg.loc[max_drg, "Untere Grenzverweildauer"])
            upper = float(drg.loc[max_drg, "Obere Grenzverweildauer"])
            # points = float(drg.loc[max_drg, 'Bewertungsrelation bei Hauptabteilung'])
            # care_relation = float(drg.loc[max_drg, 'Pflegeerlös Bewertungsrelation pro Tag'])
        else:
            mean = drg["Mittlere Verweildauer"].astype("float").mean()
            lower = drg["Untere Grenzverweildauer"].astype("float").mean()
            upper = drg["Obere Grenzverweildauer"].astype("float").mean()
            # points = drg["Bewertungsrelation bei Hauptabteilung"].astype("float").mean()
            # care_relation =drg["Pflegeerlös Bewertungsrelation pro Tag"].astype("float").mean()

        # print(max_drg)
        # points = float(drg.loc[max_drg, 'Bewertungsrelation bei Hauptabteilung'])
        # care_relation = float(drg.loc[max_drg, 'Pflegeerlös Bewertungsrelation pro Tag'])
        days = self.generate_random_days(lower, upper, mean)
        # amount = self.get_claim_ammount(points=points, date=date, days = days, state=state, care_relation = care_relation)

        return days  # , amount

    def calculate_claim(self, drg: str, days: int, date: str, state: str) -> float:
        """Calculate the claim ammount given a DRG, the duration of the stay, the date of stay and the hospital's state.

        Calculate the claim according to § 8 KHEntgG with regards to decreases or increases depending on the lenght of stay.
        Adding the care value to the DRG value calculated.

        Args:
            drg (str): String of a DRG (e.g. "G70B")
            days (int): the number of days for the duration of stay
            date (str): a String indicating a day in the format YYYY-MM-DD
            state (str): the name of a German state (e.g. "Bayern")

        Returns:
            a `float` value rounded to 2 decimals representing the claims value
        """

        drg_list = self.get_DRG_list_details()

        deets = drg_list.loc[drg_list["DRG"] == drg]

        # ratio = float(deets["Bewertungsrelation bei Hauptabteilung"].item().replace(',', '.'))
        if len(deets) == 0:
            ratio = 0
            ugv = 0
            ogv = 0
            care_ratio = 0
        else:
            ratio = deets["Bewertungsrelation bei Hauptabteilung"].item()
            # ugv = float(deets["Untere Grenzverweildauer"].item().replace(',', '.'))
            ugv = deets["Untere Grenzverweildauer"].item()
            # ogv = float(deets["Obere Grenzverweildauer"].item().replace(',', '.'))
            ogv = deets["Obere Grenzverweildauer"].item()
            # care_ratio = float(deets["Pflegeerlös Bewertungsrelation pro Tag"].item().replace(',', '.'))
            care_ratio = deets["Pflegeerlös Bewertungsrelation pro Tag"].item()

        treatment_value = float(self.get_LBFW(state))
        care_value = self.get_care_value(pd.to_datetime(date))
        claim = ratio * treatment_value

        if len(deets) != 0 and drg not in [
            "A60D",
            "B60B",
            "B70I",
            "E02E",
            "E64D",
            "F62D",
            "F73A",
            "I66H",
            "I68E",
            "J65B",
            "J68A",
            "J68B",
            "K63C",
            "L70A",
            "L70B",
            "L71Z",
            "N09A",
            "R65Z",
            "S60Z",
            "T60G",
            "U60A",
            "U60B",
            "Y63Z",
        ]:  # socalled "Ein-Belegungstag-DRG", exempted from duration calculation
            if days <= ugv:
                red_days = ugv - days + 1
                red_ratio = float(
                    deets["Untere Grenzverweildauer: Bewertungsrelation pro Tag"].item()
                    # .replace(",", ".")
                )
                reduction = red_ratio * red_days * treatment_value
                claim -= reduction
            elif days >= ogv:
                inc_days = days - ogv + 1
                inc_ratio = float(
                    deets["Obere Grenzverweildauer: Bewertungsrelation pro Tag"].item()
                    # .replace(",", ".")
                )
                increase = inc_ratio * inc_days * treatment_value
                claim += increase

        care = care_ratio * days * care_value

        claim += care
        claim = round(claim, 2)
        # print(claim)

        extrabudget = self.extrabudget
        if extrabudget.empty:
            extrabudget = pd.read_csv(
                "data/DRG/Anlage_3a.csv",
                sep=";",
            )
            extrabudget = extrabudget[["DRG", "Pflegeerlös Bewertungsrelation/Tag"]]
            extrabudget["DRG"] = extrabudget["DRG"].fillna("")
            extrabudget["DRG"] = extrabudget["DRG"].fillna("")
            extrabudget = extrabudget[~extrabudget["DRG"].str.contains("MDC")]
            self.extrabudget = extrabudget

        extraextra = extrabudget["DRG"].to_list()

        if claim <= 0 and (drg in extraextra):
            claim = (
                27089,
                18,
            )  # assumption of a claim value of 27089,18 for aDRGs based on individual contracts
        return claim

    def get_ugv(self, drg: str) -> float:
        """Get the lower boundary of stay for a given DRG.

        Args:
            drg (str): String of a DRG (e.g. "G70B")

        Returns:
            a `float` value representing the lower boundary of stay
        """
        drg_list = self.get_DRG_list_details()

        deets = drg_list.loc[drg_list["DRG"] == drg]
        if len(deets) == 0:
            ugv = 0
        else:
            ugv = deets["Untere Grenzverweildauer"].item()

        return ugv

    def get_ogv(self, drg: str) -> float:
        """Get the upper boundary of stay for a given DRG.

        Args:
            drg (str): String of a DRG (e.g. "G70B")

        Returns:
            a `float` value representing the upper boundary of stay
        """
        drg_list = self.get_DRG_list_details()

        deets = drg_list.loc[drg_list["DRG"] == drg]
        if len(deets) == 0:
            ogv = 0
        else:
            ogv = deets["Obere Grenzverweildauer"].item()

        return ogv

    def get_negative_OPS_list(self):
        icd_primary = self.get_DRG_list_primary()
        ops = self.get_DRG_list_procedures()

        joined = icd_primary.join(ops, on="DRG", lsuffix="_ICD", rsuffix="_OPS")

        joined = joined[~joined["DRG"].str.startswith("8")]
        joined = joined[~joined["DRG"].str.startswith("9")]
        joined.drop(columns=["DRG", "Prozent", "ProzentO"], inplace=True)
        joined.columns = ["ICD", "OPS"]
        res = Grouper().execute_negative_list_grouping(joined)
        res.columns = ["DRG", "Index", "ICD", "OPS"]
        res.drop(columns=["Index"], inplace=True)
        # keep only rows where DRG starts with 8
        res = res[res["DRG"].str.startswith("8")]
        res.to_csv("data/OPS/generated/negative_OPS_list.csv", index=False)


# DRG().get_negative_OPS_list()

# print(DRG().calculate_claim("I68E", 1, "2021-03-12", "Bayern"))
# (DRG().get_DRG_params("I21.0", ["J18.9", "I10.00", "I50.13", "A08.0"], ["1-275.3", "8-837.00"], True, "Hessen", pd.to_datetime("2023-07-12")))
# print(DRG().get_OPS_list_on_ICD("I50.01", ["K65.09", "S72.11", "S60.86"]))
