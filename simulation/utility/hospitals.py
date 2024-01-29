import sys

sys.path.insert(0, "")

import pandas as pd
import numpy as np

import simulation.utility.plz as plzs


class Hospitals:
    """Hospitals is used for hospital generation.

    Attributes:
        hospitals: a DataFrame containing a list of all hospitals and relevant information (e.g. number of beds)
    """

    def __init__(self, plz=None):
        """Initialize the hospitals attribute and removing duplicates
        Variables here are used to hold them in memory without the need to access the same information twice from files.

        Args:
            plz (PLZ, optional): a PLZ object. Defaults to None.
        """

        if plz is None:
            self.plz = plzs.PLZ()
        else:
            self.plz = plz

        self.hospitals = pd.read_csv(
            "data/Hospitals/krankenhausverzeichnis-3500100217005.csv",
            sep=";",
        )
        self.hospitals = self.hospitals.drop_duplicates(
            subset=["E-Mail Adresse", "Telefonvorwahl/-nummer", "Internet-Adresse"]
        )
        # hospitals = hospitals["EinrichtungsTyp"].astype(int)
        self.hospitals = self.hospitals.loc[self.hospitals["EinrichtungsTyp"] < 4]
        self.hospitals["Adresse_Postleitzahl_Standort"].fillna(0)
        self.hospitals["Adresse_Postleitzahl_Standort"] = (
            self.hospitals["Adresse_Postleitzahl_Standort"].fillna(0).astype(int)
        )
        self.hospitals["INSG"] = self.hospitals["INSG"].replace(" ", 0)
        self.hospitals["INSG"] = self.hospitals["INSG"].fillna(0).astype(int)
        self.hospitals = self.hospitals.loc[self.hospitals["INSG"] > 0]
        self.hospitals = self.hospitals[
            ["Adresse_Postleitzahl_Standort", "Adresse_Name", "INSG"]
        ]
        self.hospitals["INSG"] = self.hospitals["INSG"] / np.sum(self.hospitals["INSG"])
        self.hospitals = self.hospitals.loc[self.hospitals["INSG"] > 0]
        # print(self.hospitals)

    def get_hospitals(self) -> pd.DataFrame():
        """Get the hospitals DataFrame.

        Returns:
            a `Pandas.DataFrame()` containing relevant information on hospitals.
        """
        return self.hospitals

    def get_hospitals_from_state(self, state: str) -> pd.DataFrame():
        """Get the hospitals DataFrame with only hospitals in the given state.

        Args:
            state (str): the Name of a German state (e.g. "Bayern")

        Returns:
            a `Pandas.DataFrame()` containing relevant information on hospitals in the given state
        """
        zip, x = self.plz.get_state_plz_list(state)
        hospitals = self.hospitals[
            self.hospitals["Adresse_Postleitzahl_Standort"].isin(zip)
        ]

        hospitals["INSG"] = hospitals["INSG"] / np.sum(hospitals["INSG"])
        return hospitals


# print(Hospitals().get_hospitals_from_state("Bayern"))
