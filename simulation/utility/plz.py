import os
import random
import traceback
import numpy as np
import pandas as pd
import mpu
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class PLZ:
    """PLZ is used to create zip codes for Germany and several helper methods based on location.

    Attributes:
        plz: a Dataframe of PLZs with the probability of a person living there
    """

    def __init__(self, random_state: int = None):
        """Creating an instance of PLZ.

        Initializing plz with all German PLZs and the population distribution..
        Variables here are used to hold them in memory without the need to access the same information twice from files.

        Args:
            random_state (int): None if not reproducable, int if reproducable. Defaults to None.
        """
        self.plz = pd.read_csv("data/PLZ/georef-germany-postleitzahl.csv", sep=";")
        self.plz = self.plz.drop_duplicates(subset="Postleitzahl / Post code")
        self.plz = self.map_people_on_plz()

        if random_state is not None:
            random.seed(random_state)

    def map_people_on_plz(self) -> pd.DataFrame():
        """Get the population distribution over all PLZs.

        Returns:
            A `Pandas.DataFrame` containing all PLZs and the probability of a person living there.
        """
        peeps = pd.read_csv("data/PLZ/plz_einwohner.csv")
        peeps = peeps[["plz", "einwohner"]]

        new_plz = pd.merge(
            self.plz, peeps, left_on="Postleitzahl / Post code", right_on="plz"
        )
        new_plz = new_plz.drop("plz", axis=1)
        pop = new_plz["einwohner"].to_numpy()
        pop = pop / np.sum(pop)
        new_plz["einwohner"] = pop
        return new_plz

    def get_plz_list(self) -> tuple[list, list]:
        """Get the list of German PLZs and the responding probabilities.

        Returns:
            a tuple of two lists of German PLZs and the responding probabilities
        """
        zip = self.plz["Postleitzahl / Post code"]
        zip = zip.to_numpy()
        zip = zip.tolist()

        prob = self.plz["einwohner"]
        prob = prob.to_numpy()
        prob = prob.tolist()
        return zip, prob

    def get_distance(self, plz1: int, plz2: int) -> float:
        """Get the distance between 2 PLZs.

        The distance is calculated using haversine distance on the given plzs' coordinates.

        Args:
            plz1 (int): the first plz
            plz2 (int): the second plz

        Returns:
            a `float` value repressenting the straight-line distance in km
        """
        try:
            zip1 = self.plz.loc[
                self.plz["Postleitzahl / Post code"] == plz1, "geo_point_2d"
            ].item()
        except (ValueError, IndexError):
            try:
                zip1 = (
                    self.plz.loc[
                        self.plz["Postleitzahl / Post code"] == plz1, "geo_point_2d"
                    ]
                    .iloc[0]
                    .item()
                )
            except (ValueError, IndexError):
                return -1
        try:
            # traceback.print_exc()
            zip2 = self.plz.loc[
                self.plz["Postleitzahl / Post code"] == plz2, "geo_point_2d"
            ].item()
        except (ValueError, IndexError):
            # traceback.print_exc()
            try:
                zip2 = (
                    self.plz.loc[
                        self.plz["Postleitzahl / Post code"] == plz2, "geo_point_2d"
                    ]
                    .iloc[0]
                    .item()
                )
            except (ValueError, IndexError):
                return -1

        zip1 = tuple(float(x) for x in zip1.split(","))
        zip2 = tuple(float(x) for x in zip2.split(","))

        dist = round(mpu.haversine_distance(zip1, zip2), 2)
        return dist

    def get_random_plz(self) -> int:
        """Get a random German plz with regards to population distribution.

        Args:
            random_state (int): None if not reproducable, int if reproducable. Defaults to None.

        Returns:
            an `int` value representing a plz
        """

        plz, prob = self.get_plz_list()
        zip = np.random.choice(plz, p=prob)
        return zip

    def get_state_from_plz(self, plz1: int) -> str:
        """Get the state a plz is located in.

        Args:
            plz1 (int): the plz of interest

        Returns:
            a str for the state (e.g. "Bayern")
        """
        return self.plz.loc[
            self.plz["Postleitzahl / Post code"] == plz1, "Land name"
        ].item()

    def get_state_list(self) -> list:
        """Get a list of all states in Germany.

        Returns:
            a `list` of all states in Germany
        """
        return list(self.plz["Land name"].unique())

    def get_state_plz_list(self, state: str) -> tuple[list, list]:
        """Get the list of the given state's PLZs and the responding probabilities.

        Args:
            state (str): the name of the state of interest

        Returns:
            a tuple of two lists of the given state's PLZs and the responding probabilities
        """

        plz = self.plz.loc[self.plz["Land name"] == state]
        zip = plz["Postleitzahl / Post code"]
        zip = zip.to_numpy()
        zip = zip.tolist()

        prob = plz["einwohner"] / np.sum(plz["einwohner"])
        prob = prob.to_numpy()
        prob = prob.tolist()
        return zip, prob

    def get_random_state_plz(self, state: str) -> int:
        """Get a random plz within the given state with regards to population distribution.

        Args:
            state (str): the name of the state of interest
            random_state (int): None if not reproducable, int if reproducable. Defaults to None.

        Returns:
            an `int` value representing a plz.
        """

        plz, prob = self.get_state_plz_list(state)
        zip = np.random.choice(plz, p=prob)
        return zip

    def get_random_PLZ_ending(self, prefix: int) -> int:
        """Given the first chars of a plz, get a random PLZ.

        Args:
            prefix (int): the first few chars of a plz

        Returns:
            an `int` value as a full plz
        """
        mask = self.plz["Postleitzahl / Post code"].apply(
            lambda x: any(
                [
                    str(x).startswith(str(prefix))
                    for i in self.plz["Postleitzahl / Post code"].to_list()
                ]
            )
        )
        plz_list = self.plz[mask]
        # print(plz_list)
        probs = plz_list["einwohner"] / np.sum(plz_list["einwohner"])
        plz = np.random.choice(plz_list["Postleitzahl / Post code"], p=probs)
        return plz


# print(PLZ().get_random_state_plz("Bayern"))
