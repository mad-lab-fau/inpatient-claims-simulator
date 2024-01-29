import sys

sys.path.insert(0, "")

from multiprocessing import Pool
import pandas as pd

import simulation.utility.grouper_wrapping as grouper_wrapping
import simulation.utility.plz as plzs
import simulation.utility.config as config
import simulation.generation.drg as drgs
import simulation.generation.treatment_generation as treatment_generation
import simulation.fraud.inject_fraud as inject_fraud


class Simulation:
    """SIMULATION orchestrates the entire process of inpatient claims generation by first starting the treatment generation,
    then entering this into the DRG Grouper and using this output to calculate the claims.
    """

    def add_claim(self):
        """Adding the claim value to each row of the generated treatments and therefore creating the final csv-file."""
        treatments = pd.read_csv("data/generated data/claims_with_drg.csv")
        treatments["CLAIM"] = treatments.apply(
            lambda x: self.drg.calculate_claim(
                drg=x["DRG"],
                days=x["DURATION"],
                date=x["DISCHARGE_DATE"],
                state=self.plz.get_state_from_plz(x["PLZ_HOSP"]),
            ),
            axis=1,
        )
        # treatments["CLAIM_2"] = treatments.apply(lambda x: drg.DRG().calculate_claim("P67D", 3, 2021, "Hessen"), axis=1)
        print(treatments)
        treatments.to_csv("data/generated data/claims_final.csv", index=False)

    def generate_data(
        self,
        random_state: int,
        n_patients: int,
        patients: pd.DataFrame,
        n_hospitals: int,
        hospitals: pd.DataFrame,
        n_cases: int,
        id_list: list,
    ):
        """generate_data is a wrapper function for the multiprocessing of the treatment generation. It is called by the multiprocessing function and calls the treatment generation.

        Args:
            random_state (int): random state for reproducibility.
            n_patients (int): Number of patients to be generated.
            patients (pd.DataFrame): DataFrame of patients.
            n_hospitals (int): Number of hospitals to be generated.
            hospitals (pd.DataFrame): DataFrame of hospitals.
            n_cases (int): Number of cases to be generated.
            id_list (list): List of ids to be generated.

        Returns:
            pd.DataFrame: DataFrame of generated treatments."""
        return treatment_generation.GENERATION(random_state=random_state).generate_data(
            n_patients=n_patients,
            patients=patients,
            n_hospitals=n_hospitals,
            hospitals=hospitals,
            n_cases=n_cases,
            id_list=id_list,
        )

    def __init__(
        self,
        n_cases: int,
        n_patients: int = None,
        n_hospitals: int = 5,
        state: str = None,
        random_state: int = None,
        multiprocessing: int = 0,
    ):
        """__init__ is the constructor of the SIMULATION class. It starts the entire process of inpatient claims generation by first starting the treatment generation,
        injecting fraud, and then entering these claims into the DRG Grouper and using this output to calculate the claims.

        Args:
            n_patients (int, optional): Number of patients to be generated. Defaults to None.
            n_hospitals (int, optional): Number of hospitals to be generated. Defaults to 5.
            n_cases (int): Number of cases to be generated.
            state (str, optional): State the patients and hospitals to be located in. Defaults to None.
            random_state (int, optional): random state for reproducibility. Defaults to None.
            multiprocessing (int, optional): Number of cores to be used for multiprocessing. 0 for no multiprocessing. Defaults to 0.
        """
        config.create_json() # saving config as json for documentation
        self.plz = plzs.PLZ(random_state=random_state)
        self.drg = drgs.DRG(random_state=random_state)
        if n_patients == None:
            n_patients = n_cases * 2
        print("START SIMULATION")
        print("START TREATMENT GENERATION")
        if multiprocessing > 0:
            ids = list(range(n_cases))
            sublist_size = len(ids) // multiprocessing
            id_split = [
                ids[i : i + sublist_size] for i in range(0, len(ids), sublist_size)
            ]
            patients = treatment_generation.GENERATION(self.plz,
                random_state=random_state
            ).generate_patients(n_patients=n_patients, state=state)
            hospitals = treatment_generation.GENERATION(self.plz,
                random_state=random_state
            ).generate_hospitals(n_hospitals=n_hospitals, state=state)
            with Pool(multiprocessing) as p:
                args = [
                    (
                        random_state,
                        n_patients,
                        patients,
                        n_hospitals,
                        hospitals,
                        (n_cases // multiprocessing),
                    )
                ]
                lst = p.starmap(
                    self.generate_data, [(*args[0], id_list) for id_list in id_split]
                )  # p.map(lambda id_list: self.generate_data(args, id_list), id_split)
                result = pd.concat(lst, axis=0, ignore_index=True)
                result.to_csv(
                    "data/generated data/claims.csv", index=False
                )  # = claims.csv
        else:
            treatment_generation.GENERATION(self.plz, random_state=random_state).generate_data(
                n_patients=n_patients, n_hospitals=n_hospitals, n_cases=n_cases
            )

        print("FINISHED TREATMENT GENERATION -- START FRAUD INJECTION")
        inject_fraud.inject(config.fraudulent_hospitals)  # = claims_with_fraud.csv
        print("FINISHED FRAUD INJECTION -- START GROUPING")
        grouper_wrapping.Grouper().execute_grouping()  # = claims_with_drg.csv
        print("FINISHED GROUPING -- CALCULATING CLAIM")
        self.add_claim()  # = claims_final.csv
        print("FINISHED SIMULATION -- ENJOY!!")
        open(
            "data/generated data/grouper data/input.txt", "w"
        ).close()  # clear input.csv after each run


if __name__ == "__main__":
    Simulation(
        n_cases=config.n_cases,
        n_hospitals=config.n_hospitals,
        n_patients=config.n_patients,
        state=config.state,
        multiprocessing=8,
    )

