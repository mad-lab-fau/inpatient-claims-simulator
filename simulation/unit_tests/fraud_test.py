import sys

sys.path.insert(0, "")

import unittest
import pandas as pd

import simulation.fraud.create_fraud as create_fraud
import simulation.fraud.inject_fraud as inject_fraud
import simulation.utility.grouper_wrapping as grouper_wrapping



class TestFraud(unittest.TestCase):
    """This class is used to test the fraud functions.
    
    Args:
        unittest (unittest.TestCase): the unittest class
    """
    def setUp(self):
        self.grouper = grouper_wrapping.Grouper()

    def test_increase_breathing(self):
        b_hours = 23
        coefficient = 0.1
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).increase_ventilation(b_hours, coefficient),
            25,
        )
        b_hours = 24
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).increase_ventilation(b_hours, coefficient),
            25,
        )
        b_hours = 22
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).increase_ventilation(b_hours, coefficient),
            22,
        )

    def test_decrease_weight(self):
        weight = 1000
        coefficient = 0.1
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).decrease_weight(weight, coefficient), 998
        )
        weight = 1080
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).decrease_weight(weight, coefficient), 998
        )
        weight = 1240
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).decrease_weight(weight, coefficient), 1240
        )

    def test_change_to_caesarean(self):
        primary = "O80"
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).change_to_caesarean(primary), "O82"
        )
        primary = "O81"
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).change_to_caesarean(primary), None
        )

    def test_change_ICD_order(self):
        primary = "A01"
        secondary = ["A02", "A03"]
        id = 1
        temp_comb = pd.DataFrame()
        result = create_fraud.FRAUD(self.grouper).change_ICD_order(
            id, primary, secondary, temp_comb
        )
        result_series = result["ICD"].tolist()[1]
        self.assertTrue((result_series == ["A02", "A01", "A03"]))
        result_series = result["ICD"].tolist()[2]
        self.assertTrue((result_series == ["A03", "A01", "A02"]))

    def test_newborn_add_personal_care(self):
        age = 0
        secondary = ["A01", "A02", "A03"]
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).newborn_add_personal_care(age, secondary),
            ["A01", "A02", "A03", "Z74.1"],
        )
        age = 1
        secondary = ["A01", "A02", "A03"]
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).newborn_add_personal_care(20, secondary),
            ["A01", "A02", "A03"],
        )
        age = 0
        secondary = ["A01", "A02", "A03", "Z74.1"]
        self.assertEqual(
            create_fraud.FRAUD(self.grouper).newborn_add_personal_care(age, secondary),
            ["A01", "A02", "A03", "Z74.1"],
        )

    def test_extract_primary_secondary(self):
        id = 1
        result = pd.DataFrame({'ID': [1, 1, 2, 3, 3], 'EFF_WEIGHT': [0.5, 0.6, 0.6, 0.8, 0.85]})
        combinations = pd.DataFrame({'ICD': [["A01", "A02", "A03"], ["B01", "B02", "B03"], ["C01", "C02", "C03"], ["D01", "D02", "D03"], ["Z01", "E02", "E03"]]})

        # Call the extract_primary_secondary method with test data
        primary, secondary = inject_fraud.extract_primary_secondary(id, result, combinations)

        # Check the returned values
        self.assertEqual(primary, "B01")
        self.assertEqual(secondary, ["B02", "B03"])

        id = 2
        primary, secondary = inject_fraud.extract_primary_secondary(id, result, combinations)
        self.assertEqual(primary, "C01")
        self.assertEqual(secondary, ["C02", "C03"])

        id = 3
        primary, secondary = inject_fraud.extract_primary_secondary(id, result, combinations)
        self.assertEqual(primary, "D01")
        self.assertEqual(secondary, ["D02", "D03"])




if __name__ == "__main__":
    unittest.main()
