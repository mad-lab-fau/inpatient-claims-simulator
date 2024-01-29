import sys

sys.path.insert(0, "")

import unittest
import pandas as pd

import simulation.generation.icd as icd
import simulation.generation.adjust_coding as adjust_coding
import simulation.utility.utils as utils
import simulation.utility.icd_ops_mapping as icd_ops_mapping


class TestICD_OPS(unittest.TestCase):
    """This class is used to test the Coding Guidelines."""
    def test_main_diagnose_R75(self):
        main_diagnose = "R75"
        diagnoses = ["A01", "B20"]
        main_diagnose, diagnoses = adjust_coding.Adjustments().check_validity_HIV(
            main_diagnose, diagnoses
        )
        self.assertEqual(main_diagnose, "B24")
        self.assertEqual(diagnoses, ["A01", "B20"])

    def test_excluded_codes(self):
        main_diagnose = "A01"
        diagnoses = ["A02", "B20", "B21"]
        main_diagnose, diagnoses = adjust_coding.Adjustments().check_validity_HIV(
            main_diagnose, diagnoses
        )
        self.assertEqual(diagnoses, ["A02", "B20"])
        self.assertEqual(main_diagnose, "A01")

    def test_combined_HIV(self):
        main_diagnose = "R75"
        diagnoses = ["A02", "B20", "B21"]
        main_diagnose, diagnoses = adjust_coding.Adjustments().check_validity_HIV(
            main_diagnose, diagnoses
        )
        self.assertEqual(main_diagnose, "B24")
        self.assertEqual(diagnoses, ["A02", "B20"])

    def test_remove_duplicate_operation(self):
        diagnoses = ["A01", "B20"]
        operations = ["8-541.4", "8-541.4"]
        adjust_coding.Adjustments().check_zytotoxic_materials(diagnoses, operations)
        self.assertEqual(operations, ["8-541.4"])

    def test_remove_diagnose_Z511(self):
        diagnoses = ["A01", "Z51.1"]
        operations = ["8-541.4"]
        adjust_coding.Adjustments().check_zytotoxic_materials(diagnoses, operations)
        self.assertEqual(diagnoses, ["A01"])

    def test_remove_E538(self):
        diagnoses = ["A01", "D52.0", "E53.8"]
        primary_diagnose = "E01"
        result = adjust_coding.Adjustments().check_folic_acid(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result, ["A01", "D52.0"])

    def test_do_not_remove_E538(self):
        diagnoses = ["A01", "E53.8"]
        primary_diagnose = "E01"
        result = adjust_coding.Adjustments().check_folic_acid(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result, ["A01", "E53.8"])

    def test_multiple_folic(self):
        diagnoses = ["A01", "D52.1", "D52.8"]
        primary_diagnose = "E01"
        result = adjust_coding.Adjustments().check_folic_acid(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result, ["A01", "D52.1", "D52.8"])

    def test_add_comorbidity(self):
        diagnoses = ["A01", "E10.74"]
        primary_diagnose = "E01"
        result = adjust_coding.Adjustments().add_probable_diabetic_comorbidities(
            primary_diagnose, diagnoses, 0.7
        )
        self.assertEqual(len(result), 3)

    def test_do_not_add_comorbidity(self):
        diagnoses = ["A01"]
        primary_diagnose = "E01"
        result = adjust_coding.Adjustments().add_probable_diabetic_comorbidities(
            primary_diagnose, diagnoses, 0.7
        )
        self.assertEqual(len(result), 1)

    def test_change_main_diagnose(self):
        main_diagnose = "E16.0"
        diagnoses = ["A01", "E10.74"]
        result = adjust_coding.Adjustments().no_metabolic_when_diabetes(
            main_diagnose, diagnoses, gender="M", age=25, zip=80339
        )
        self.assertNotEqual(result[0], "E160")

    def test_do_not_change_main_diagnose(self):
        main_diagnose = "A01"
        diagnoses = ["A01", "E10.74"]
        result = adjust_coding.Adjustments().no_metabolic_when_diabetes(
            main_diagnose, diagnoses, gender="M", age=25, zip=80339
        )
        self.assertEqual(result[0], "A01")

    def test_add_operation(self):
        pump = 1
        operations = ["A01"]
        result = adjust_coding.Adjustments().add_ops_on_pump(
            pump, operations, "8-851.4"
        )
        self.assertEqual(result, ["A01", "8-851.4"])

    def test_do_not_add_operation_bc_already_there(self):
        pump = 1
        operations = ["A01", "8-851.4"]
        result = adjust_coding.Adjustments().add_ops_on_pump(pump, operations)
        self.assertEqual(result, ["A01", "8-851.4"])

    def test_do_not_add_operation_bc_no_pump(self):
        pump = 0
        operations = ["A01"]
        result = adjust_coding.Adjustments().add_ops_on_pump(pump, operations)
        self.assertEqual(result, ["A01"])

    def test_add_ops_on_revisioned_heart(self):
        operations = ["5-352.00"]
        result = adjust_coding.Adjustments().add_ops_on_revisioned_heart(operations)
        self.assertEqual(result, ["5-352.00", "5-379.5"])

        operations = ["5-352.00", "5-379.5"]
        result = adjust_coding.Adjustments().add_ops_on_revisioned_heart(operations)
        self.assertEqual(result, ["5-352.00", "5-379.5"])

        operations = []
        result = adjust_coding.Adjustments().add_ops_on_revisioned_heart(operations)
        self.assertEqual(result, [])

    def test_add_relevant_icd_birth(self):
        diagnoses = ["Z11"]
        primary_diagnose = "O80"
        result = adjust_coding.Adjustments().add_relevant_icd_birth(
            primary_diagnose, diagnoses
        )
        self.assertTrue(result[-1].startswith("Z37"))

        diagnoses = ["O80", "Z37.0"]
        result = adjust_coding.Adjustments().add_relevant_icd_birth(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result, ["O80", "Z37.0"])

        diagnoses = []
        primary_diagnose = "Z80"
        result = adjust_coding.Adjustments().add_relevant_icd_birth(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result, [])

    def test_add_relevant_icd_babies_hie(self):
        diagnoses = ["P91.3"]
        primary_diagnose = "A00"
        result = adjust_coding.Adjustments().add_relevant_icd_babies_hie(
            primary_diagnose, diagnoses
        )
        self.assertTrue(result[-1].startswith("P2"))

        diagnoses = ["P91.4"]
        result = adjust_coding.Adjustments().add_relevant_icd_babies_hie(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result[-2], "P90")
        self.assertTrue(result[-1].startswith("P2"))

        diagnoses = ["P91.5"]
        result = adjust_coding.Adjustments().add_relevant_icd_babies_hie(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result[-2], "P90")
        self.assertTrue(result[-1].startswith("P2"))

        diagnoses = ["B02", "P91.5", "A01"]
        result = adjust_coding.Adjustments().add_relevant_icd_babies_hie(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result[-2], "P90")
        self.assertTrue(result[-1].startswith("P2"))

        diagnoses = []
        result = adjust_coding.Adjustments().add_relevant_icd_babies_hie(
            primary_diagnose, diagnoses
        )
        self.assertEqual(result, [])

    def test_get_age_group(self):
        self.assertEqual(utils.get_age_group(0), "unter 1 Jahr")
        self.assertEqual(utils.get_age_group(3), "1 bis unter 5 Jahre")
        self.assertEqual(utils.get_age_group(7), "5 bis unter 10 Jahre")
        self.assertEqual(utils.get_age_group(14), "10 bis unter 15 Jahre")
        self.assertEqual(utils.get_age_group(17), "15 bis unter 18 Jahre")

    def test_get_gender_group(self):
        self.assertEqual(utils.get_gender_group("M"), "männlich")
        self.assertEqual(utils.get_gender_group("F"), "weiblich")
        self.assertEqual(utils.get_gender_group("x"), "Insgesamt")

    def test_correct_sectio(self):
        primary = "O82"
        secondary = ["A01", "O01"]
        operations = ["8-851.4"]
        result = adjust_coding.Adjustments().correct_sectio(
            22, "F", primary, secondary, operations
        )
        self.assertTrue(any(value.startswith("5-74") for value in result))

        primary = "A01"
        secondary = ["A01", "O82"]
        result = adjust_coding.Adjustments().correct_sectio(
            22, "F", primary, secondary, operations
        )
        self.assertTrue(any(value.startswith("5-74") for value in result))

        primary = "O82"
        secondary = ["A01", "O01"]
        operations = ["5-740.0"]
        self.assertEqual(sum(value.startswith("5-74") for value in operations), 1)

    def test_correct_sectio_no_sectio(self):
        primary = "O80"
        secondary = ["A01", "O01"]
        operations = ["8-851.4"]
        result = adjust_coding.Adjustments().correct_sectio(
            22, "F", primary, secondary, operations
        )

        self.assertFalse(any(value.startswith("5-74") for value in result))

    def test_check_frustrane(self):
        primary = "O47.9"
        self.assertNotEqual(
            adjust_coding.Adjustments().check_frustrane(primary), "O47.9"
        )

        primary = "O47.0"
        self.assertEqual(adjust_coding.Adjustments().check_frustrane(primary), "O47.0")

    def test_replace_old_myocard_infarct(self):
        primary = "I25.21"
        self.assertEqual(
            adjust_coding.Adjustments().replace_old_myocard_infarct(primary), "I25.8"
        )

    def test_replace_fluid_surpluss(self):
        primary = "E87.7"
        self.assertNotEqual(
            adjust_coding.Adjustments().replace_fluid_surpluss(primary), "E87.7"
        )

    def test_check_Z01_diagnoses(self):
        primary = "Z01.6"
        self.assertNotEqual(
            adjust_coding.Adjustments().check_Z01_diagnoses(primary), "Z01.6"
        )

        primary = "Z01.7"
        self.assertNotEqual(
            adjust_coding.Adjustments().check_Z01_diagnoses(primary), "Z01.7"
        )

    def test_replace_unknown_examination(self):
        primary = "Z04.9"
        self.assertNotEqual(
            adjust_coding.Adjustments().replace_unknown_examination(primary), "Z04.9"
        )

    def test_replace_senile_degeneration(self):
        primary = "G31.1"
        self.assertEqual(
            adjust_coding.Adjustments().replace_senile_degeneration(primary), "G31.9"
        )

        primary = "G31.2"
        self.assertEqual(
            adjust_coding.Adjustments().replace_senile_degeneration(primary), "G31.2"
        )

    def test_replace_chemo(self):
        primary = "Z51.1"
        self.assertNotEqual(adjust_coding.Adjustments().replace_chemo(primary), "Z51.1")
        primary = "Z51.9"
        self.assertEqual(adjust_coding.Adjustments().replace_chemo(primary), "Z51.9")
    

    def test_change_gender_for_pregnancy(self):
        primary = "O80"
        secondary = ["A01", "O01"]
        gender = "F"
        self.assertEqual(adjust_coding.Adjustments().change_gender_for_pregnancy(gender, primary, secondary), "F")

        primary = "A02"
        self.assertEqual(adjust_coding.Adjustments().change_gender_for_pregnancy(gender, primary, secondary), "F")

        gender = "M"
        self.assertEqual(adjust_coding.Adjustments().change_gender_for_pregnancy(gender, primary, secondary), "F")

        primary = "O80"
        secondary = ["A01"]
        self.assertEqual(adjust_coding.Adjustments().change_gender_for_pregnancy(gender, primary, secondary), "F")

        primary = "A02"
        self.assertEqual(adjust_coding.Adjustments().change_gender_for_pregnancy(gender, primary, secondary), "M")

    def TestICD_OPS_Mapping(self):
        self.assertEqual(icd_ops_mapping.get_ops("G"), ["5-01", "5-02", "5-03", "5-04", "5-05"])
        self.assertEqual(icd_ops_mapping.get_ops("G10"), ["5-01", "5-02", "5-03", "5-04", "5-05"])
        self.assertEqual(icd_ops_mapping.get_ops("G10.0"), ["5-01", "5-02", "5-03", "5-04", "5-05"])
        self.assertEqual(icd_ops_mapping.get_ops("A10"), None)
        self.assertEqual(icd_ops_mapping.get_ops("C73"), ["5-06", "5-07"])
        self.assertEqual(icd_ops_mapping.get_ops("C73.0"), ["5-06", "5-07"])
        
    """def test_get_randim_primary_icd_all(self):
        def test_get_random_primary_icd(age, gender, plz):
            icdo = icd.ICD()

            prob = icdo.get_primary_probability(age, gender, plz, "Z11")
            print(prob)
            n=1000
            arr=[]

            for i in range(n):
                arr.append(icdo.get_random_primary_icd(age, gender, plz))
            
            count = arr.count("Z11")
            self.assertAlmostEqual(count, round(prob*n), delta=n/100)

        genders = ["M", "F", "x"]
        ages = [0, 3, 7, 14, 17, 19, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 100]
        plz_list = [69115, 80331, 10115, 14467, 28195, 20095, 60311, 30159, 19053, 40213, 55116, 66111, 1067, 39104, 24103, 99084]

        for g in genders:
            for a in ages:
                for p in plz_list:
                    test_get_random_primary_icd(a, g, p)
                    print(str(a) + " " + g + " " + str(p) + " done")"""


if __name__ == "__main__":
    unittest.main()
