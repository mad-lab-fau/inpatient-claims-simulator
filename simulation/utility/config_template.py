import json

grouper = (
    "'C:/Program Files (x86)/IMC clinicon GmbH/IMC Navigator/"  # path to grouper.exe
)

n_cases = 1000000  # number of cases to be generated
n_patients = round(2 * n_cases)  # number of patients to be generated
n_hospitals = 300  # number of hospitals to be generated
state = "Bayern"  # state the patients and hospitals to be located in, None if no state
ventilation_to_linear = (
    2500  # number of ventilation hours when to switch to linear distribution
)

probability_diabetic_comorbidity = 0.4  # probability of a diabetic comorbidity

# every 30th hospital in a range of 300 is fraudulent
fraudulent_hospitals = [
    i for i in range(0, n_hospitals, 3)
]  # list of fraudulently acting hospitals

fraud_coefficient_ventilation = (
    0.12  # coefficient for the fraud threshold in ventilation hours
)
fraud_coefficient_weight = 0.12  # coefficient for the fraud threshold in weight
ratio_change_bloody = (
    0.5  # 0 for only bloody, 1 for only change in ICD order, 0.5 for both
)
fraud_probability = 0.2  # probability of a fraudulent case in a fraudulent hospital


def create_json():
    """Creates a JSON file with the parameters for the simulation."""
    parameters = {
        "n_cases": n_cases,
        "n_patients": n_patients,
        "n_hospitals": n_hospitals,
        "state": state,
        "ventilation_to_linear": ventilation_to_linear,
        "probability_diabetic_comorbidity": probability_diabetic_comorbidity,
        "fraudulent_hospitals": fraudulent_hospitals,
        "fraud_coefficient_ventilation": fraud_coefficient_ventilation,
        "fraud_coefficient_weight": fraud_coefficient_weight,
        "ratio_change_bloody": ratio_change_bloody,
        "fraud_probability": fraud_probability,
    }

    # Convert the parameters dictionary to a JSON string
    json_string = json.dumps(parameters, indent=4)

    # Write the JSON string to a file
    with open("data/generated data/parameters.json", "w") as json_file:
        json_file.write(json_string)
