# Inpatient Claim Simulation and Fraud Detection
<!-- badges: start -->
![GitHub](https://img.shields.io/github/license/mad-lab-fau/inpatient-claims-simulator)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10581728.svg)](https://doi.org/10.5281/zenodo.10581728)
<!-- badges: end -->
This is the repository for the "Simulation and Detection of Healthcare Fraud in German Inpatient Claims Data" paper submitted to ICCS 2024 in the Health Thematic Track.

## Description

This project contains two parts, [Claims Simulation](simulation/) and [Fraud Detection](detection/).

The **Simulator** generates German inpatient claims according to the regulations valid in 2021. Based on this data, claims are changed in a fraudulent way. 

The fraud types included are: 
1. Increases in ventilation hours
2. Changing vaginal births to cesarean sections
3. Decreasing the weight of newborns
4. Adding the need for personal care to a newborn's treatment
4. Releasing people too early from hospital (bloody release)
5. Change the order of ICD codes 

Factors not simulated:
- no inpatient ward
- the outcome of a treatment (cured, death, etc.) is not simulated
- vacations during long hospital stays are not simulated
- the reason for admissions is not simulated


The **Detection** uses the generated data to train models. Tested algorithms (from Scikit-Learn):
- [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Multi-Layer Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

The models with the best results are Gradient Boosting and Random Forest.


## Visuals

### Claims Simulation
**1. Start Simulation:** Patients and Hospitals
![Generating patients and hospitals](doc/svg/Sequence%20Diagram%201.svg)

**2. Initialize Treatment:** Get ICD- and OPS-Codes, ventilation, duration
![Initialize Treatment](doc/svg/Sequence%20Diagram%202.svg)

**3. Adjust Treatments:** to coding guidelines 
![First adjustment to coding guidelines](doc/svg/Sequence%20Diagram%203.svg)

**4. Inject Fraud:** following the fraud patterns
![Inject Fraud](doc/svg/Sequence%20Diagram%204.svg)

**5. Finishing up:** adjusting the fraudulent claims to coding guidelines and calculating claims
![Finishing up](doc/svg/Sequence%20Diagram%205.svg)


More visualizations and UML diagrams can be found in the directory [doc](/doc/svg/).


## Installation

1. Download this repository
2. Install requirements with pip:
```
pip install -r requirements.txt
```
3. Install a DRG-Grouper (here the grouper from IMC Clinicon is used (https://www.imc-clinicon.de/tools/imc-navigator/index_ger.html))
4. Adjust [config_template.py](simulation/utility/config_template.py) to your requirements and save it as config.py

**IMPORTANT:** This project is built and tested with Python 3.9!

## Usage
### Generation
After installing the code and adjusting the config_template.py as described in [Installation](#installation)

In case you want to use another DRG-Grouper, you need to modify [grouper_wrapping.py](simulation/utility/grouper_wrapping.py) accordingly.

If everything is set up, execute from the project's root directory:
```
python simulation/simulate.py
```

Make sure, you configured your config.py correctly.

If everything works, several .csv-files are generated in the directory [data/generated data](/data/generated%20data/):
1. claims.csv: initial inpatient treatments, not containing fraud, DRGs, and claims
2. claims_with_fraud.csv: claims.csv with injected fraud
3. claims_with_drg.csv: claims_with_fraud.csv after grouping the treatments
4. **claims_final.csv: final inpatient treatments**

### Detection
First preprocess your data according to [preprocessing.py](detection/preprocessing.py). Then select your classifier by commenting everything else (if you want to train all in one run, do not change anything). To train the models execute
```
python detection/classifying.py
```

The models trained are saved in the directory [models](/data/models/).

## Data
The simulated data used for training the machine learning algorithms can be accessed at [zenodo.org](https://zenodo.org/records/10581728) 

## Support

In case questions occur, contact me or create an issue.

## Roadmap

This code is not maintained anymore. Further necessary developments:
- Improve the OPS-Code generation
- Model the treatment outcome
- Simulate inpatient ward (via simulating outpatient treatment)
- etc.

## Authors and acknowledgment

Special thanks to my supervisors René Raab, Kai Klede and Prof. Dr. Bjoern Eskofier. 

Furthermore, thanks to [AOK Bayern](https://www.aok.de/pk/bayern/) and Dominik Schirmer for providing the necessary validation data.

Thanks to [IMC Clinicon](https://www.imc-clinicon.de/) and Gunter Damian for giving me access to IMC Navigator, a certified DRG Grouper.

## Project status

Until further notice, the development of this project stopped after 29.11.2023. Feel free to contact me (see [Support](#support)), if you have ideas and use cases for collaboration.