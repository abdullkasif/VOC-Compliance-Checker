# Data inspection & target properties documentation  

## Objective

The goal of this task is to **inspect the downloaded QM9 dataset subset
(1,000 molecules)** and verify its structure.\
We focus on three main checks:
1. Total number of molecules in the dataset
2. Atom count of the first molecule
3. Values of the 19 available quantum chemical targets (from QM9
dataset)

⚠️ **Important Note**:\
The QM9 dataset does **not provide vapor pressure directly**. Instead,
it includes 19 molecular properties (quantum descriptors).\
Later tasks will use these properties to **predict vapor pressure** for
VOC compliance checking.

------------------------------------------------------------------------

## Code Used

``` python
import pandas as pd
import os

# Load the dataset from the specified path
path = os.path.join(os.getcwd(),'data', 'qm9_voc_subset_1000_light.csv')
df = pd.read_csv(path)

print("QM9 VOC Subset Data Inspection")

# Total samples in dataset
print(f'Total samples in dataset: {len(df)}')

# FIrst Molecule's atom count
first_atom = df.iloc[0]["atoms"].split()
print(f'First molecule atom count: {len(first_atom)}')

# Show all 19 QM9 targets for the first molecule
targets = [col for col in df.columns if col.startswith('target_')]
print(f"Available target columns ({len(targets)} total): {targets}")
print("Target values for first molecule:")
print(df.iloc[0][targets].to_dict())


print("\n⚠️ Note: Vapor pressure is NOT a direct target in QM9.")
print("We will predict vapor pressure later using these 19 molecular properties.")

```

------------------------------------------------------------------------

## Expected Output

-   **Total samples in dataset**: `1000`
-   **Atom count of first molecule**: typically between `5-10` atoms
-   **List of 19 targets**: `target_0, target_1, ..., target_18`
-   **Target values for first molecule**: dictionary of property values

Example snippet:

    Total samples in dataset: 1000
    Atom count for first molecule: 9
    Available target columns (19 total): ['target_0', 'target_1', ..., 'target_18']
    Target values for first molecule:
    {'target_0': 0.234, 'target_1': 12.78, 'target_2': -0.245, ... }
    ⚠️ Note: Vapor pressure is NOT a direct target in QM9.

------------------------------------------------------------------------

## Relevance for VOC Compliance

-   The QM9 targets provide **quantum chemical descriptors** (dipole
    moment, polarizability, HOMO-LUMO gap, etc.).
-   These descriptors serve as **input features** for machine learning
    models.
-   Vapor pressure prediction will be based on these descriptors.
-   Ensures we stay aligned with **German TA Luft regulations** for VOC
    compliance.

------------------------------------------------------------------------

## Proof of Execution

```
(tu9-mpp) kasif-ak@fedora:~/Projects/VOC-Compliance-Checker$ python src/data_inspection.py 
QM9 VOC Subset Data Inspection
Total samples in dataset: 1000
First molecule atom count: 5
Available target columns (19 total): ['target_0', 'target_1', 'target_2', 'target_3', 'target_4', 'target_5', 'target_6', 'target_7', 'target_8', 'target_9', 'target_10', 'target_11', 'target_12', 'target_13', 'target_14', 'target_15', 'target_16', 'target_17', 'target_18']
Target values for first molecule:
{'target_0': 0.0, 'target_1': 13.210000038146973, 'target_2': -10.549854278564451, 'target_3': 3.186453342437744, 'target_4': 13.736308097839355, 'target_5': 35.36410140991211, 'target_6': 1.2176822423934937, 'target_7': -1101.48779296875, 'target_8': -1101.4097900390625, 'target_9': -1101.384033203125, 'target_10': -1102.02294921875, 'target_11': 6.468999862670898, 'target_12': -17.172182083129883, 'target_13': -17.286823272705078, 'target_14': -17.38965606689453, 'target_15': -16.151918411254883, 'target_16': 157.71180725097656, 'target_17': 157.70997619628906, 'target_18': 157.7069854736328}

⚠️ Note: Vapor pressure is NOT a direct target in QM9.
We will predict vapor pressure later using these 19 molecular properties.

```
---