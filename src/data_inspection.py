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
