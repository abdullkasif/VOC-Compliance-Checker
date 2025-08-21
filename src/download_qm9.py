#import necessary libraries
import os
import gc
import torch
import pandas as pd
from torch_geometric.datasets import QM9
import pickle

def main():
    print("Starting the download of the QM9 dataset...")

    # Define the path where the dataset will be saved
    path = os.path.join(os.getcwd(),'data')
    os.makedirs(path, exist_ok=True)

    # Download the QM9 dataset
    dataset = QM9(path)
    print(f"QM9 dataset downloaded and saved to {path}")
    print(f"Number of molecules in the dataset: {len(dataset)}")

    #----------------------------------------
    print("Creating a subset of VOC compliant molecules...")

    def is_voc_compliant(data):
        atomic_numbers = set(data.z.tolist())
        return atomic_numbers.issubset({1, 6, 7, 8}) # Hydrogen, Carbon, Nitrogen, Oxygen only

    voc_compliant = []
    for i in range(len(dataset)):
        mol = dataset[i]
        if is_voc_compliant(mol):
            voc_compliant.append({
                'smiles': mol.smiles,
                'atoms': mol.z.tolist(),
                'y': mol.y.squeeze().tolist(),
            })
        
        if len(voc_compliant) >= 1000:
            break
        if (i+1) % 5000 == 0:
            print(f"Processed {i+1} molecules...")

        gc.collect()  # Free up memory after processing each molecule

    print(f"Total VOC compliant molecules found: {len(voc_compliant)}") 

    # Save the VOC compliant molecules to a pkl file
    subset_path_pkl = os.path.join(path, 'qm9_voc_compliant.pkl')
    with open(subset_path_pkl, 'wb') as f:
        pickle.dump(voc_compliant, f, protocol=4)
    print(f"VOC compliant molecules saved to {subset_path_pkl}")

    #----------------------------------------
    # Save the VOC compliant molecules to a CSV file
    print("Creating CSV file for VOC compliant molecules...")

    csv_data = []

    for mol in voc_compliant:
        row = {
            "smiles": mol["smiles"],
            "atoms": " ".join(map(str, mol["atoms"])),
        }
        # Add targets
        for j, target_val in enumerate(mol["y"]):
            row[f"target_{j}"] = target_val
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)

    subset_path_csv = os.path.join(path, "qm9_voc_subset_1000_light.csv")
    df.to_csv(subset_path_csv, index=False)
    print(f"Saved CSV subset: {subset_path_csv}")

    # File size check
    file_size_pkl = os.path.getsize(subset_path_pkl) / (1024 * 1024)
    file_size_csv = os.path.getsize(subset_path_csv) / (1024 * 1024)
    print(f"Pickle file size: {file_size_pkl:.2f} MB")
    print(f"CSV file size: {file_size_csv:.2f} MB")

    # Verification
    print(" Verifying output format...")
    with open(subset_path_pkl, "rb") as f:
        loaded = pickle.load(f)
    print(f" Reloaded {len(loaded)} molecules from Pickle")
    print(" Example molecule (from Pickle):", loaded[0]["smiles"])
    print(" Example targets (first 3):", loaded[0]["y"][:3])
    print(" Example CSV columns:")
    print(df.head(1))
    print(f"CSV shape: {df.shape}")  # Should be (1000, 21) - 1000 rows, 21 columns

    print(f" CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    main()
    


