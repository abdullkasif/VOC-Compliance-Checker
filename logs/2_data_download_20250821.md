
# QM9 dataset download & subset creation (VOC-compliant molecules only)  

##  Objective
Download the **QM9 dataset** (a standard molecular benchmark) and extract a **subset of 1,000 VOC-compliant molecules** (C, H, N, O atoms only).  
Store results in both **Pickle** and **CSV** format for later tasks.

---

## Hardware & Environment
- **Machine**: Acer Aspire 5 (16GB RAM, RTX 2050 GPU)  
- **OS**: Linux (Ubuntu 22.04 recommended)  
- **Python**: 3.10 (inside `tu9-mpp` Conda environment)  
- **Key Libraries**:  
  - `torch`  
  - `torch-geometric`  
  - `pandas`  
  - `pickle`  

---

##  Code Used
```python
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
```

---

##  Output & Verification
- **Total dataset molecules**: ~130,000  
- **Subset extracted**: 1,000 VOC-compliant molecules (H, C, N, O only)  
- **Saved files**:  
  - `qm9_voc_compliant.pkl` (Pickle, ~0.24MB)  
  - `qm9_voc_subset_1000_light.csv` (CSV, ~0.39MB)  

**CSV shape**: `(1000, 21)` → 1,000 rows × 21 columns  
- `smiles`  
- `atoms`  
- `target_0` … `target_18`  

---

##  Proof of Run
```
(tu9-mpp) kasif-ak@fedora:~/Projects/VOC-Compliance-Checker$ python src/download_qm9.py 
Starting the download of the QM9 dataset...
Downloading https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
Extracting /home/kasif-ak/Projects/VOC-Compliance-Checker/data/raw/qm9.zip
Downloading https://ndownloader.figshare.com/files/3195404
Processing...
100%|███████████████████████████████████████████████████████████████████████████████████████████| 133885/133885 [01:55<00:00, 1155.40it/s]
Done!
QM9 dataset downloaded and saved to /home/kasif-ak/Projects/VOC-Compliance-Checker/data
Number of molecules in the dataset: 130831
Creating a subset of VOC compliant molecules...
Total VOC compliant molecules found: 1000
VOC compliant molecules saved to /home/kasif-ak/Projects/VOC-Compliance-Checker/data/qm9_voc_compliant.pkl
Creating CSV file for VOC compliant molecules...
Saved CSV subset: /home/kasif-ak/Projects/VOC-Compliance-Checker/data/qm9_voc_subset_1000_light.csv
Pickle file size: 0.24 MB
CSV file size: 0.39 MB
 Verifying output format...
 Reloaded 1000 molecules from Pickle
 Example molecule (from Pickle): [H]C([H])([H])[H]
 Example targets (first 3): [0.0, 13.210000038146973, -10.549854278564453]
 Example CSV columns:
              smiles      atoms  target_0  target_1   target_2  ...  target_14  target_15   target_16   target_17   target_18
0  [H]C([H])([H])[H]  6 1 1 1 1       0.0     13.21 -10.549854  ... -17.389656 -16.151918  157.711807  157.709976  157.706985

[1 rows x 21 columns]
CSV shape: (1000, 21)
CUDA available: True

 ```
 ---
