# Data Loader & VRAM Verification

## Code Used
```python
import torch
import pickle
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

DATA_PATH = os.path.join('data', 'qm9_voc_compliant.pkl')

def main():
    with open(DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Dataset loaded from {DATA_PATH}.")
    print(f"Total number of samples: {len(dataset)}")
    print(f"First element type: {type(dataset[0])}")

    # If dict, convert to Data objects
    if isinstance(dataset[0], dict):
        dataset = [Data(**d) for d in dataset]
        print("Converted dicts to PyG Data objects")

    loader = DataLoader(dataset, batch_size=250, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = next(iter(loader)).to(device)
    print(f"One batch moved to {device}")
    print(batch)

    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
```

## Output (Sample Run)
```
Dataset loaded from data/qm9_voc_compliant.pkl.
Total number of samples: 1000
First element type: <class 'dict'>
Converted dicts to PyG Data objects
One batch moved to cuda
DataBatch(y=[250], smiles=[250], atoms=[250])
Press Enter to exit...
```

## VRAM Check (via `nvidia-smi`)
Baseline (before loading batch):
```
GPU Memory Usage: 11 MiB / 4096 MiB
```

After moving batch (batch_size=250) to CUDA:
```
GPU Memory Usage: ~14 MiB / 4096 MiB
```
---
## Proof of Run
1. VRAM Usage before loading batch:
```
(tu9-mpp) kasif-ak@fedora:~/Projects/VOC-Compliance-Checker$ nvidia-smi
Thu Aug 21 16:05:07 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.05              Driver Version: 575.64.05      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2050        Off |   00000000:01:00.0 Off |                  N/A |
| N/A   49C    P8              4W /   30W |      11MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2796      G   /usr/bin/gnome-shell                      1MiB |
+-----------------------------------------------------------------------------------------+
```

2. VRAM usage after moving batch (batch_size=250) to CUDA:
```
(tu9-mpp) kasif-ak@fedora:~/Projects/VOC-Compliance-Checker$ nvidia-smi
Thu Aug 21 16:09:52 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.05              Driver Version: 575.64.05      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2050        Off |   00000000:01:00.0 Off |                  N/A |
| N/A   47C    P8              4W /   30W |      14MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2796      G   /usr/bin/gnome-shell                      1MiB |
+-----------------------------------------------------------------------------------------+
```
---
## Verification
- Data loader works with `batch_size=16` and larger (`250` tested).  
- VRAM usage confirmed to remain **well under 3.5 GB**.  
- Task 8 completed successfully.
---