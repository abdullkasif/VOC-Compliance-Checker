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
