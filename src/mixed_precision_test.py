import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

DATA_PATH = os.path.join('data', 'qm9_voc_compliant.pkl')

class DummyModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    
    print("Stimulating mixed precision training with dummy data...")

    # Specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instead of using dataset[0].x, we create random features
    input_dim = 32  # arbitrary feature size
    model = DummyModel(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    print("Model initialized.")

    # Instead of using batch.num_nodes, simulate batch_size features
    batch_size = 100000  # use larger loader batch size
    print(f"Simulating batch size: {batch_size}")
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randn(batch_size, 1, device=device)

    # --- FP32 Training Step ---
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    fp32_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"[FP32] Peak VRAM usage: {fp32_mem:.2f} MB")

    # ----------- FP16 Training Step with AMP -----------
    scaler = torch.amp.GradScaler(device=device) 
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        out = model(x)
        loss = loss_fn(out, y)     

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    fp16_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"[FP16 AMP] Peak VRAM usage: {fp16_mem:.2f} MB")

    # ----------- Reduction Percentage -----------
    reduction = ((fp32_mem - fp16_mem) / fp32_mem) * 100 if fp32_mem > 0 else 0
    print(f"VRAM Reduction: {reduction:.2f}%")

    input("Press Enter to exit...")
    
if __name__ == "__main__":
    main()