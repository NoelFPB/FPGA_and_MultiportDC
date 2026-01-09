import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# --- MODEL DEFINITION ---
class ChipModel(nn.Module):
    def __init__(self, num_heaters=49, num_outputs=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_heaters, 512), # Wider
            nn.SiLU(),                   # Smoother than ReLU for interference
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

def train_offline():
    # 0. GPU / CPU Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps") # For Mac M1/M2
    print(f"Training on: {device}")

    # 1. LOAD DATA
    print("Loading 40k dataset...")
    data = np.load("chip_dataset.npz")
    X, y = data['X'], data['y']

    # 2. PREPARE TENSORS
    tx = torch.Tensor(X) / 5.0 # Scaled Inputs
    ty = torch.Tensor(y) /11.0      # Raw Volts Outputs

    # Split 80/20
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    # BATCH SIZE: 128 or 256 is the "sweet spot" for 40k samples
    train_loader = DataLoader(
        TensorDataset(tx[train_idx].to(device), ty[train_idx].to(device)), 
        batch_size=128, 
        shuffle=True
    )
    
    tx_val, ty_val = tx[val_idx].to(device), ty[val_idx].to(device)

    # 3. SETUP
    model = ChipModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Slower LR for large data
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    print("Starting Fast Training Loop...")
    for epoch in range(1000):
        model.train()
        train_running_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        # Validation every epoch
        model.eval()
        with torch.no_grad():
            v_pred = model(tx_val)
            v_loss = loss_fn(v_pred, ty_val)

        if epoch % 10 == 0:
            avg_train = train_running_loss / len(train_loader)
            print(f"Epoch {epoch:4d} | Train: {avg_train:.4f} | Val: {v_loss.item():.4f}")

        # SAVE BEST MODEL & EARLY STOPPING
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "decoder_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > 300: # Stop if no improvement for 100 epochs
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done! Best Validation Error: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_offline()