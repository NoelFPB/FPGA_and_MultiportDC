import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Use the same model class as before
class ChipModel(nn.Module):
    def __init__(self, num_heaters=21, num_outputs=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_heaters, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_outputs)
        )
    def forward(self, x):
        return self.net(x)

def train_offline():
    # 1. LOAD DATA
    print("Loading dataset...")
    data = np.load("chip_dataset.npz")
    X, y = data['X'], data['y']

    # 2. PREPARE TENSORS
    tx = torch.Tensor(X) / 5.0 # Scaled Inputs
    ty = torch.Tensor(y)       # Raw Volts Outputs

    # Split 80/20
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(TensorDataset(tx[train_idx], ty[train_idx]), batch_size=64, shuffle=True)
    tx_val, ty_val = tx[val_idx], ty[val_idx]

    # 3. SETUP
    model = ChipModel()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting Offline Training...")
    for epoch in range(2000):
        model.train()
        train_running_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            v_pred = model(tx_val)
            v_loss = loss_fn(v_pred, ty_val)

        if epoch % 50 == 0:
            avg_train = train_running_loss / len(train_loader)
            print(f"Epoch {epoch:4d} | Train: {avg_train:.6f} | Val: {v_loss.item():.6f}")

        # EARLY STOPPING LOGIC
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "decoder_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve > 200:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training Complete. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_offline()