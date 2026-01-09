import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# --- UPDATED MODEL DEFINITION ---
class ChipModel(nn.Module):
    def __init__(self, num_heaters=49, num_outputs=6):
        super().__init__()
        # Tighter architecture to prevent overfitting on 5k-40k samples
        self.net = nn.Sequential(
            nn.Linear(num_heaters, 128), 
            nn.SiLU(), 
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

def train_offline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. LOAD DATA
    data = np.load("chip_dataset.npz")
    X, y = data['X'], data['y']

    # 2. SPLIT DATA (Before normalization to prevent data leakage)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    x_train_raw, x_val_raw = X[train_idx], X[val_idx]
    y_train_raw, y_val_raw = y[train_idx], y[val_idx]

    # 3. CALCULATE Z-SCORE STATISTICS (ON TRAIN SET ONLY)
    # Adding a tiny epsilon (1e-8) to prevent division by zero
    x_mean, x_std = x_train_raw.mean(axis=0), x_train_raw.std(axis=0) + 1e-8
    y_mean, y_std = y_train_raw.mean(axis=0), y_train_raw.std(axis=0) + 1e-8

    # 4. APPLY NORMALIZATION
    # This "zooms in" on the interference wiggles by removing the large DC offsets
    x_train = (x_train_raw - x_mean) / x_std
    y_train = (y_train_raw - y_mean) / y_std
    x_val = (x_val_raw - x_mean) / x_std
    y_val = (y_val_raw - y_mean) / y_std

    # 5. SAVE STATISTICS (CRITICAL: You need these for the Digital Twin later)
    np.savez("normalization_stats.npz", 
             x_mean=x_mean, x_std=x_std, 
             y_mean=y_mean, y_std=y_std)

    # 6. PREPARE DATALOADERS
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train).to(device), 
                      torch.FloatTensor(y_train).to(device)), 
        batch_size=128, 
        shuffle=True
    )
    
    tx_val = torch.FloatTensor(x_val).to(device)
    ty_val = torch.FloatTensor(y_val).to(device)

    # 7. SETUP WITH WEIGHT DECAY (Regularization)
    model = ChipModel().to(device)
    # Weight decay forces the model to stay "smooth" like a sine wave
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    print("Starting Optimized Training Loop...")
    for epoch in range(2000): # Increased max epochs since LR is lower
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

        if epoch % 20 == 0:
            avg_train = train_running_loss / len(train_loader)
            # Convert back to Volts for physical understanding
            rmse_volts = np.sqrt(v_loss.item()) * np.mean(y_std)
            print(f"Epoch {epoch:4d} | Train: {avg_train:.4f} | Val: {v_loss.item():.4f} | RMSE: {rmse_volts:.3f}V")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "decoder_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > 150: 
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done! Best Normalized Val Loss: {best_val_loss:.6f}")
    final_rmse = np.sqrt(best_val_loss.cpu().item()) * np.mean(y_std)
    print(f"Final Physical Error: ±{final_rmse:.4f} Volts")

if __name__ == "__main__":
    train_offline()