import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class RobustChipModel(nn.Module):
    def __init__(self, num_heaters=49, num_outputs=6):
        super().__init__()
        # Small, regularized network to handle sparse 49D data
        self.net = nn.Sequential(
            nn.Linear(num_heaters, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2), # Stop the model from memorizing random noise
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

def train_offline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load("chip_dataset.npz")
    X, y = data['X'], data['y']

    # MIN-MAX SCALE X (0.1V - 4.9V -> 0.0 - 1.0)
    X_scaled = (X - 0.1) / (4.9 - 0.1)
    
    # Z-SCORE Y
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-8
    Y_scaled = (y - y_mean) / y_std

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_scaled[indices[:split]]).to(device), 
                                            torch.FloatTensor(Y_scaled[indices[:split]]).to(device)), 
                              batch_size=256, shuffle=True)
    
    tx_val = torch.FloatTensor(X_scaled[indices[split:]]).to(device)
    ty_val = torch.FloatTensor(Y_scaled[indices[split:]]).to(device)

    model = RobustChipModel().to(device)
    # VERY LOW LEARNING RATE to prevent divergence
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    print("Epoch | Val Loss | RMSE (Volts)")
    for epoch in range(500):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                v_loss = loss_fn(model(tx_val), ty_val).item()
                rmse = np.sqrt(v_loss) * np.mean(y_std)
                print(f"{epoch:5d} | {v_loss:.4f} | {rmse:.4f}V")

                
if __name__ == "__main__":
    train_offline()