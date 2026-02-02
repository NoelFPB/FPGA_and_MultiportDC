import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. Setup MPS (M3 GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class HighResChipModel(nn.Module):
    def __init__(self, num_heaters=49, num_outputs=6):
        super().__init__()
        self.num_heaters = num_heaters
        # Expanded to 5 features: V, V^2, Sin, Cos, and Neighbor Interaction (V_i * V_i+1)
        self.expanded_dim = (num_heaters * 6) + (num_heaters - 1)
        
        # Physics Parameters
        # raw parameter (unbounded)
        self.phase_const_raw = nn.Parameter(torch.zeros(num_heaters))

        # choose a physical-ish range for phase_const
        self.phase_min = 0.1
        self.phase_max = 4.5
        self.phase_bias = nn.Parameter(torch.randn(num_heaters) * 0.1)

        self.net = nn.Sequential(
            nn.Linear(self.expanded_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_outputs),
            # --- CONSTRAINT: Force non-negative output ---
            nn.Softplus() 
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, v_std, v_raw):
        if self.training:
            v_raw = v_raw + torch.randn_like(v_raw) * 0.005
        v_sq = v_raw**2
        phase_const = self.phase_min + (self.phase_max - self.phase_min) * torch.sigmoid(self.phase_const_raw)
        phase = (phase_const * v_sq) + self.phase_bias
        
        # 2. Cumulative Phase (The "Traveler" Logic)
        cum_phase = torch.cumsum(phase, dim=1) 
        
        s_feat = torch.sin(phase)
        c_feat = torch.cos(phase)
        s_cum = torch.sin(cum_phase) # Global interference context
        c_cum = torch.cos(cum_phase)
        
        # 3. Neighbor Interaction
        v_interact = v_raw[:, :-1] * v_raw[:, 1:]
    
        x = torch.cat([v_std, v_std**2, s_feat, c_feat, s_cum, c_cum, v_interact], dim=1)
        return self.net(x)

def focal_photonic_loss(pred, target, gamma=2.0):
    mse = (pred - target) ** 2
    # Centers weight on peaks (0 and 1) where logic matters most
    weights = torch.pow(torch.abs(target - 0.5) * 2.0, gamma) + 0.1
    return (mse * weights).mean()

def train_offline():
    data = np.load("chip_dataset.npz")
    X, y = data['X'], data['y'] 
    X_raw = X.astype(np.float32)

    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_stdized = (X_raw - X_mean) / X_std
    X_tensor = torch.from_numpy(X_stdized).float()   # standardized for NN stability
    Xraw_tensor = torch.from_numpy(X_raw).float()    # raw volts for physics
    
    # Calculate min/range for each of the 6 channels separately
    y_min = y.min(axis=0) 
    y_max = y.max(axis=0)
    y_range = y_max - y_min + 1e-5
    Y_norm = (y - y_min) / y_range

    Y_tensor = torch.from_numpy(Y_norm).float()

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_loader = DataLoader(
    TensorDataset(
        X_tensor[indices[:split]],      # standardized X
        Xraw_tensor[indices[:split]],   # raw X
        Y_tensor[indices[:split]]
    ),
    batch_size=512,
    shuffle=True
    )

    val_x = X_tensor[indices[split:]].to(device)
    val_xraw = Xraw_tensor[indices[split:]].to(device)
    val_y = Y_tensor[indices[split:]].to(device)
    
    model = HighResChipModel(num_heaters=49).to(device)
    
    # Differential LR: Physics vs Network
    optimizer = optim.AdamW([
    {'params': [model.phase_const_raw, model.phase_bias], 'lr': 1e-2, 'weight_decay': 0},
    {'params': model.net.parameters(), 'lr': 5e-4, 'weight_decay': 1e-2}
], amsgrad=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[5e-3, 2e-3],
        steps_per_epoch=len(train_loader), epochs=1000
    )

    y_range_t = torch.tensor(y_range, device=device, dtype=torch.float32)
    y_min_t = torch.tensor(y_min, device=device, dtype=torch.float32)

    print(f"{'Epoch':<8} | {'Val Focal':<12} | {'RMSE (V)':<10}")
    
    for epoch in range(501):
        model.train()
        for batch_x, batch_xraw, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_xraw = batch_xraw.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x, batch_xraw)
            mse = nn.MSELoss()(pred, batch_y)

            foc = focal_photonic_loss(pred, batch_y)
            loss = 0.8*mse + 0.2*foc

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                alphas = (model.phase_min + (model.phase_max - model.phase_min) *
                        torch.sigmoid(model.phase_const_raw)).detach().cpu().numpy()
                pred = model(val_x, val_xraw)
                v_loss = focal_photonic_loss(pred, val_y).item()
                
                # De-normalize to calculate real-world metrics
                pred_v = (pred * y_range_t) + y_min_t
                actual_v = (val_y * y_range_t) + y_min_t
                err_v = pred_v - actual_v
                
                rmse_per_dim = torch.sqrt(torch.mean(err_v**2, dim=0))
                overall_rmse = torch.sqrt(torch.mean(err_v**2))
                rel_err_pct = (rmse_per_dim / y_range_t) * 100

                print(f"\nEpoch {epoch} | Focal Loss: {v_loss:.6f}")
                print(f"Alpha Range: {alphas.min():.2f} - {alphas.max():.2f}")
                print(f"Overall RMSE: {overall_rmse.item():.4f}V")
                print(f"Rel Error % per channel: {rel_err_pct.cpu().numpy()}")

                # Quick Logic Sample Check
                print(f"Sample 0 Logic (Expected vs Predicted):")
                print(f"  Exp: {actual_v[0].cpu().numpy()}")
                print(f"  Prd: {pred_v[0].cpu().numpy()}")
                print("-" * 50)


                pred_std = pred_v.std(dim=0).cpu().numpy()
                actual_std = actual_v.std(dim=0).cpu().numpy()
                print(f"Prediction Std Dev: {pred_std}")
                print(f"Actual Std Dev:     {actual_std}")

if __name__ == "__main__":
    train_offline()