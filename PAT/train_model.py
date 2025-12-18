import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes
import time
# ==========================================
# 1. DEFINE THE DIGITAL TWIN (PyTorch Model)
# ==========================================
# Save this in train_model.py
class ChipModel(nn.Module):
    def __init__(self, num_heaters=40, num_outputs=6):
        super().__init__()
        # Inputs: 40 heaters
        # Outputs: 6 Control Lines (SelectA, SelectB, Load0...Load3)
        self.net = nn.Sequential(
            nn.Linear(num_heaters, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Deeper network for complex logic
            nn.ReLU(),
            nn.Linear(256, num_outputs) # Output layer has 6 neurons now!
        )

    def forward(self, x):
        return self.net(x)

# Note: In collect_data(), ensure you save ALL 6 channels, not just channel 3.
# vals = self.scopes.read_many() -> returns [ch1, ch2, ch3, ch4, ch5, ch6...]
# y_data.append(vals[0:6])
# ==========================================
# 2. DATA COLLECTION & TRAINING ROUTINE
# ==========================================
class ChipTrainer:
    def __init__(self):
        self.controller = DualAD5380Controller()
        # Setup scope for channel 3 (index 3) as per your config
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2,3,4])
        self.output_channel = 4 
        self.model = ChipModel()
        
    def collect_data(self, n_samples=1000):
        print(f"Collecting {n_samples} samples for system identification...")
        X_data = []
        y_data = []
        
        for i in range(n_samples):
            # Random uniform sampling 0.5V to 4.1V
            config = np.random.uniform(0.5, 4.1, 40)
            
            # Apply to hardware
            for h, v in enumerate(config):
                self.controller.set(h, v)
            
            # Wait for thermal settling
            time.sleep(0.05)   # <--- Fixed: was 'torch.time.sleep'
            
            # Measure
            vals = self.scopes.read_many(avg=1)
            
            # --- FIX IS HERE ---
            # We explicitly check "is not None" instead of just "if vals"
            if vals is not None and len(vals) > 3:
                val = vals[self.output_channel]
                X_data.append(config)
                y_data.append(val)
            # -------------------
                
            if i % 50 == 0:
                print(f"  {i}/{n_samples} collected")
                
        return np.array(X_data), np.array(y_data)

    def train_and_save(self, path="chip_model.pth"):
        X, y = self.collect_data(n_samples=5000) # Collect real data
        
        # Convert to Tensors
        tensor_x = torch.Tensor(X)
        tensor_y = torch.Tensor(y).unsqueeze(1)
        
        print("Training Digital Model...")
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3) #
        loss_fn = nn.MSELoss()
        
        # Simple training loop
        self.model.train()
        for epoch in range(500):
            optimizer.zero_grad()
            pred = self.model(tensor_x)
            loss = loss_fn(pred, tensor_y)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} Loss: {loss.item():.6f}")
                
        # Save the trained model
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}. You can now run the optimizer.")
        self.cleanup()

    def cleanup(self):
        try: self.scopes.close()
        except: pass

if __name__ == "__main__":
    trainer = ChipTrainer()
    trainer.train_and_save()