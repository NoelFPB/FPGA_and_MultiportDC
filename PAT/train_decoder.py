import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pyvisa 

#MAYBE OBSOLETE

# Import your hardware libraries
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes

# --- CONFIGURATION ---
NUM_HEATERS = 49        # Updated to match your new structure
NUM_OUTPUTS = 6         # SelB, SelA, L0, L1, L2, L3
SCOPE_INDICES = [0, 1, 2, 3, 4, 5] # Which scope channels to record (Indices 0-7)

# ==========================================
# 1. MULTI-OUTPUT MODEL
# ==========================================
class ChipModel(nn.Module):
    def __init__(self, num_heaters=NUM_HEATERS, num_outputs=NUM_OUTPUTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_heaters, 128),
            nn.Tanh(), # Tanh is smoother than ReLU for interference curves
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. ROBUST TRAINER (MULTI-CHANNEL)
# ==========================================
class ChipTrainer:
    def __init__(self):
        self.check_connection()

        self.controller = DualAD5380Controller()
        
        print("Connecting to Scopes...")
        # Assuming you need 8 channels total to cover the 6 outputs safely
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2,3,4])
        
        # Define which scope indices correspond to your 6 Decoder Outputs
        # [Index 0 = Scope1 Ch1, Index 4 = Scope2 Ch1, etc.]
        self.target_indices = SCOPE_INDICES
        
        self.model = ChipModel(num_heaters=NUM_HEATERS, num_outputs=NUM_OUTPUTS) 
        
    def check_connection(self):
        try:
            rm = pyvisa.ResourceManager()
            print("--- VISA RESOURCE SCAN ---")
            list_res = rm.list_resources()
            for res in list_res:
                print(f"Found: {res}")
            print("--------------------------")
        except:
            print("Could not scan VISA resources.")

    def collect_data(self, n_samples=1000):
        print(f"Collecting {n_samples} samples for {NUM_OUTPUTS} outputs...")
        X_data = []
        y_data = []
        
        # Inside collect_data
        for i in range(n_samples):
            if i < n_samples // 2:
                # 50% purely random (good for general physics)
                config = np.random.uniform(0.1, 4.9, NUM_HEATERS)
            else:
                # 50% "Biased" samples:
                # Keep 42 trainable heaters random, but force the 5 inputs 
                # to be exactly 0.5V or 4.1V
                config = np.random.uniform(0.1, 4.9, NUM_HEATERS)
                for h_inp in [42, 43, 44, 45, 46]:
                    config[h_inp] = 0.5 if np.random.random() > 0.5 else 4.1
            
            # 2. Hardware Set
            for h, v in enumerate(config):
                self.controller.set(h, v)
            
            # 3. Thermal Sleep
            time.sleep(0.2) 
            
            # 4. Read All Channels
            try:
                # Returns list of 8 floats (if 2 scopes x 4 channels)    
                vals = self.scopes.read_many(avg=1)
                
                if vals is not None and len(vals) > max(self.target_indices):
                    # Extract only the 6 relevant channels
                    current_outputs = [vals[idx] for idx in self.target_indices]
                    
                    # Sanity check: Ensure no value is garbage (e.g. infinite or >12V)
                    if all(abs(v) < 12.0 for v in current_outputs):
                        X_data.append(config)
                        y_data.append(current_outputs) # Append the VECTOR
            except Exception as e:
                print(f"Read error at step {i}: {e}")
                continue
                
            if i % 50 == 0:
                print(f"  {i}/{n_samples} collected")
                
        return np.array(X_data), np.array(y_data)

    def train_and_save(self, path="decoder_model.pth"):
            # 1. Collect Data
            X, y = self.collect_data(n_samples=10000)
            
            # 2. Split indices 80/20
            indices = np.random.permutation(len(X))
            split = int(0.8 * len(X))
            train_idx, val_idx = indices[:split], indices[split:]
            
            # 3. Create Tensors (Scaling input by 5.0, keeping output raw Volts)
            tx_train = torch.Tensor(X[train_idx]) / 5.0
            ty_train = torch.Tensor(y[train_idx])
            
            tx_val = torch.Tensor(X[val_idx]) / 5.0
            ty_val = torch.Tensor(y[val_idx])
            
            print(f"Training on {len(train_idx)} samples, Validating on {len(val_idx)} samples.")
            
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            
            # --- TRAINING LOOP ---
            for epoch in range(500):
                # A. Training Step
                self.model.train()
                optimizer.zero_grad()
                
                train_pred = self.model(tx_train)
                train_loss = loss_fn(train_pred, ty_train)
                
                train_loss.backward()
                optimizer.step()
                
                # B. Validation Step (Every 100 epochs)
                if epoch % 50 == 0:
                    self.model.eval() # Set to evaluation mode
                    with torch.no_grad(): # Disable gradient calculation for speed/memory
                        val_pred = self.model(tx_val)
                        val_loss = loss_fn(val_pred, ty_val)
                    
                    print(f"Epoch {epoch:4d} | Train Loss: {train_loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

            # 4. Save
            torch.save(self.model.state_dict(), path)
            print(f"Multi-output model saved to {path}")

    def cleanup(self):
        print("Closing connection...")
        try: self.scopes.close()
        except: pass

if __name__ == "__main__":
    trainer = None
    try:
        print('Starting Decoder Training...')
        trainer = ChipTrainer()
        trainer.train_and_save("decoder_model.pth")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if trainer:
            trainer.cleanup()
        print("Clean exit.")