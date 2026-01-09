import numpy as np
import time
import pyvisa
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes

# --- CONFIGURATION ---
NUM_SAMPLES = 5000
NUM_HEATERS = 49
SCOPE_INDICES = [0, 1, 2, 3, 4, 5] 
INPUT_HEATER_INDICES = [42, 43, 44, 45, 46]

def collect():
    print("Initializing Hardware...")
    controller = DualAD5380Controller()
    scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2])
    
    X_data = []
    y_data = []
 
    print(f"Starting collection of {NUM_SAMPLES} samples...")
    try:
        for i in range(1, NUM_SAMPLES + 1): # Start at 1 for cleaner modulo math
            # 1. Generate Configuration
            config = np.random.uniform(0.1, 4.9, NUM_HEATERS)
            if i >= NUM_SAMPLES // 2:
                for h_inp in INPUT_HEATER_INDICES:
                    config[h_inp] = 0.5 if np.random.random() > 0.5 else 4.1
      
            # 2. Apply to Hardware
            for h, v in enumerate(config):
                controller.set(h, v)
            
            # 3. Wait for Thermal Equilibrium
            time.sleep(0.2) 
            
            vals = scopes.read_many(avg=1)
            
            if vals is not None and len(vals) > max(SCOPE_INDICES):
                current_outputs = [vals[idx] for idx in SCOPE_INDICES]
                if all(abs(v) < 12.0 for v in current_outputs):
                    X_data.append(config)
                    y_data.append(current_outputs)

            # 5. LIVE CORRELATION MONITOR (Every 100 samples)
            if i % 100 == 0 and len(X_data) >= 100:
                # We check correlation between sum of heater Volts and Scope Channel 0
                x_recent = np.sum(np.array(X_data[-100:]), axis=1)
                y_recent = np.array(y_data[-100:])[:, 0]
                
                corr = np.corrcoef(x_recent, y_recent)[0, 1]
                
                print(f"Progress: {i}/{NUM_SAMPLES} | Recent Corr: {corr:.4f}")
                
                if abs(corr) < 0.01:
                    print("⚠️  WARNING: Correlation is near zero. Data might be desynced!")

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        if len(X_data) > 0:
            np.savez("chip_dataset.npz", X=np.array(X_data), y=np.array(y_data))
            print(f"Saved {len(X_data)} samples to chip_dataset.npz")
        
        scopes.close()
        print("Hardware connections closed.")

if __name__ == "__main__":
    collect()