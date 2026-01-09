import torch
import torch.optim as optim
import numpy as np
import json
import time
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes
from PAT.train_decoder import ChipModel 
from PAT.decoder_logic import generate_full_truth_table

class PATDecoderOptimizer:
    def __init__(self, model_path="decoder_model.pth"):
        self.INPUT_HEATERS = [42, 43, 44, 45, 46]         # 5 Input Pins: OP0, OP1, OP2, OP3, CFLAG
        self.opt_indices = list(range(42))  # Heaters 0-41 are trainable
        
        # Load the Digital Model
        print(f"Loading Digital Twin from {model_path}...")
        self.digital_model = ChipModel(num_outputs=6)
        self.digital_model.load_state_dict(torch.load(model_path))
        self.digital_model.eval()
        # Hardware Setup
        self.controller = DualAD5380Controller()
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2])
        self.heater_params = torch.nn.Parameter(torch.rand(42) * 0.1) # Trainable Parameters
        self.optimizer = optim.Adam([self.heater_params], lr=0.1)

    def calibrate_6_channels(self):
        print("Calibrating 6 output channels...", end="")
        vals = np.zeros((20, 6)) # 20 probes, 6 channels
        
        for i in range(15):
            # Random config
            for h in range(49):
                self.controller.set(h, np.random.uniform(0.1, 4.9))
            time.sleep(0.2)
            read = self.scopes.read_many(avg=1) 
            
            if read is not None and len(read) >= 6:
                vals[i, :] = read[0:6]
            
        # Determine targets per channel
        self.L_targets = []
        self.H_targets = []
        
        print("\nCalibration Results:")
        for ch in range(6):
            v_min = np.min(vals[:, ch])
            v_max = np.max(vals[:, ch])
            
            # Renormalization: Logic 0 = 10%, Logic 1 = 90%
            L = v_min + 0.10 * (v_max - v_min)
            H = v_max - 0.10 * (v_max - v_min)
            
            self.L_targets.append(L)
            self.H_targets.append(H)
            print(f"  CH{ch}: Range [{v_min:.2f}V, {v_max:.2f}V] -> Low:{L:.2f}V, High:{H:.2f}V")

        # 1. Find the global range that ALL channels can satisfy
        global_v_min = np.max(np.min(vals, axis=0)) # The highest "Low" across all channels
        global_v_max = np.min(np.max(vals, axis=0)) # The lowest "High" across all channels

        # 2. Define unified targets (e.g., 10% and 90% of the shared range)
        unified_L = global_v_min + 0.10 * (global_v_max - global_v_min)
        unified_H = global_v_max - 0.10 * (global_v_max - global_v_min)
        self.L_targets = [unified_L] * 6
        self.H_targets = [unified_H] * 6
        print(f"\nUnified Calibration Results:")
        print(f" Global Low Target: {unified_L:.2f}V | Global High Target: {unified_H:.2f}V")
        # Define the hardware decision threshold (midpoint)
        self.decision_threshold = (unified_L + unified_H) / 2.0
        print(f" Decision Threshold: {self.decision_threshold:.2f}V")
        
    def optimize(self, steps=200):
        # 1. Calibrate Targets
        self.calibrate_6_channels()
        
        # 2. Generate Truth Table
        logic_inputs_np, target_outputs_np = generate_full_truth_table(self.L_targets, self.H_targets)
        target_tensor = torch.Tensor(target_outputs_np)
        
        print(f"Starting Decoder Optimization ({steps} steps)...")    
        best_loss = float('inf')
        best_config = None
        best_acc = 0.0
        for step in range(steps):
            self.optimizer.zero_grad()
            # --- STOCHASTIC BATCH SELECTION ---
            batch_size = 8
            batch_indices = np.random.choice(32, batch_size, replace=False)
            
            # --- PHASE 1: PHYSICAL FORWARD PASS ---
            phys_outputs = []
            current_heaters = self.heater_params.detach().numpy()
            
            for idx in batch_indices:
                logic_bits = logic_inputs_np[idx] 
                full_config = np.ones(49) * 0.1 # Initialize everything to a low baseline
                # 1. Fill Trainable Heaters (Layers 1-6)
                full_config[self.opt_indices] = current_heaters
                # 2. Fill Input Heaters (Layer 7)
                for k in range(5):
                    h_inp = self.INPUT_HEATERS[k]
                    full_config[h_inp] = 0.5 if logic_bits[k] == 0 else 4.1

                # Hardware I/O
                for h, v in enumerate(full_config): self.controller.set(h, v)
                time.sleep(0.25) # Slightly faster sleep for speed
                
                read = self.scopes.read_many(avg=1)
                if read is None or len(read) < 6:
                    phys_outputs.append([0.0]*6) 
                else:
                    phys_outputs.append(read[0:6]) 
            
            y_phys = torch.from_numpy(np.array(phys_outputs, dtype=np.float32))
            y_target = target_tensor[batch_indices]
            
            # --- PHASE 2: DIGITAL BACKWARD PASS (MARGIN LOSS) ---
            # A. Re-construct Input Tensor
            batch_input_tensor = torch.zeros(batch_size, 49)
             # 1. Map all 42 trainable params to the first 42 heaters in one shot
            batch_input_tensor[:, self.opt_indices] = self.heater_params
            
            current_batch_logic = logic_inputs_np[batch_indices]
            for i in range(batch_size):
                bits = current_batch_logic[i] # [OP0, OP1, OP2, OP3, CFLAG]
                for k in range(5):
                    h_idx = self.INPUT_HEATERS[k] # 42, 43, 44, 45, 46
                    val = 0.5 if bits[k] == 0 else 4.1
                    batch_input_tensor[i, h_idx] = val

            # --- PHASE 2: B. Calculate Gradient (Relative + Absolute) ---
            target_indices = y_target.argmax(dim=1) 
            vals_correct = y_phys[range(batch_size), target_indices] 

            # 1. Identify Margin Violations (Is the winner "winning" by enough?)
            y_masked = y_phys.clone()
            y_masked[range(batch_size), target_indices] = -float('inf') 
            vals_max_wrong, runner_up_indices = y_masked.max(dim=1) 

            margin = 2.0  
            margin_violations = (vals_correct - vals_max_wrong) < margin
            # Standardizing variable names to match your "violations" line
            high_violations = vals_correct < self.decision_threshold
            low_violations_mask = y_masked > self.decision_threshold
            low_violations = low_violations_mask.any(dim=1)

            # Re-calculate scalar loss for logging
            losses = torch.relu(margin - (vals_correct - vals_max_wrong))
            
            # Combined violations boolean mask
            violations = margin_violations | high_violations | low_violations
       
            # --- PHASE 2: C. Inject Combined Gradient ---
            grad_vector = torch.zeros_like(y_phys)

            for i in range(batch_size):
                if margin_violations[i]:
                    grad_vector[i, target_indices[i]] -= 1.0 
                    grad_vector[i, runner_up_indices[i]] += 1.0 
                
                if high_violations[i]:
                    grad_vector[i, target_indices[i]] -= 0.5 
                
                # Use the mask we created to push down any high "Low" channels
                for ch in range(6):
                    if ch != target_indices[i] and y_phys[i, ch] > self.decision_threshold:
                        grad_vector[i, ch] += 0.5 

            y_model = self.digital_model(batch_input_tensor/5)
            y_model.backward(gradient=grad_vector * 1.0)
            
            # --- PHASE 3: UPDATE ---
            torch.nn.utils.clip_grad_norm_([self.heater_params], max_norm=1.0)
            self.optimizer.step()
            
            with torch.no_grad():
                self.heater_params.clamp_(0.1, 4.9)
            
            # --- LOGGING & SAVING ---
            margin_loss = torch.mean(losses).item()
            # Accuracy is: batch size minus any row that had a violation
            acc = 1.0 - (violations.float().sum().item() / batch_size)
            
            # SAVE LOGIC: Accuracy first, then Loss
            if (acc > best_acc) or (acc == best_acc and margin_loss < best_loss):
                best_acc = acc
                best_loss = margin_loss
                best_config = {
                    "bg_weights": self.heater_params.detach().cpu().numpy().tolist(),
                    "loss": best_loss,
                    "accuracy": best_acc
                }
            if step % 1 == 0:
                print(f"Step {step}: Loss = {margin_loss:.4f} | Acc = {acc*100:.0f}%")

        print("\nOptimization Complete.")
        if best_config:
            filename = "decoder_config.json"
            with open(filename, "w") as f:
                json.dump(best_config, f, indent=4)
            print(f"Best configuration (Loss: {best_loss:.4f}) saved to {filename}")
        
        self.cleanup()

    def cleanup(self):
        try: self.scopes.close()
        except: pass

if __name__ == "__main__":
    opt = PATDecoderOptimizer(model_path="decoder_model.pth")
    try:
        opt.optimize(steps=60)
    except KeyboardInterrupt:
        print("\nStopping early...")
        opt.cleanup()