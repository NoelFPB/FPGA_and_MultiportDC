import torch
import torch.optim as optim
import numpy as np
import time
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes
# Ensure these files exist and have the classes we discussed
from PAT.train_model import ChipModel 
from decoder_logic import generate_full_truth_table

class PATDecoderOptimizer:
    def __init__(self, model_path="chip_model_6ch.pth"):
        # 5 Input Pins: OP0, OP1, OP2, OP3, CFLAG
        self.INPUT_HEATERS = [27, 28, 29, 30, 31] 
        # The other 35 heaters are trainable weights
        self.opt_indices = [h for h in range(40) if h not in self.INPUT_HEATERS]
        
        # Load the 6-Output Digital Model
        print(f"Loading Digital Twin from {model_path}...")
        self.digital_model = ChipModel(num_outputs=6)
        try:
            self.digital_model.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print(f"ERROR: {model_path} not found. Did you retrain the model for 6 outputs?")
            exit()
        self.digital_model.eval()
        
        # Hardware Setup
        self.controller = DualAD5380Controller()
        # Reading 6 channels: Scope1 (1-4) and Scope2 (1-2)
        # Adjust IDs if your scope setup is different
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2])
        
        # Trainable Parameters (35 Heaters)
        self.heater_params = torch.nn.Parameter(
            torch.rand(35) * (4.1 - 0.5) + 0.5
        )
        
        # Optimizer
        self.optimizer = optim.Adam([self.heater_params], lr=0.01)

    def calibrate_6_channels(self):
        """
        Wiggle heaters to find the physical Min/Max voltage for ALL 6 output channels separately.
        """
        print("Calibrating 6 output channels...", end="")
        vals = np.zeros((20, 6)) # 20 probes, 6 channels
        
        for i in range(20):
            # Random config
            for h in range(40):
                self.controller.set(h, np.random.uniform(0.5, 4.1))
            time.sleep(0.01)
            
            read = self.scopes.read_many(avg=1) 
            if read and len(read) >= 6:
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

    def optimize(self, steps=200):
        # 1. Calibrate Targets
        self.calibrate_6_channels()
        
        # 2. Generate Truth Table (32 inputs x 6 outputs)
        logic_inputs_np, target_outputs_np = generate_full_truth_table(self.L_targets, self.H_targets)
        
        target_tensor = torch.Tensor(target_outputs_np)
        
        print(f"Starting Decoder Optimization ({steps} steps)...")
        
        for step in range(steps):
            self.optimizer.zero_grad()
            
            # --- STOCHASTIC BATCH SELECTION ---
            # We pick 8 random instructions (out of 32) to verify physically this step
            batch_size = 8
            batch_indices = np.random.choice(32, batch_size, replace=False)
            
            # --- PHASE 1: PHYSICAL FORWARD PASS ---
            phys_outputs = []
            
            # Detach params for hardware application (numpy only)
            current_heaters = self.heater_params.detach().numpy()
            
            for idx in batch_indices:
                logic_bits = logic_inputs_np[idx] 
                
                # Construct full 40-heater config
                full_config = np.zeros(40)
                
                # A. Fill Trainable Heaters
                for k, h_idx in enumerate(self.opt_indices):
                    full_config[h_idx] = current_heaters[k]
                    
                # B. Fill Logic Input Heaters (0.5V or 4.1V)
                for k, h_inp in enumerate(self.INPUT_HEATERS):
                    full_config[h_inp] = 0.5 if logic_bits[k] == 0 else 4.1
                
                # Apply to Hardware
                for h, v in enumerate(full_config): 
                    self.controller.set(h, v)
                
                # Measure (Expect 6 values)
                read = self.scopes.read_many(avg=1)
                if read is None or len(read) < 6:
                    # Fallback for read error: use 0s (or previous valid) to prevent crash
                    phys_outputs.append([0.0]*6) 
                else:
                    phys_outputs.append(read[0:6]) 
            
            # Convert physical measurements to Tensor
            y_phys = torch.Tensor(phys_outputs) # Shape (8, 6)
            y_target = target_tensor[batch_indices] # Shape (8, 6)
            
            # --- PHASE 2: DIGITAL BACKWARD PASS ---
            
            # 1. Calculate Real Error Gradient
            error = y_phys - y_target
            
            # 2. Re-construct Input Tensor for Digital Model
            # We must build this using Torch Tensors to keep the gradient graph alive
            batch_input_tensor = torch.zeros(batch_size, 40)
            
            # Fill trainable columns (connects to self.heater_params)
            for k, h_idx in enumerate(self.opt_indices):
                # We assign the parameter to the entire column k
                batch_input_tensor[:, h_idx] = self.heater_params[k]
                
            # Fill logic input columns (Fixed values, no gradient needed)
            current_batch_logic = logic_inputs_np[batch_indices]
            for i in range(batch_size):
                bits = current_batch_logic[i]
                for k, h_inp in enumerate(self.INPUT_HEATERS):
                    val = 0.5 if bits[k] == 0 else 4.1
                    batch_input_tensor[i, h_inp] = val
            
            # 3. Digital Forward Pass
            y_model = self.digital_model(batch_input_tensor)
            
            # 4. Inject Physical Error
            # We tell PyTorch: "The output was y_model, but the gradient at the output is 'error'"
            y_model.backward(gradient=error)
            
            # --- PHASE 3: UPDATE ---
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_([self.heater_params], max_norm=1.0)
            
            self.optimizer.step()
            
            # Clamp heaters to safe range
            with torch.no_grad():
                self.heater_params.clamp_(0.5, 4.1)
            
            # Logging
            if step % 5 == 0:
                loss = torch.mean(error**2).item()
                print(f"Step {step}: MSE Loss = {loss:.4f}")

        print("Optimization Complete.")
        self.cleanup()

    def cleanup(self):
        try: self.scopes.close()
        except: pass

if __name__ == "__main__":
    # Ensure chip_model_6ch.pth exists first!
    opt = PATDecoderOptimizer(model_path="chip_model_6ch.pth")
    try:
        opt.optimize(steps=200)
    except KeyboardInterrupt:
        print("\nStopping early...")
        opt.cleanup()