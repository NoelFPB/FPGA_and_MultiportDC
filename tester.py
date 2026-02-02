import torch
import numpy as np
import time
import json
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes

# --- 1. DEFINE THE TARGET LOGIC (Your Code) ---
def get_expected_logic(op3, op2, op1, op0, cflag):
    """ Returns binary list [SelB, SelA, L0, L1, L2, L3] """
    # Default: ALL LOW
    out = [0, 0, 0, 0, 0, 0] 
    
    # Logic from your provided snippet
    if   (op3==1 and op2==0 and op1==1 and op0==1): out = [1, 1, 1, 1, 0, 1] # Out IM
    elif (op3==1 and op2==1 and op1==1 and op0==1): out = [1, 1, 1, 1, 1, 0] # JMP
    # JNC and JMP default to all 0s in your logic
    
    return out

class DecoderTester:
    def __init__(self, config_file="decoder_hybrid_best.json"):
        # Hardware
        self.controller = DualAD5380Controller()
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2])
        time.sleep(1) # Allow scopes to initialize
        # Pin Definitions
        self.INPUT_HEATERS =[15, 16, 17, 18, 19] 
        self.opt_indices = [h for h in range(14) if h not in self.INPUT_HEATERS]
        
        # Load Config
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded configuration from {config_file}")
            print(f"Validation Loss was: {self.config.get('loss', 'N/A')}")
        except FileNotFoundError:
            print("ERROR: Config file not found. Run optimization first!")
            exit()

    def set_weights(self):
        """Apply the trained background weights"""
        weights = self.config["bg_weights"] # Ensure your optimizer saves this list!
        for i, h_idx in enumerate(self.opt_indices):
            self.controller.set(h_idx, weights[i])
        print("Background weights applied.")

    
    def run_test(self):
        self.set_weights()
        
        # 1. TEST ONLY THE OPTIMIZED CURRICULUM
        test_cases = [
            (1, 0, 1, 1, 1, "Out IM"),
            (1, 1, 1, 1, 1, "JMP")
        ]
        
        print(f"\n{'INSTRUCTION':<12} | {'Expected':<8} | {'Measured':<8} | {'VOLTAGES (V0-V5)':<32} | {'CONTRAST':<8} | STATUS")
        print("-" * 105)
        
        passes = 0
        threshold = 2.0  # Logic threshold for 1 vs 0
        
        for op3, op2, op1, op0, c, name in test_cases:
            # Set Inputs
            bits = [op3, op2, op1, op0, c]
            for k, h_inp in enumerate(self.INPUT_HEATERS):
                val = 0.1 if bits[k] == 0 else 3.3
                self.controller.set(h_inp, val)
            
            time.sleep(0.5) # Allow thermal settling
            
            # Measure (Average higher for the final report)
            read = self.scopes.read_many(avg=3)
            if read is None: continue
            vals = np.array(read[0:6])
            
            # Logic Analysis
            expected_logic = np.array(get_expected_logic(op3, op2, op1, op0, c))
            measured_logic = [1 if v > threshold else 0 for v in vals]
            
            # Calculate Contrast (Smallest High / Largest Low)
            highs = vals[expected_logic == 1]
            lows = vals[expected_logic == 0]
            contrast = np.min(highs) / (np.max(lows) + 1e-3) if len(lows) > 0 else 9.9
            
            # Formatting strings
            exp_str = "".join(str(x) for x in expected_logic)
            meas_str = "".join(str(x) for x in measured_logic)
            volt_str = " ".join([f"{v:4.1f}" for v in vals])
            
            is_match = (measured_logic == expected_logic.tolist())
            status = "PASS" if is_match else "FAIL"
            if is_match: passes += 1
            
            print(f"{name:<12} | {exp_str:<8} | {meas_str:<8} | {volt_str:<32} | {contrast:>7.2f}:1 | {status}")

        print("-" * 105)
        print(f"Final Validation: {passes}/{len(test_cases)} Passed")
    
    def cleanup(self):
        try: self.scopes.close()
        except: pass

if __name__ == "__main__":
    tester = DecoderTester()
    try:
        tester.run_test()
    finally:
        tester.cleanup()