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
    if   (op3==0 and op2==0 and op1==0 and op0==0): out = [0, 0, 1, 0, 0, 0] # ADD A, Im
    elif (op3==0 and op2==0 and op1==0 and op0==1): out = [0, 1, 1, 0, 0, 0] # MOV A, B
    elif (op3==0 and op2==0 and op1==1 and op0==0): out = [0, 1, 0, 1, 0, 0] # IN A
    elif (op3==0 and op2==0 and op1==1 and op0==1): out = [0, 1, 1, 1, 0, 0] # MOV A, Im
    elif (op3==0 and op2==1 and op1==0 and op0==0): out = [1, 0, 0, 0, 1, 0] # MOV B, A
    elif (op3==0 and op2==1 and op1==0 and op0==1): out = [0, 1, 0, 0, 1, 1] # ADD B, Im
    elif (op3==0 and op2==1 and op1==1 and op0==0): out = [1, 0, 0, 0, 1, 1] # IN B
    elif (op3==0 and op2==1 and op1==1 and op0==1): out = [1, 0, 0, 1, 1, 0] # MOV B, Im
    elif (op3==1 and op2==0 and op1==0 and op0==1): out = [1, 0, 0, 0, 1, 0] # OUT B
    elif (op3==1 and op2==0 and op1==1 and op0==1): out = [0, 1, 0, 0, 1, 0] # OUT Im
    # JNC and JMP default to all 0s in your logic
    
    return out

class DecoderTester:
    def __init__(self, config_file="decoder_config.json"):
        # Hardware
        self.controller = DualAD5380Controller()
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2])
        
        # Pin Definitions
        self.INPUT_HEATERS = [42, 43, 44, 45, 46]
        self.opt_indices = [h for h in range(49) if h not in self.INPUT_HEATERS]
        
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
        
        # Test specific instructions
        test_cases = [
            (0,0,0,0,0, "ADD A, Im"),
            (0,0,0,1,0, "MOV A, B"),
            (0,0,1,0,0, "IN A"),
            (0,0,1,1,0, "MOV A, Im"),
            (0,1,0,0,0, "MOV B, A"),
            (0,1,0,1,0, "ADD B, Im"),
            (0,1,1,0,0, "IN B"),
            (0,1,1,1,0, "MOV B, Im"),
            (1,0,0,1,0, "OUT B"),
            (1,0,1,1,0, "OUT Im")
        ]
        
        print(f"\n{'INSTRUCTION':<12} | {'Expected':<14} | {'Measured':<14} | {'VOLTAGES':<35} | STATUS")
        print("-" * 90)
        
        passes = 0
        
        for op3, op2, op1, op0, c, name in test_cases:
            # 1. Set Inputs
            bits = [op3, op2, op1, op0, c]
            for k, h_inp in enumerate(self.INPUT_HEATERS):
                val = 0.5 if bits[k] == 0 else 4.1
                self.controller.set(h_inp, val)
            
            time.sleep(0.1)
            
            # 2. Measure
            read = self.scopes.read_many(avg=4)
            if read is None or len(read) < 6:
                print("Scope Read Error")
                continue
                
            vals = read[0:6]
            
            # 3. Digitalize (Thresholding)
            # You might need to verify which channel is which!
            # Assuming: Scope1_CH1=SelB, S1_CH2=SelA, S1_CH3=L0...
            threshold = 3.0 # Anything above 1.5V is a '1'
            measured_logic = [1 if v > threshold else 0 for v in vals]
            
            # 4. Compare
            expected_logic = get_expected_logic(op3, op2, op1, op0, c)
            
            is_match = (measured_logic == expected_logic)
            status = "PASS" if is_match else "FAIL"
            if is_match: passes += 1
            
            # 5. Print Row
            exp_str = "".join(str(x) for x in expected_logic)
            meas_str = "".join(str(x) for x in measured_logic)
            volt_str = " ".join([f"{v:.1f}" for v in vals])
            
            # Highlight fail in red (if your terminal supports it) or just text
            print(f"{name:<12} | {exp_str:<14} | {meas_str:<14} | {volt_str:<35} | {status}")

        print("-" * 90)
        print(f"Final Result: {passes}/{len(test_cases)} Passed")
        
    def cleanup(self):
        try: self.scopes.close()
        except: pass

if __name__ == "__main__":
    tester = DecoderTester()
    try:
        tester.run_test()
    finally:
        tester.cleanup()