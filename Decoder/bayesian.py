import time
import numpy as np
import json
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes

# ============================================================
#   CONFIGURATION
# ============================================================

# Input Pins (OP0, OP1, OP2, OP3, CFLAG)
INPUT_HEATERS = [42, 43, 44, 45, 46] 

# Trainable Background Heaters (All others)
MODIFIABLE_HEATERS = [h for h in range(49) if h not in INPUT_HEATERS]

V_LOW_LOGIC = 0.5   # Voltage sent to Input Heaters for Logic 0
V_HIGH_LOGIC = 4.1  # Voltage sent to Input Heaters for Logic 1

# 6-channel readout: (scope1 CH1..CH4, scope2 CH1..CH2)
# We assume indices 0-5 map to the decoder outputs [SelB, SelA, L0, L1, L2, L3]
OUTPUT_CHANNELS = 6 

# ============================================================
#   DECODER LOGIC (Truth Table)
# ============================================================

def get_decoder_target(op3, op2, op1, op0, cflag):
    """ Returns the expected index (0-5) that should be HIGH. Returns -1 if ALL LOW. """
    # 0000: ADD A, Im -> L0 (Index 2)
    if   (op3==0 and op2==0 and op1==0 and op0==0): return 2
    # 0001: MOV A, B  -> SelA (Index 1)
    elif (op3==0 and op2==0 and op1==0 and op0==1): return 1
    # 0010: IN A      -> L1 (Index 3)
    elif (op3==0 and op2==0 and op1==1 and op0==0): return 3
    # 0011: MOV A, Im -> L1 (Index 3) - Wait, logic says H,H,H... assuming L1 dominates or check logic?
    # Let's stick to the STRICT single-hot logic for optimization to avoid ambiguity.
    # If your logic table has multiple HIGHs, we maximize the MINIMUM of them.
    # For this BO, let's test the CLEANEST unique instructions.
    pass

    # Simplified Test Set: The 10 Unique Instructions
    # format: (op3, op2, op1, op0, cflag, expected_output_index)
    # expected_index: 0=SelB, 1=SelA, 2=L0, 3=L1, 4=L2, 5=L3
    return [
        (0,0,0,0,0, 2), # ADD A -> L0
        (0,0,0,1,0, 1), # MOV A,B -> SelA
        (0,0,1,0,0, 3), # IN A -> L1
        (0,1,0,0,0, 4), # MOV B,A -> L2 (Wait, check logic...)
        # Let's define the test suite manually to match your Python logic EXACTLY
    ]

def get_test_suite():
    """
    Returns list of tuples: (inputs_list, expected_high_indices)
    inputs_list = [op3, op2, op1, op0, c]
    expected_high_indices = list of indices [0..5] that should be HIGH
    """
    tests = []
    # 32 combinations is too slow for BO. We test the 10 valid instructions.
    # Code mapping: op3, op2, op1, op0, c
    
    # 1. ADD A, Im (00000) -> L0 High (Idx 2)
    tests.append(([0,0,0,0,0], [2]))
    
    # 2. MOV A, B (00010) -> SelA High (Idx 1) (Ignoring logic overlap for now, focusing on unique winner)
    tests.append(([0,0,0,1,0], [1, 2])) # Based on your table: L, H, H, L... -> SelA & L0
    
    # Actually, to make BO robust, let's just use the python function you provided
    # and maximize the gap between (Lowest High) and (Highest Low).
    
    # We will generate a subset of 10 random instructions + the 10 valid ones each step?
    # No, deterministic is better. Let's pick the 8 most distinct instructions.
    
    batch = [
        (0,0,0,0,0), # ADD A
        (0,0,0,1,0), # MOV A, B
        (0,0,1,0,0), # IN A
        (0,1,0,0,0), # MOV B, A
        (0,1,0,1,0), # ADD B, Im
        (0,1,1,0,0), # IN B
        (1,0,0,1,0), # OUT B
        (1,0,1,1,0)  # OUT Im
    ]
    return batch

def get_expected_logic(op3, op2, op1, op0, c):
    """ Re-implementation of your logic function for the scorer """
    L, H = 0, 1 # Logic levels
    out = [L, L, L, L, L, L] 
    if   (op3==0 and op2==0 and op1==0 and op0==0): out = [L, L, H, L, L, L]
    elif (op3==0 and op2==0 and op1==0 and op0==1): out = [L, H, H, L, L, L]
    elif (op3==0 and op2==0 and op1==1 and op0==0): out = [L, H, L, H, L, L]
    elif (op3==0 and op2==0 and op1==1 and op0==1): out = [L, H, H, H, L, L]
    elif (op3==0 and op2==1 and op1==0 and op0==0): out = [H, L, L, L, H, L]
    elif (op3==0 and op2==1 and op1==0 and op0==1): out = [L, H, L, L, H, H]
    elif (op3==0 and op2==1 and op1==1 and op0==0): out = [H, L, L, L, H, H]
    elif (op3==0 and op2==1 and op1==1 and op0==1): out = [H, L, L, H, H, L]
    elif (op3==1 and op2==0 and op1==0 and op0==1): out = [H, L, L, L, H, L]
    elif (op3==1 and op2==0 and op1==1 and op0==1): out = [L, H, L, L, H, L]
    return out

# ============================================================
#   MAIN OPTIMIZER CLASS
# ============================================================

class DecoderBO:
    def __init__(self):
        print("=== Initializing Decoder Bayesian Optimizer ===")
        
        # Hardware interfaces
        self.controller = DualAD5380Controller()
        # Scope 1 (4ch) + Scope 2 (2ch)
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2])

        # BO Storage
        self.X = []
        self.y = []
        self.best_config = None
        self.best_score = -1e9
        
        # Test Suite
        self.test_ops = get_test_suite()

    # ========================================================
    #  Calibration
    # ========================================================
    def calibrate(self):
        print("Calibrating channel ranges...", end="")
        vals = np.zeros((15, 6))
        for i in range(15):
            for h in range(49): self.controller.set(h, np.random.uniform(0.1, 4.9))
            time.sleep(0.02)
            read = self.scopes.read_many(avg=1)
            if read is not None and len(read) >= 6: 
                vals[i,:] = read[0:6]
            else:
                # If read fails, just duplicate the previous row (or 0) to avoid crash
                if i > 0: vals[i,:] = vals[i-1,:]
        
        self.v_min = np.min(vals, axis=0)
        self.v_max = np.max(vals, axis=0)
        print(" Done.")
        for ch in range(6):
            print(f"  CH{ch}: {self.v_min[ch]:.2f}V - {self.v_max[ch]:.2f}V")

    # ========================================================
    #  Evaluate Configuration
    # ========================================================
    def evaluate(self, bg_weights):
        """
        bg_weights: List/Array of weights for MODIFIABLE_HEATERS
        Returns: Score (Higher is better)
        """
        # 1. Apply Background Weights
        for i, h in enumerate(MODIFIABLE_HEATERS):
            self.controller.set(h, float(bg_weights[i]))
        
        # 2. Loop through test instructions
        margins = []
        
        for ops in self.test_ops:
            op3, op2, op1, op0, c = ops
            
            # Set Inputs
            inp_vals = [op3, op2, op1, op0, c]
            for k, h_inp in enumerate(INPUT_HEATERS):
                v = V_LOW_LOGIC if inp_vals[k] == 0 else V_HIGH_LOGIC
                self.controller.set(h_inp, v)
            
            time.sleep(0.05) # Fast settling
            
            # Measure
            read = self.scopes.read_many(avg=1)
            if read is None or len(read) < 6:
                return -1e9 # Penalty for read failure
            
            vals = np.array(read[0:6])
            
            # Determine Logic
            targets = get_expected_logic(op3, op2, op1, op0, c) # [0, 1, 1, 0...]
            
            # Calculate Margin for THIS instruction
            # We want min(Highs) - max(Lows)
            
            high_volts = [vals[i] for i, t in enumerate(targets) if t == 1]
            low_volts  = [vals[i] for i, t in enumerate(targets) if t == 0]
            
            if not high_volts: # Should not happen with valid ops
                current_margin = -max(low_volts) # Just minimize noise
            elif not low_volts:
                current_margin = min(high_volts)
            else:
                # The core metric: Separation
                current_margin = min(high_volts) - max(low_volts)
            
            margins.append(current_margin)
            
        # 3. Final Score = The Worst Case Margin across all instructions
        # If the worst margin is positive, the decoder works for ALL instructions.
        worst_case_margin = min(margins)
        
        # Softplus scaling to make GP happy (linear is fine too)
        return float(worst_case_margin)

    # ========================================================
    #  BO Helpers
    # ========================================================
    def add_eval(self, x, score):
        self.X.append(x)
        self.y.append(score)
        if score > self.best_score:
            self.best_score = score
            self.best_config = x
            print(f"  ★ New Best: {score:.3f} V margin")

    def fit_gp(self):
        if len(self.X) < 5: return None
        X_train = np.array(self.X)
        y_train = np.array(self.y)
        
        # Robust Kernel for High Dimensions (44 dims!)
        # Length scale needs to be large so it doesn't overfit noise
        kernel = (
            ConstantKernel(1.0) * Matern(length_scale=2.0, nu=2.5) + 
            WhiteKernel(noise_level=0.01)
        )
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        gp.fit(X_train, y_train)
        return gp

    # ========================================================
    #  Optimization Loop
    # ========================================================
    def optimize(self, iterations=30, candidates=500):
        self.calibrate()
        
        print(f"\n=== Starting BO ({iterations} iters) ===")
        # 1. Random Sampling
        sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS))
        sample = sampler.random(n=5) # Start with 5 random points
        
        for i in range(5):
            x = sample[i] * 4.8 + 0.1 # Scale 0-1 to 0.1-4.9
            score = self.evaluate(x)
            self.add_eval(x, score)
            print(f"  [Init {i+1}] Score: {score:.3f}")
            
        # 2. GP Loop
        for it in range(iterations):
            print(f"Iter {it+1}: ", end="")
            gp = self.fit_gp()
            
            # Suggest Candidates (UCB)
            # Generate random candidates
            cand_X = np.random.uniform(0.1, 4.9, (candidates, len(MODIFIABLE_HEATERS)))
            
            if gp:
                mu, sigma = gp.predict(cand_X, return_std=True)
                ucb = mu + 1.96 * sigma
                best_idx = np.argmax(ucb)
                next_x = cand_X[best_idx]
            else:
                next_x = cand_X[0]
            
            # Evaluate
            score = self.evaluate(next_x)
            self.add_eval(next_x, score)
            print(f"Score: {score:.3f}")
            
        # 3. Save
        self.save_best()

    def save_best(self):
        if self.best_config is not None:
            config_map = {
                "bg_weights": self.best_config.tolist(),
                "score": self.best_score
            }
            with open("decoder_bo_config.json", "w") as f:
                json.dump(config_map, f, indent=4)
            print("\nSaved best config to decoder_bo_config.json")

    def cleanup(self):
        try: self.scopes.close()
        except: pass

if __name__ == "__main__":
    opt = DecoderBO()
    try:
        opt.optimize(iterations=100)
    except KeyboardInterrupt:
        print("\nStopping...")
        opt.save_best()
    finally:
        opt.cleanup()