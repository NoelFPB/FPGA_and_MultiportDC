import time
import numpy as np
import json
import warnings
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.exceptions import ConvergenceWarning
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes

# Silencing the length-scale warnings to keep the console clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)
USE = 16
# ============================================================
#   CONFIGURATION
# ============================================================
INPUT_HEATERS = [15, 16, 17, 18, 19] 
MODIFIABLE_HEATERS = [h for h in range(21) if h not in INPUT_HEATERS]

V_LOW_LOGIC = 0.1   
V_HIGH_LOGIC = 4.1  
OUTPUT_CHANNELS = 6 

# ============================================================
#   DECODER LOGIC (8 Instructions)
# ============================================================
def get_test_suite():    
    return [(0,0,0,0,0), (0,0,0,1,0), (0,0,1,0,0), (0,1,0,0,0), 
            (0,1,0,1,0), (0,1,1,0,0), (1,0,0,1,0), (1,0,1,1,0)]

def get_expected_logic(op3, op2, op1, op0, c):
    L, H = 0, 1
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

class HybridDecoderBO:
    def __init__(self):
        print("=== Initializing Hybrid 44-Heater Optimizer ===")
        self.controller = DualAD5380Controller()
        self.scopes = RigolDualScopes(channels_scope1=[1,2,3,4], channels_scope2=[1,2])
        self.test_ops = get_test_suite()

        self.X, self.y = [], []
        self.best_config, self.best_score = None, -1e9

    def evaluate(self, bg_weights):
        # 1. Apply Background Weights
        for i, h in enumerate(MODIFIABLE_HEATERS):
            self.controller.set(h, float(bg_weights[i]))
        time.sleep(0.1) 
        
        # --- DEFINE TARGET LOGIC LEVELS ---
        V_OH_TARGET = 3.0  # Target for Logic 1 (Minimum)
        V_OL_TARGET = 2.0  # Target for Logic 0 (Maximum)
        GAP_TARGET = V_OH_TARGET - V_OL_TARGET # Target separation (2.5V)
        
        # Weights (Sum to 1.0)
        W_ER = 0.6
        W_STR = 0.2
        W_CONS = 0.2

        instruction_scores = []
        
        for ops in self.test_ops: 
            # Set Inputs
            for k, h_inp in enumerate(INPUT_HEATERS):
                self.controller.set(h_inp, V_LOW_LOGIC if ops[k] == 0 else V_HIGH_LOGIC)
            
            time.sleep(0.1) 
            read = self.scopes.read_many(avg=1)
            if read is None or len(read) < 6: return -2.0
            
            try:
                vals = np.nan_to_num(np.array(read[0:6]), nan=0.0)
                targets = np.array(get_expected_logic(*ops))
                high_volts, low_volts = vals[targets == 1], vals[targets == 0]

                if len(high_volts) == 0 or len(low_volts) == 0:
                    instruction_scores.append(-1.0); continue

                min_high = np.min(high_volts)
                max_low  = np.max(low_volts)

                # --- TARGET-BASED NORMALIZATION ---

                # 1. Extinction/Gap Score (0.7 Weight)
                # We target the actual separation gap. 
                # If gap >= 2.5V, score approaches 1.0.
                actual_gap = min_high - max_low
                er_score = np.tanh(actual_gap / GAP_TARGET)
                
                # 2. Strength Score (0.2 Weight)
                # Reward high absolute voltage. If min_high >= 3.5V, score approaches 1.0.
                str_score = np.tanh(min_high / V_OH_TARGET)
                
                # 3. Low-Level Penalty (Integrated into consistency or as a gate)
                # We want max_low to be as close to 0 as possible. 
                # If max_low exceeds V_OL_TARGET, we reduce the consistency score.
                low_penalty = np.exp(-max(0, max_low - V_OL_TARGET) / 0.5)
                
                variation = np.std(high_volts) + np.std(low_volts)
                cons_score = np.exp(-variation / 0.4) * low_penalty

                # Combine into final weighted FOM
                fom = (W_ER * er_score) + (W_STR * str_score) + (W_CONS * cons_score)
                instruction_scores.append(fom)

            except Exception as e:
                print(f"Error in math: {e}")
                instruction_scores.append(-1.0)
            
        return float(np.min(instruction_scores))
    
    def fit_gp(self):
        if len(self.X) < 10: return None
        n_dims = len(MODIFIABLE_HEATERS)
        kernel = (ConstantKernel(1.0) * Matern(length_scale=[2.0]*n_dims, nu=2.5) + 
                  WhiteKernel(noise_level=1e-4))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
        gp.fit(np.array(self.X), np.array(self.y))
        return gp

    def explore_around_best(self, n_samples=15, radius=0.3):
        """ The 'Hill-Climbing' strategy from your 2-bit code """
        if self.best_config is None: return
        print(f"\n--- Local Tweak around Best (Radius ±{radius}V) ---")
        base_x = np.array(self.best_config)
        
        for i in range(n_samples):
            # Tweak 15% of heaters randomly
            indices = np.random.choice(len(MODIFIABLE_HEATERS), int(USE * 0.15), replace=False)
            new_x = base_x.copy()
            new_x[indices] += np.random.uniform(-radius, radius, len(indices))
            new_x = np.clip(new_x, 0.1, 4.9)
            
            score = self.evaluate(new_x)
            self.add_eval(new_x, score)
            print(f"  Local {i+1}/{n_samples}: {score:.3f}")

    def add_eval(self, x, score):
        self.X.append(x)
        self.y.append(score)
        if score > self.best_score:
            self.best_score, self.best_config = score, x
            print(f"  ★ NEW BEST: {score:.3f}")

    def optimize(self, total_cycles=20):
        # 1. Initial Sampling
        print("Starting Initial Latin Hypercube Sampling (50 samples)...")
        sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS))
        for x in (sampler.random(10) * 4.5 + 0.1):
            self.add_eval(x, self.evaluate(x))

        # 2. Main Hybrid Loop
        for cycle in range(total_cycles):
            print(f"\n=== Hybrid Cycle {cycle+1}/{total_cycles} ===")
            gp = self.fit_gp()
            
            # --- INCREASED CANDIDATE DENSITY ---
            # 80% random (Global), 20% near best (Local refinement)
            n_cands = 1000 
            cands_rand = np.random.uniform(0.1, 4.5, (int(n_cands*0.8), USE))
            cands_local = np.clip(np.array(self.best_config) + np.random.normal(0, 0.4, (int(n_cands*0.2), USE)), 0.1, 4.9)
            candidates = np.vstack([cands_rand, cands_local])
            
            # 3. EVALUATE A BATCH OF BO SUGGESTIONS
            # This makes the BO work harder before the next Local Search
            print(f"GP predicting {n_cands} candidates...")
            
            # Adaptive UCB (Exploration vs Exploitation)
            beta = 3.0 if self.best_score < 0.4 else 2.5
            mu, sigma = gp.predict(candidates, return_std=True)
            ucb_values = mu + beta * sigma
            
            # Get the top 3 unique suggestions from the BO
            best_indices = np.argsort(ucb_values)[-5:][::-1] 
            
            print(f"Evaluating top 3 BO suggestions (beta={beta})...")
            for idx in best_indices:
                suggestion = candidates[idx]
                score = self.evaluate(suggestion)
                self.add_eval(suggestion, score)
                print(f"  BO Suggestion Score: {score:.3f}")

            # 4. Trigger Local Search less frequently
            # Now it only triggers every 3 cycles to give BO more 'room' to work
            if (cycle + 1) % 3 == 0:
                # Fewer local samples so it doesn't dominate the budget
                self.explore_around_best(n_samples=5, radius=0.35)

        self.save_best()

    def save_best(self):
        """
        Saves the configuration in the specific 44-value list format.
        Matches the structure: {"bg_weights": [...44 values...], "score": 0.0}
        """
        if self.best_config is not None:
            # 1. Convert the 44 optimized numpy values to a standard Python list
            weights_list = self.best_config.tolist()
            
            # 2. Construct the exact dictionary structure you provided
            output_data = {
                "bg_weights": weights_list,
                "score": float(self.best_score)
            }
            
            # 3. Save to file
            with open("decoder_hybrid_best.json", "w") as f:
                json.dump(output_data, f, indent=4)
            
            print("\n" + "="*40)
            print("Configuration saved!")
            print(f"File: decoder_hybrid_best.json")
            print(f"Final Score: {self.best_score:.6f}")
            print("="*40)

if __name__ == "__main__":
    opt = HybridDecoderBO()
    try: opt.optimize()
    except KeyboardInterrupt: opt.save_best()
    finally: opt.scopes.close()