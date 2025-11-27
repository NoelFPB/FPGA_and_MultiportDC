import time
import numpy as np
import random
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes


# ============================================================
#  CONFIGURATION
# ============================================================
# The heaters that you can use as inputs are 28,29,30,31,32,33,334


GATE_TYPE = "AND"
INPUT_HEATERS = [27, 28]       # Input A, B heaters
V_LOW = 0.1 # 0 doesnt work for some reason
V_HIGH = 4.9

# 8-channel readout: (scope1 CH1..CH4, scope2 CH1..CH4)
#       0        1         2         3         4         5         6         7
#   [S1_CH1, S1_CH2, S1_CH3, S1_CH4, S2_CH1, S2_CH2, S2_CH3, S2_CH4]
OUTPUT_CHANNEL_INDEX = 1   # <-- CHANGE HERE to select logic output channel

MODIFIABLE_HEATERS = [h for h in range(40) if h not in INPUT_HEATERS]
INPUT_COMBINATIONS = [
    (V_LOW, V_LOW),
    (V_LOW, V_HIGH),
    (V_HIGH, V_LOW),
    (V_HIGH, V_HIGH)
]


# ============================================================
#  TRUTH TABLE
# ============================================================

def generate_truth_table(gate_type):
    truth = {
        "AND":  [0,0,0,1],
        "OR":   [0,1,1,1],
        "NAND": [1,1,1,0],
        "NOR":  [1,0,0,0],
        "XOR":  [0,1,1,0],
        "XNOR": [1,0,0,1]
    }
    return truth[gate_type]


# ============================================================
#  MAIN OPTIMIZER CLASS
# ============================================================

class LogicGateOptimizer:
    def __init__(self, gate_type=GATE_TYPE):
        self.gate_type = gate_type
        self.truth = generate_truth_table(gate_type)

        print(f"=== Optimizing {gate_type} Gate ===")
        for (inp, val) in zip(INPUT_COMBINATIONS, self.truth):
            print(f"  Input {inp} -> Expected: {'HIGH' if val else 'LOW'}")

        # Hardware interfaces
        self.controller = DualAD5380Controller()
        self.scopes = RigolDualScopes(
            channels_scope1=[1,2,3,4],
            channels_scope2=[1,2,3,4]
        )

        # BO storage
        self.X = []
        self.y = []
        self.best_config = None
        self.best_score = -1e9


    # ========================================================
    #  Heater Send
    # ========================================================
    def set_heaters(self, config):
        for h, v in config.items():
            self.controller.set(h, float(v))
        time.sleep(0.02)


    # ========================================================
    #  Measurement
    # ========================================================
    def measure_output(self):
        values = self.scopes.read_many(avg=1)
        if values is None or len(values) < 8:
            return None
        return values[OUTPUT_CHANNEL_INDEX]


    # ========================================================
    #  Evaluate one configuration
    # ========================================================
    def evaluate(self, config):

        high_vals = []
        low_vals  = []
        all_vals  = []

        for idx, (a, b) in enumerate(INPUT_COMBINATIONS):

            cfg = config.copy()
            cfg[INPUT_HEATERS[0]] = a
            cfg[INPUT_HEATERS[1]] = b

            self.set_heaters(cfg)
            time.sleep(0.15)

            val = self.measure_output()
            if val is None or np.isnan(val):
                return -1e9

            all_vals.append(val)
            if self.truth[idx] == 1:
                high_vals.append(val)
            else:
                low_vals.append(val)

        # Basic logic check
        if not high_vals or not low_vals:
            return -1e9

        min_high = min(high_vals)
        max_low  = max(low_vals)
        separation = min_high - max_low

        # scoring
        if separation <= 0:
            return separation * 20    # heavy penalty

        er_linear = min_high / max_low
        er_db = 10*np.log10(er_linear)

        # soft scoring
        score = (
            50 * np.tanh(separation) +
            30 * np.tanh(er_db / 5.0) +
            20 * np.tanh(np.mean(high_vals) / 4.0)
        )

        return float(score)


    # ========================================================
    #  BO dataset insert
    # ========================================================
    def add_eval(self, x, score):
        self.X.append(np.array(x, float))
        self.y.append(score)

        if score > self.best_score:
            self.best_score = score
            self.best_config = {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}
            print(f"New best {score:.2f}")


    # ========================================================
    #  Fit GP
    # ========================================================
    def fit_gp(self):
        if len(self.X) < 5:
            return None

        X = np.array(self.X)
        y = np.array(self.y)

        kernel = (
            ConstantKernel(1.0, (0.1,100)) *
            Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1,10)) +
            WhiteKernel(0.1, (1e-3,1))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        gp.fit(X, y)
        return gp


    # ========================================================
    #  Random config generator
    # ========================================================
    def random_config(self):
        x = np.random.uniform(V_LOW, V_HIGH, len(MODIFIABLE_HEATERS))
        return {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}, x


    # ========================================================
    #  Initial sampling
    # ========================================================
    def initial_sampling(self, n=20):
        print("\n=== Initial Sampling ===")
        sampler = qmc.LatinHypercube(len(MODIFIABLE_HEATERS))
        samples = sampler.random(n)

        for i in range(n):
            x = V_LOW + samples[i] * (V_HIGH - V_LOW)
            config = {h: x[j] for j, h in enumerate(MODIFIABLE_HEATERS)}

            score = self.evaluate(config)
            self.add_eval(x, score)

            print(f"  [{i+1}/{n}] Score = {score:.2f}")


    # ========================================================
    #  Main BO loop
    # ========================================================
    def optimize(self, iterations=20, candidates_per_iter=300):

        self.initial_sampling(2)

        for it in range(iterations):
            print(f"\n=== BO Iteration {it+1}/{iterations} ===")

            gp = self.fit_gp()
            if gp is None:
                print("  GP not ready, random sampling only.")
                for _ in range(3):
                    cfg, x = self.random_config()
                    score = self.evaluate(cfg)
                    self.add_eval(x, score)
                continue

            # generate candidate pool
            cand = np.random.uniform(V_LOW, V_HIGH, 
                                     size=(candidates_per_iter, len(MODIFIABLE_HEATERS)))

            mu, sigma = gp.predict(cand, return_std=True)
            ucb = mu + 2.0 * sigma

            best_idx = np.argsort(ucb)[-3:]   # top 3
            for idx in best_idx:
                x = cand[idx]
                config = {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}
                score = self.evaluate(config)
                self.add_eval(x, score)
                print(f"   → Score {score:.2f}")

        print("\n=== Optimization Done ===")
        self.test_best_configuration()
        print(f"Best Score: {self.best_score:.2f}")
       


    # ========================================================
    #  Final test of best configuration
    # ========================================================
    def test_best_configuration(self):
        if self.best_config is None:
            print("\n[FINAL TEST] No best configuration available.")
            return

        print("\n=== FINAL TEST OF BEST CONFIGURATION ===")

        results = []
        high_vals = []
        low_vals = []

        for idx, (a, b) in enumerate(INPUT_COMBINATIONS):

            cfg = self.best_config.copy()
            cfg[INPUT_HEATERS[0]] = a
            cfg[INPUT_HEATERS[1]] = b

            self.set_heaters(cfg)
            time.sleep(0.2)

            val = self.measure_output()
            val = float(val) if val is not None else np.nan

            expected = "HIGH" if self.truth[idx] == 1 else "LOW"
            results.append((a, b, expected, val))

            if self.truth[idx] == 1:
                high_vals.append(val)
            else:
                low_vals.append(val)

        # ---- PRINT TABLE ----
        print("\nInput A | Input B | Expected |  Measured Output (V)")
        print("------------------------------------------------------")
        for r in results:
            print(f"{r[0]:7.2f} | {r[1]:7.2f} | {r[2]:8s} | {r[3]:18.4f}")

        # ---- EXTINCTION RATIO ----
        if high_vals and low_vals:
            min_high = min(high_vals)
            max_low  = max(low_vals)
            sep = min_high - max_low

            if sep > 0:
                er_linear = min_high / max_low
                er_db = 10 * np.log10(er_linear)
                print("\nLogic gate working ✓")
                print(f"  Worst-case separation: {sep:.4f} V")
                print(f"  Extinction ratio:      {er_db:.2f} dB")
                print(f"  min(HIGH) = {min_high:.4f} V")
                print(f"  max(LOW)  = {max_low:.4f} V")
            else:
                print("\n⚠️  Logic levels overlap — gate not reliable.")
                print(f"  min(HIGH) = {min_high:.4f} V")
                print(f"  max(LOW)  = {max_low:.4f} V")
                print(f"  Overlap   = {-sep:.4f} V")

        # ---- PRETTY CONFIG PRINT ----
        print("\n=== FINAL HEATER CONFIGURATION ===")
        sorted_items = sorted(self.best_config.items(), key=lambda x: x[0])
        for h, v in sorted_items:
            print(f"Heater {h:2d}: {float(v):.4f} V")



    # ========================================================
    #  Cleanup
    # ========================================================
    def cleanup(self):
        try: self.scopes.close()
        except: pass


# ============================================================
#  RUN
# ============================================================

def main():
    opt = LogicGateOptimizer(GATE_TYPE)
    try:
        opt.optimize(iterations=10)
    finally:
        opt.cleanup()


if __name__ == "__main__":
    main()
