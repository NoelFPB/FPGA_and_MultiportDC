import json
from Lib.DualBoard import DualAD5380Controller

def apply_background_weights(config_file="decoder_hybrid_best.json"):
    # 1. Initialize the DAC Controller
    print("Connecting to hardware...")
    controller = DualAD5380Controller()
    
    # 2. Define Heater Mapping
    # (Matches your logic: 21 heaters total, skipping indices 15-19 used for inputs)
    INPUT_HEATERS = [15, 16, 17, 18, 19] 
    opt_indices = [h for h in range(14) if h not in INPUT_HEATERS]
    
    # 3. Load and Apply Weights
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            weights = config["bg_weights"]
            
        print(f"Applying {len(weights)} weights from {config_file}...")
        
        for i, h_idx in enumerate(opt_indices):
            controller.set(h_idx, float(weights[i]))
            
        print("Success: Hardware configuration updated.")
        
    except FileNotFoundError:
        print(f"ERROR: {config_file} not found.")
    except KeyError:
        print(f"ERROR: 'bg_weights' key not found in {config_file}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    apply_background_weights()