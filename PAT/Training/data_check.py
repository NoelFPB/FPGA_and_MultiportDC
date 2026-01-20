import numpy as np

# Load the dataset you uploaded
data = np.load("chip_dataset.npz")
X = data['X']
y = data['y']

print("=== DATASET INSPECTION ===")
print(f"Total Samples: {len(X)}")
print(f"X (Heaters) Shape: {X.shape} | y (Scope) Shape: {y.shape}")

# Check Input Voltages
print(f"\nInput Voltage (X) Range: {X.min():.3f}V to {X.max():.3f}V")
print(f"X Mean: {X.mean():.3f}V | X Std: {X.std():.3f}V")

# Check Output Voltages (Scope)
print(f"\nOutput Voltage (y) Range: {y.min():.3f}V to {y.max():.3f}V")
print(f"y Mean: {y.mean():.3f}V | y Std: {y.std():.3f}V")

# Physical Check: If X.max() is < 1.0, your data is already scaled/normalized!
if X.max() < 1.0:
    print("\n⚠️ WARNING: X data appears to be pre-scaled. The sin(V^2) logic will break.")
if y.std() < 0.1:
    print("⚠️ WARNING: y data has very low variation. The scope might not have captured interference.")