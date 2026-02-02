import numpy as np
import matplotlib.pyplot as plt

data = np.load("full_chip_characterization.npz")
X, y = data['X'], data['y']

# Let's pick Heater #0 and Output Channel #0 as an example
h_idx = 32 
out_idx = 1

plt.figure(figsize=(8, 5))
# Plotting the Heater Voltage Squared (Power) vs Output Voltage
plt.scatter(X[:, h_idx]**2, y[:, out_idx], alpha=0.1, s=2, color='blue')

plt.title(f"Phase Shifter Response: Heater {h_idx} vs. Output {out_idx}")
plt.xlabel(f"Phase Shifter Power (Voltage $V^2$)")
plt.ylabel("Scope Intensity (V)")
plt.grid(True, alpha=0.3)
plt.show()