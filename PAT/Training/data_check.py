import matplotlib.pyplot as plt
import numpy as np
# Load your data
data = np.load("chip_dataset.npz")
y = data['y']

# Check the distribution
plt.hist(y.flatten(), bins=50)
plt.title("Distribution of Scope Voltages")
plt.xlabel("Volts")
plt.ylabel("Frequency")
plt.show()

print(f"Standard Deviation of data: {np.std(y):.4f}")


data = np.load("chip_dataset.npz")
y = data['y']

# Split the data into the beginning, middle, and end of your lab session
start_mean = np.mean(y[:1000])
mid_mean = np.mean(y[20000:21000])
end_mean = np.mean(y[-1000:])

print(f"Mean Voltage (Start): {start_mean:.4f}V")
print(f"Mean Voltage (Mid):   {mid_mean:.4f}V")
print(f"Mean Voltage (End):   {end_mean:.4f}V")

drift = abs(start_mean - end_mean)
print(f"Total Session Drift: {drift:.4f}V")

from scipy.stats import pearsonr

X = data['X']
print(X)
y = data['y']

# Check correlation between total heater power and the first scope channel
# We use the absolute value because interference can be positive or negative
corr, _ = pearsonr(np.sum(X, axis=1), y[:, 0])
print(f"Input-Output Correlation: {corr:.4f}")

# Check for unique outputs
unique_outputs = len(np.unique(y.round(decimals=4), axis=0))
print(f"Unique Samples: {unique_outputs} out of {len(y)}")


import numpy as np
data = np.load("chip_dataset.npz")
X, y = data['X'], data['y']

# Check correlation for each of the 6 outputs separately
for i in range(6):
    c = np.corrcoef(np.sum(X, axis=1), y[:, i])[0, 1]
    print(f"Global Correlation Output Ch{i}: {c:.6f}")

    # Check if Heater #42 (an input) has any individual correlation
h_idx = 3 
c_single = np.corrcoef(X[:, h_idx], y[:, 0])[0, 1]
print(f"Single Heater (#42) Correlation: {c_single:.6f}")