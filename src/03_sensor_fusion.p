!mamba install numpy pandas matplotlib scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset containing our array of 3 sensors
try:
    df = pd.read_csv('../data/simulated_sensor_array_data.csv')
except FileNotFoundError:
    df = pd.read_csv('simulated_sensor_array_data.csv')

# 2. Adaptive Sensor Fusion (Dynamic Weighted Averaging)
# Calculate rolling variance for each sensor (window size = 10 steps) to measure real-time noise
window = 10
var_1 = df['Sensor_1_Raw'].rolling(window=window, min_periods=1).var().replace(0, 1e-6)
var_2 = df['Sensor_2_Raw'].rolling(window=window, min_periods=1).var().replace(0, 1e-6)
var_3 = df['Sensor_3_Raw'].rolling(window=window, min_periods=1).var().replace(0, 1e-6)

# Calculate weights based on Inverse Variance
# A higher variance means lower confidence, resulting in a lower weight for that sensor
inv_var_1 = 1 / var_1
inv_var_2 = 1 / var_2
inv_var_3 = 1 / var_3
total_inv_var = inv_var_1 + inv_var_2 + inv_var_3

w1 = inv_var_1 / total_inv_var
w2 = inv_var_2 / total_inv_var
w3 = inv_var_3 / total_inv_var

# Apply the weights to get the final fused reading
df['Fused_Reading'] = (w1 * df['Sensor_1_Raw']) + (w2 * df['Sensor_2_Raw']) + (w3 * df['Sensor_3_Raw'])
df['Fused_Reading'] = df['Fused_Reading'].fillna(df['Sensor_1_Raw']) # Fill initial gaps

# 3. Calculate the Error Margin Reduction
mae_raw = np.mean(np.abs(df['Sensor_1_Raw'] - df['True_Value']))
mae_fused = np.mean(np.abs(df['Fused_Reading'] - df['True_Value']))

print(f"Mean Absolute Error (Single Raw Sensor): {mae_raw:.2f}")
print(f"Mean Absolute Error (Fused Array): {mae_fused:.2f}")
print(f"Error Reduction Achieved: {((mae_raw - mae_fused) / mae_raw) * 100:.1f}%")

# 4. Visualize the Fusion Result
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['True_Value'], label='True State', color='green', linewidth=3)
plt.plot(df['Time'], df['Sensor_1_Raw'], label='Sensor 1 (Highly Noisy)', color='red', alpha=0.2)
plt.plot(df['Time'], df['Fused_Reading'], label='Adaptive Fused Output', color='purple', linewidth=2)
plt.title('Adaptive Sensor Fusion (Dynamic Weighted Averaging)')
plt.xlabel('Time Steps')
plt.ylabel('Measurement Value')
plt.legend()
plt.grid(True)
plt.show()

# Save the fused dataset for the final TinyML step
try:
    df.to_csv('../data/fused_sensor_data.csv', index=False)
except OSError:
    df.to_csv('fused_sensor_data.csv', index=False)
