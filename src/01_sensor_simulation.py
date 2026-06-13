!mamba install numpy pandas matplotlib scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# 1. Define Simulation Parameters
np.random.seed(42)
num_samples = 1000
time_steps = np.arange(num_samples)

# Simulate a true environmental variable (e.g., actual air quality or temperature)
# Using a sine wave to represent natural diurnal fluctuations
true_state = 50 + 20 * np.sin(2 * np.pi * time_steps / 200)

# 2. Simulate Low-Cost Sensor Flaws (MQ-series/DHT)
# Flaw A: Stochastic Noise (High variance, random signal interference)
stochastic_noise = np.random.normal(0, 5, num_samples)  # Represents the ±12% baseline error margin

# Flaw B: Temporal Drift (Measurement degradation over time)
# Simulating a gradual upward drift in the sensor reading
temporal_drift = 0.015 * time_steps 

# 3. Generate the Raw Sensor Reading
raw_sensor_reading = true_state + stochastic_noise + temporal_drift

# Create a redundant array of 3 slightly different cheap sensors for the fusion step later
sensor_array = {
    'Time': time_steps,
    'True_Value': true_state,
    'Sensor_1_Raw': raw_sensor_reading,
    'Sensor_2_Raw': true_state + np.random.normal(0, 6, num_samples) + (0.012 * time_steps),
    'Sensor_3_Raw': true_state + np.random.normal(0, 4.5, num_samples) + (0.018 * time_steps)
}

df = pd.DataFrame(sensor_array)

# Save the synthetic dataset
output_path = '../data/simulated_sensor_array_data.csv'
df.to_csv(output_path, index=False)
print(f"✅ Simulated dataset saved to {output_path}")

# Optional: Visualize the flaw (True vs Raw)
plt.figure(figsize=(10, 5))
plt.plot(df['Time'], df['True_Value'], label='True Environmental State', color='green', linewidth=2)
plt.plot(df['Time'], df['Sensor_1_Raw'], label='Raw Sensor 1 (Noise + Drift)', color='red', alpha=0.6)
plt.title('Simulation of Low-Cost Sensor Limitations')
plt.xlabel('Time Steps')
plt.ylabel('Measurement Value')
plt.legend()
plt.grid(True)
plt.show()
