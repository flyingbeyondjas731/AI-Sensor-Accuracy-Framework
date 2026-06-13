!mamba install numpy pandas matplotlib scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the simulated noisy data generated in the previous step
# Note: Ensure this path matches where your CSV was saved
data_path = '../data/simulated_sensor_array_data.csv' 
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    # Fallback for Jupyter if paths get mixed up
    df = pd.read_csv('simulated_sensor_array_data.csv') 

# 2. Define the 1D Kalman Filter Function
def apply_kalman_filter(measurements, Q=1e-4, R=25):
    """
    Q: Process variance (How much we trust our internal model's prediction)
    R: Measurement variance (How much we trust the noisy sensor)
    """
    n_iter = len(measurements)
    sz = (n_iter,)
    
    # Allocate space for arrays
    xhat = np.zeros(sz)      # a posteriori estimate of x
    P = np.zeros(sz)         # a posteriori error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # Kalman gain
    
    # Initial guesses
    xhat[0] = measurements[0]
    P[0] = 1.0
    
    for k in range(1, n_iter):
        # Time update (Predict)
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        # Measurement update (Correct)
        K[k] = Pminus[k] / ( Pminus[k] + R )
        xhat[k] = xhatminus[k] + K[k] * (measurements[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat

# 3. Apply the filter to Sensor 1
# We tune Q and R to aggressively filter the stochastic noise
filtered_signal = apply_kalman_filter(df['Sensor_1_Raw'].values, Q=1e-4, R=25)
df['Sensor_1_Filtered'] = filtered_signal

# 4. Visualize the improvement
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['True_Value'], label='True State', color='green', linewidth=2)
plt.plot(df['Time'], df['Sensor_1_Raw'], label='Raw Sensor (Noise + Drift)', color='red', alpha=0.3)
plt.plot(df['Time'], df['Sensor_1_Filtered'], label='Kalman Filtered Signal', color='blue', linewidth=2)
plt.title('Recursive Kalman Filtering for Noise Suppression')
plt.xlabel('Time Steps')
plt.ylabel('Measurement Value')
plt.legend()
plt.grid(True)
plt.show()

# Save the updated dataset for the next step (Sensor Fusion)
try:
    df.to_csv('../data/filtered_sensor_data.csv', index=False)
except OSError:
    df.to_csv('filtered_sensor_data.csv', index=False)
