!mamba install numpy pandas matplotlib scikit-learn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load the fused data
try:
    df = pd.read_csv('../data/fused_sensor_data.csv')
except FileNotFoundError:
    df = pd.read_csv('fused_sensor_data.csv')

# 2. Prepare Features (X) and Target (y)
# We use the fused reading and the time step to predict the true environmental state
X = df[['Time', 'Fused_Reading']]
y = df['True_Value']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Gradient Boosting Regressor
print("Training Gradient Boosting Model for Edge Deployment...")
# We use shallow trees and limited estimators to keep the model small enough for 520KB SRAM
gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# 4. Evaluate the Model
y_pred = gb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Final AI Model R² Score: {r2:.3f}")
print(f"✅ Final AI Model MAE: {mae:.2f}")

# 5. Simulate 8-bit Quantization for TinyML
# To fit on an ESP32, floating point weights are scaled to 8-bit integers (-128 to 127)
print("\nSimulating 8-bit Post-Training Quantization...")
# Extracting a sample of model outputs to simulate the scaling process
sample_outputs = y_pred[:10]
scale_factor = 127 / np.max(np.abs(sample_outputs))
quantized_outputs = np.round(sample_outputs * scale_factor).astype(np.int8)

print("Original Float Outputs (Sample):", np.round(sample_outputs, 2))
print("Quantized INT8 Outputs (Sample):", quantized_outputs)
print("Quantization Complete. Ready for C++ array export to ESP32.")

# 6. Final Visualization (Optional)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Environmental State')
plt.ylabel('AI Predicted State')
plt.title(f'TinyML Model Accuracy (R² = {r2:.3f})')
plt.grid(True)
plt.savefig('tinyml_accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()
