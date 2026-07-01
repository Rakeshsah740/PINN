# ============================================================
# XGBOOST FOR FATIGUE LIFE PREDICTION
# ============================================================

import pandas as pd
import jax.numpy as jnp
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. LOAD YOUR DATA 
# ============================================================

df_train = pd.read_excel("V3_train.xlsx")
df_test = pd.read_excel("V3_test.xlsx")

# Feature columns
feature_columns = [
    'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
    'Pb 82', 'Sn 50', 'Ti 22',
    'T5 ?', 'T6 ?', 'T7 ?',
    'sigma_a'
]

# Prepare data
X_train = df_train[feature_columns].values.astype(float)
#y = np.log10(df['N'].values.astype(float))
y_train=jnp.log10(df_train['N'].values.astype(float))

X_test = df_test[feature_columns].values.astype(float)
y_test = jnp.log10(df_test['N'].values.astype(float))

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""

# ============================================================
# 3. TRAIN XGBOOST MODEL
# ============================================================


model = xgb.XGBRegressor(
    n_estimators=200,                           # Number of trees
    max_depth=6,                                # Maximum tree depth
    learning_rate=0.075,                         # Learning rate   
    subsample=0.8,                              # Subsample ratio of the training instances
    colsample_bytree=0.8,                       # Subsample ratio of columns when constructing each tree    
    random_state=42,                            # For reproducibility
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False                              # Suppress training logs
)

# ============================================================
# 4. EVALUATE
# ============================================================

y_pred = model.predict(X_test)

#mse = np.mean((y_pred - y_test)**2)
mse = jnp.mean((y_pred - y_test)**2)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("XGBOOST RESULTS")
print("="*50)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f} log10 cycles")
print(f"Test R²:  {r2:.4f}")

# Factor analysis
N_pred_cycles = 10 ** y_pred
N_true_cycles = 10 ** y_test
ratio = N_pred_cycles / N_true_cycles

#within_2x = jnp.mean((ratio >= 0.5) & (ratio <= 2))
within_2x = jnp.mean((ratio >= 0.5) & (ratio <= 2))
within_5x = jnp.mean((ratio >= 0.2) & (ratio <= 5))
within_10x = jnp.mean((ratio >= 0.1) & (ratio <= 10))

print(f"\nPredictions within:")
print(f"  2x: {within_2x:.1%}")
print(f"  5x: {within_5x:.1%}")
print(f"  10x: {within_10x:.1%}")

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10))

# ============================================================
# 6. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    'True_N_log10': y_test,
    'Predicted_N_log10': y_pred,
    'True_N_cycles': N_true_cycles,
    'Predicted_N_cycles': N_pred_cycles,
    'Error_log10': y_pred - y_test
})

results.to_excel('XGBoost_Results.xlsx', index=False)
print("\nResults saved to: XGBoost_Results.xlsx")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
#plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line for perfect predictions
plt.xlabel('True log10(N)') 
plt.ylabel('Predicted log10(N)')
plt.title('XGBoost: True vs Predicted log10(N), R² = {:.3f}'.format(r2))


importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

top10 = importance.head(10)

# Plot only top 10
plt.figure(figsize=(10, 6))
plt.barh(top10['feature'], top10['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 10 XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



""" 
from setgraphviz import setgraphviz


booster = model.get_booster()

# Plot Tree 1 
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
xgb.plot_tree(booster, num_trees=0, ax=ax1)
ax1.set_title('Tree 1')
plt.tight_layout()
plt.savefig('tree_1.png', dpi=300)

plt.show()
print("\nDone!")



#stress_range = np.linspace(50, 500, 500) 
stress_range = np.linspace(50, 500, 1000)  # More points for smoother curve

# Replicate your alloy composition for every stress level in the range
alloy_data = []
for sigma in stress_range:
    row = [
        92.3150, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480, # Elements (Al to Mg)
        0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280, # Elements (Cr to Ti)
        0, 1, 0,                                        # T5=0, T6=1, T7=0
        sigma                                           # The changing stress level
    ]
    alloy_data.append(row)



# Convert list to a matrix array for the model
X_sweep = np.array(alloy_data)

# ============================================================
# 2. PREDICT CODES (Assumes model was trained on log10(N))
# ============================================================
predicted_log_N = model.predict(X_sweep)

# ============================================================
# 3. FIND THE CLOSEST STRESS FOR N = 10,000 (log10 = 4)
# ============================================================
target_log = 4.0  # because log10(10000) = 4

# Find where the predicted value is closest to 4.0
closest_index = np.abs(predicted_log_N - target_log).argmin()
estimated_stress = stress_range[closest_index]

print("=" * 50)
print(f"Target Fatigue Life: 10,000 cycles")
print(f"Estimated Required Stress: {estimated_stress:.2f} MPa")
print(f"Model's Closest Prediction: 10^{predicted_log_N[closest_index]:.4f} cycles")
print("=" * 50)

"""

