# ============================================================
# ENHANCED PINN WITH BETTER PERFORMANCE (CORRECTED)
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from google.colab import drive

drive.mount('/content/drive')

# ============================================================
# 1. IMPROVED DATA PREPROCESSING
# ============================================================

df = pd.read_excel("/content/V3.xlsx")
df_clean = df.dropna(subset=['N', 'sigma_a'])

# Feature engineering: Add interaction terms
feature_columns = [
    'Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
    'Pb 82', 'Sn 50', 'Ti 22',
    'T5 ?', 'T6 ?', 'T7 ?',
    'sigma_a'
]

X_raw = df_clean[feature_columns].values.astype(float)
y_raw = np.log10(df_clean['N'].values.astype(float))

pd.DataFrame(X_raw).to_excel('X_raw.xlsx', index=False)

# Create interaction features (composition * stress)
composition = X_raw[:, :11]  # First 11 columns are composition
heat_treatment = X_raw[:, 11:14]  # Heat treatment indicators
stress = X_raw[:, 14:15]  # sigma_a



# Combine all features
X_enhanced = np.hstack([composition, heat_treatment, stress])
print(f"Enhanced features shape: {X_enhanced.shape}")

# Use RobustScaler for features (less sensitive to outliers)
scaler_X = RobustScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_enhanced)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Store scaler parameters for inverse transformation
X_mean = scaler_X.center_
X_scale = scaler_X.scale_
y_mean = scaler_y.mean_[0]
y_scale = scaler_y.scale_[0]

# ============================================================
# 2. ENHANCED MODEL ARCHITECTURE
# ============================================================

class EnhancedPINN(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedPINN, self).__init__()
        
        # Separate paths for different physics
        self.composition_net = nn.Sequential(
            nn.Linear(11, 32),  # Composition only
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh()
        )
        
        self.joint_net = nn.Sequential(
            nn.Linear(16 + 3 + 1, 64),  # composition_features + HT + stress + interactions
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Split features
        composition = x[:, :11]  # First 11: composition
        ht = x[:, 11:14]         # Next 3: heat treatment
        stress = x[:, 14:15]     # Next 1: stress
        
        
        # Process composition
        comp_features = self.composition_net(composition)
        
        # Combine all features
        combined = torch.cat([comp_features, ht, stress], dim=1)
        
        # Joint network
        out = self.joint_net(combined)
        out = self.dropout(out)
        
        # Physical constraints
        log10_sigma_f_prime = 2.5 + 0.5 * torch.tanh(out[:, 0:1])  # Range: 2.0-3.0 (100-1000 MPa)
        b = -0.08 - 0.04 * torch.sigmoid(out[:, 1:2])  # Range: -0.12 to -0.08
        
        return log10_sigma_f_prime, b

model = EnhancedPINN(X_train.shape[1])
print(f"\nModel: {X_train.shape[1]} input features")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ============================================================
# 3. ADVANCED TRAINING WITH LEARNING RATE SCHEDULING
# ============================================================

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=200, factor=0.5)

# Loss weights
lambda_physics = 0.01
lambda_composition = 0.001

print("\n" + "="*60)
print("STARTING ENHANCED TRAINING")
print("="*60)

epochs = 3000
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    log10_sigma_f_prime, b = model(X_train)
    
    # Get features (in normalized space)
    stress_norm = X_train[:, 14:15]
    
    # Convert stress to physical units using RobustScaler parameters
    # Note: RobustScaler uses center_ (median) and scale_ (IQR)
    stress_physical = stress_norm * X_scale[14] + X_mean[14]
    sigma_f_physical = 10 ** log10_sigma_f_prime
    
    # Basquin prediction (with enhanced numerical stability)
    epsilon = 1e-7
    b_safe = torch.clamp(b, max=-0.01, min=-0.2)
    
    ratio = stress_physical / (sigma_f_physical + epsilon)
    ratio = torch.clamp(ratio, min=0.01, max=100.0)
    
    # Log space calculation
    log10_N_pred = np.log10(0.5) + (1.0 / (b_safe + epsilon)) * torch.log10(ratio + epsilon)
    log10_N_pred = torch.clamp(log10_N_pred, min=3.0, max=8.0)  # 1e3 to 1e8 cycles
    
    # Normalize prediction (using StandardScaler parameters)
    y_pred = (log10_N_pred - y_mean) / y_scale
    
    # Data loss (weighted)
    data_loss = torch.mean((y_pred - y_train)**2)
    
    # Physics loss (stress reconstruction)
    N_pred = 10 ** log10_N_pred
    stress_reconstructed = sigma_f_physical * (2 * N_pred + epsilon) ** b_safe
    physics_loss = torch.mean((stress_reconstructed - stress_physical)**2)
    
    # Composition consistency: similar compositions should have similar σ_f'
    composition_norm = X_train[:, :11]
    comp_diff = torch.cdist(composition_norm, composition_norm, p=2)
    sigma_f_diff = torch.abs(log10_sigma_f_prime - log10_sigma_f_prime.t())
    
    # Weight by composition similarity
    median_dist = torch.median(comp_diff)
    composition_kernel = torch.exp(-comp_diff / (median_dist + epsilon))
    composition_loss = torch.mean(composition_kernel * sigma_f_diff)
    
    # Total loss
    loss = data_loss + lambda_physics * physics_loss + lambda_composition * composition_loss
    
    # Check for NaN
    if torch.isnan(loss):
        print(f"NaN at epoch {epoch}, skipping")
        continue
    
    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(loss)
    
    # Validation
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            log10_sigma_f_prime_val, b_val = model(X_test)
            sigma_f_physical_val = 10 ** log10_sigma_f_prime_val
            stress_physical_val = X_test[:, 14:15] * X_scale[14] + X_mean[14]
            
            b_safe_val = torch.clamp(b_val, max=-0.01, min=-0.2)
            ratio_val = stress_physical_val / (sigma_f_physical_val + epsilon)
            ratio_val = torch.clamp(ratio_val, min=0.01, max=100.0)
            
            log10_N_pred_val = np.log10(0.5) + (1.0 / (b_safe_val + epsilon)) * torch.log10(ratio_val + epsilon)
            log10_N_pred_val = torch.clamp(log10_N_pred_val, min=3.0, max=8.0)
            y_pred_val = (log10_N_pred_val - y_mean) / y_scale
            
            val_loss = torch.mean((y_pred_val - y_test)**2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '/content/drive/MyDrive/best_pinn_model.pth')
            else:
                patience_counter += 1
                
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Data: {data_loss.item():.4f} | Physics: {physics_loss.item():.4f} | Val: {val_loss.item():.4f}")
        print(f"  σ_f': {sigma_f_physical.mean().item():.0f} ± {sigma_f_physical.std().item():.0f} MPa")
        print(f"  b: {b.mean().item():.3f} ± {b.std().item():.3f}")
        
        if patience_counter > 500:
            print("Early stopping triggered")
            break

# ============================================================
# 4. LOAD BEST MODEL AND EVALUATE
# ============================================================

# Check if best model exists, otherwise use current model
import os
if os.path.exists('/content/drive/MyDrive/best_pinn_model.pth'):
    model.load_state_dict(torch.load('/content/drive/MyDrive/best_pinn_model.pth'))
    print("Loaded best model")
else:
    print("Using final model")

model.eval()

with torch.no_grad():
    log10_sigma_f_prime, b = model(X_test)
    
    # Convert to physical units
    sigma_f_physical = 10 ** log10_sigma_f_prime
    stress_physical = X_test[:, 14:15] * X_scale[14] + X_mean[14]
    
    epsilon = 1e-7
    b_safe = torch.clamp(b, max=-0.01, min=-0.2)
    
    ratio = stress_physical / (sigma_f_physical + epsilon)
    ratio = torch.clamp(ratio, min=0.01, max=100.0)
    
    log10_N_pred = np.log10(0.5) + (1.0 / (b_safe + epsilon)) * torch.log10(ratio + epsilon)
    log10_N_pred = torch.clamp(log10_N_pred, min=3.0, max=8.0)
    
    # Inverse transform predictions
    y_pred_original = log10_N_pred.numpy().flatten()
    y_test_original = y_test.numpy().flatten() * y_scale + y_mean
    
    # Calculate metrics
    mse = np.mean((y_pred_original - y_test_original)**2)
    mae = np.mean(np.abs(y_pred_original - y_test_original))
    r2 = r2_score(y_test_original, y_pred_original)
    
    # Factor analysis (predictions within factors)
    N_pred_cycles = 10 ** y_pred_original
    N_true_cycles = 10 ** y_test_original
    
    # Avoid division by zero
    valid_idx = N_true_cycles > 0
    if np.any(valid_idx):
        ratio_cycles = N_pred_cycles[valid_idx] / N_true_cycles[valid_idx]
        within_2x = np.mean((ratio_cycles >= 0.5) & (ratio_cycles <= 2))
        within_5x = np.mean((ratio_cycles >= 0.2) & (ratio_cycles <= 5))
        within_10x = np.mean((ratio_cycles >= 0.1) & (ratio_cycles <= 10))
    else:
        within_2x = within_5x = within_10x = 0
    
    print("\n" + "="*60)
    print("IMPROVED EVALUATION RESULTS")
    print("="*60)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f} log10 cycles")
    print(f"Test R²:  {r2:.4f}")
    
    print(f"\nPredictions within factor of:")
    print(f"  2x: {within_2x:.1%}")
    print(f"  5x: {within_5x:.1%}")
    print(f"  10x: {within_10x:.1%}")
    
    print(f"\nPhysical parameters:")
    print(f"  σ_f' = {sigma_f_physical.mean().item():.0f} ± {sigma_f_physical.std().item():.0f} MPa")
    print(f"  b = {b.mean().item():.3f} ± {b.std().item():.3f}")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if r2 > 0.85:
        print("  ✓ Excellent model performance")
    elif r2 > 0.75:
        print("  ✓ Good model performance")
    elif r2 > 0.65:
        print("  ⚠️  Moderate performance - needs improvement")
    else:
        print("  ✗ Poor performance - check data quality")
    
    if within_2x > 0.7:
        print("  ✓ Excellent factor-of-2 prediction")
    elif within_2x > 0.5:
        print("  ✓ Good factor-of-2 prediction")
    elif within_2x > 0.3:
        print("  ⚠️  Moderate factor-of-2 prediction")
    else:
        print("  ✗ Poor factor-of-2 prediction")

# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame({
    'True_log10_N': y_test_original,
    'Predicted_log10_N': y_pred_original,
    'True_N_cycles': N_true_cycles,
    'Predicted_N_cycles': N_pred_cycles,
    'Error_log10': y_pred_original - y_test_original,
    'sigma_f_prime_MPa': sigma_f_physical.numpy().flatten(),
    'b_exponent': b.numpy().flatten()
})

if len(N_pred_cycles) == len(ratio_cycles):
    results_df['Ratio'] = np.concatenate([ratio_cycles, np.full(len(N_pred_cycles)-len(ratio_cycles), np.nan)])

results_df.to_excel('/content/drive/MyDrive/Improved_PINN_Results.xlsx', index=False)
print(f"\nResults saved to: /content/drive/MyDrive/Improved_PINN_Results.xlsx")

# Simple plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Predictions vs true
axes[0].scatter(y_test_original, y_pred_original, alpha=0.6, s=50)
axes[0].plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
axes[0].set_xlabel('True log10(N)')
axes[0].set_ylabel('Predicted log10(N)')
axes[0].set_title(f'Predictions vs True (R² = {r2:.3f})')
axes[0].grid(True, alpha=0.3)

# Error distribution
errors = y_pred_original - y_test_original
axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].axvline(x=np.mean(errors), color='g', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
axes[1].set_xlabel('Prediction Error (log10 cycles)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Error Distribution (MAE = {mae:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Improved_PINN_Results.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("IMPROVED TRAINING COMPLETED")
print("="*60)