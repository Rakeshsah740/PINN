import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# ============================================================
# 1. DATA LOADING & PREPARATION
# ============================================================
"""
df = pd.read_excel("V3.xlsx")

feature_columns = [
    'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
    'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'sigma_a'
]

X = df[feature_columns].values.astype(float)
y = np.log10(df['N'].values.astype(float)).reshape(-1, 1)

# Keep track of where sigma_a is (it's the last column index: 15)
sigma_a_idx = 15

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)
""" 

df_train = pd.read_excel("V3_train.xlsx")
df_test = pd.read_excel("V3_test.xlsx")

feature_columns = [
    'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
    'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'sigma_a'
]
X_train_np = df_train[feature_columns].values.astype(float)
y_train_np = np.log10(df_train['N'].values.astype(float)).reshape(-1, 1)
X_test_np = df_test[feature_columns].values.astype(float)
y_test_np = np.log10(df_test['N'].values.astype(float)).reshape(-1, 1)

sigma_a_idx = 15



# Crucial: Scale features so the neural network branches stabilize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# Convert to JAX arrays
X_train = jnp.array(X_train_scaled)
X_test = jnp.array(X_test_scaled)
y_train = jnp.array(y_train_np)
y_test = jnp.array(y_test_np)

# We also need unscaled raw sigma_a values for the Basquin physical equation
# since the law expects real physical units, not scaled deviations.
sigma_a_train_raw = jnp.array(X_train_np[:, [sigma_a_idx]])
sigma_a_test_raw = jnp.array(X_test_np[:, [sigma_a_idx]])

# ============================================================
# 2. PHYSICS-INFORMED NEURAL NETWORK WITH TRAINABLE BASQUIN
# ============================================================
class PhysicsInformedNN(nn.Module):
    @nn.compact
    def __call__(self, x, sigma_a_raw):
        # --- Data-Driven Branch (Standard Neural Network) ---
        nn_out = nn.Dense(features=20)(x)
        nn_out = nn.relu(nn_out)
        nn_out = nn.Dense(features=20)(nn_out)
        nn_out = nn.relu(nn_out)
        nn_out = nn.Dense(features=48)(nn_out)
        nn_out = nn.relu(nn_out)
        nn_out = nn.Dense(features=48)(nn_out)
        nn_out = nn.relu(nn_out)
        nn_out = nn.Dense(features=20)(nn_out)
        nn_out = nn.relu(nn_out)
        nn_pred = nn.Dense(features=1)(nn_out)

        # --- Physics Branch (Trainable Basquin Parameters) ---
        # Initialize log10(sigma_f') around a realistic value (e.g., log10(1000) ~ 3.0)
        log10_sigma_f = self.param('log10_sigma_f', lambda key: jnp.array([3.0]))
        # Initialize b (fatigue exponent is usually negative, e.g., -0.1)
        b = self.param('b', lambda key: jnp.array([-0.1]))

        # Prevent division by zero or log of negative numbers
        sigma_a_safe = jnp.maximum(sigma_a_raw, 1e-6)
        
        # Basquin equation rearranged for log10(N)
        # log10(N) = (1/b) * (log10(sigma_a) - log10(sigma_f'))
        basquin_pred = (1.0 / (b + 1e-8)) * (jnp.log10(sigma_a_safe) - log10_sigma_f)

        # --- Hybrid Combination ---
        # The network learns a residual correction on top of the physical Basquin baseline
        final_pred = basquin_pred + nn_pred
        return final_pred

# Initialize model and parameters
key = jax.random.PRNGKey(0)
model = PhysicsInformedNN()
params = model.init(key, X_train[0:1], sigma_a_train_raw[0:1])

# ============================================================
# 3. LOSS AND TRAINING STEP DEFINITIONS
# ============================================================
def mse_loss(params, x, sigma_a_raw, y):
    predictions = model.apply(params, x, sigma_a_raw)
    return jnp.mean((predictions - y) ** 2)

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, sigma_a_raw, y):
    loss, grads = jax.value_and_grad(mse_loss)(params, x, sigma_a_raw, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ============================================================
# 4. TRAINING LOOP
# ============================================================
num_epochs = 500  # Bumped to 500 for better convergence
train_loss_history = []
test_loss_history = []
epoch_history = []

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    params, opt_state, train_loss = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train)
    
    if epoch % 10 == 0 or epoch == 1:
        test_loss = mse_loss(params, X_test, sigma_a_test_raw, y_test)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        epoch_history.append(epoch)


        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

# Inspect the learned physical constants
learned_sigma_f = 10 ** float(params['params']['log10_sigma_f'][0])
learned_b = float(params['params']['b'][0])
print(f"\nLearned Basquin Parameters -> sigma_f': {learned_sigma_f:.2f}, b: {learned_b:.4f}")

# ============================================================
# 5. MEASURE PERFORMANCE & PLOT PREDICTIONS
# ============================================================
# Generate predictions on test set
y_pred = np.array(model.apply(params, X_test, sigma_a_test_raw))

r2 = r2_score(y_test_np, y_pred)
mae = mean_absolute_error(y_test_np, y_pred)

print(f"\nPINN Model R² Score: {r2:.4f}")
print(f"PINN Model MAE:      {mae:.4f}")

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, y_pred, alpha=0.7, color='teal', label='PINN Predictions')
plt.plot([y_train_np.min(), y_test_np.max()], [y_train_np.min(), y_test_np.max()], 'r--', label='Perfect Fit')
plt.xlabel('True log10(N)') 
plt.ylabel('Predicted log10(N)')
plt.title(f'PINN (Basquin Law): True vs Predicted (R² = {r2:.3f})')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()


plt.figure(figsize=(10, 6))

# Use epoch_history directly for the X-axis mapping
plt.plot(epoch_history, train_loss_history, label='Train Loss', color='blue', linewidth=2)
plt.plot(epoch_history, test_loss_history, label='Test Loss', color='orange', linestyle='--', linewidth=2)

plt.title('PINN Loss Over Epochs', fontsize=14, fontweight='bold')      
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=12)
plt.show()

