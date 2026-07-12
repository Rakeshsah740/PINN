"""
Tunning of the physics loss weight (lambda_phys) in a Physics-Informed Neural Network (PINN) for fatigue life prediction.
This script systematically varies the weight of the physics loss term in the total loss function.

"""
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

import jax
print(jax.__version__)
print(jax.devices())

# ============================================================
# 1. DATA LOADING & PREPARATION
# ============================================================
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
        # Initialize log10(sigma_f') 
        log10_sigma_f = self.param('log10_sigma_f', lambda key: jnp.array([3.0]))

        # Initialize b 
        b = self.param('b', lambda key: jnp.array([-0.1]))

        return nn_pred

# Initialize model and parameters
key = jax.random.PRNGKey(0)                                                 # Always exact same initialization of weights for reproducibility. This is crucial for debugging and consistent results.
model = PhysicsInformedNN()
params = model.init(key, X_train[0:1], sigma_a_train_raw[0:1])  

def mse_loss(params, x, sigma_a_raw, y):
    predictions = model.apply(params, x, sigma_a_raw)
    return jnp.mean((predictions - y) ** 2)

def physics_loss(params,x, sigma_a_raw):
    nn_pred = model.apply(params, x, sigma_a_raw)
    log10_sigma_f = params['params']['log10_sigma_f'][0]
    b = params['params']['b'][0]
    basquin_pred = (1.0 / (b + 1e-8)) * (jnp.log10(sigma_a_raw) - log10_sigma_f) - jnp.log10(2)
    return jnp.mean((nn_pred - basquin_pred) ** 2)

def total_loss(params, x, sigma_a_raw, y, lambda_phys):
    mse = mse_loss(params, x, sigma_a_raw, y)
    phys = physics_loss(params, x, sigma_a_raw)
    return mse + lambda_phys * phys

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, sigma_a_raw, y, lambda_phys):
    loss, grads = jax.value_and_grad(total_loss)(params, x, sigma_a_raw, y, lambda_phys)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 10000]
r2_scores = []
mae_scores = []

num_epochs = 400 

print("Starting Parameter Tuning Sweep...")
for lamb in lambda_values:
    print(f"Training with lambda_phys = {lamb}...")
    
    # CRUCIAL: Re-initialize parameters so each run starts fresh
    key = jax.random.PRNGKey(42)
    params = model.init(key, X_train[0:1], sigma_a_train_raw[0:1])
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    
    # Internal training loop for current lambda
    for epoch in range(1, num_epochs + 1):
        params, opt_state, _ = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train, lamb)
        
    # Evaluate performance on test set after training completes
    y_pred = np.array(model.apply(params, X_test, sigma_a_test_raw))
    
    current_r2 = r2_score(y_test_np, y_pred)
    current_mae = mean_absolute_error(y_test_np, y_pred)
    
    # Save the scores to our tracking lists
    r2_scores.append(current_r2)
    mae_scores.append(current_mae)
    print(f"-> Final R²: {current_r2:.4f} | MAE: {current_mae:.4f}\n")

# PLOT PERFORMANCE VS LAMBDA
# ============================================================
plt.figure(figsize=(10, 5))

# Plot R² Score profile
plt.plot(lambda_values, r2_scores, marker='o', color='dodgerblue', linewidth=2.5, markersize=8)

plt.xscale('log') # Changes X-axis to logarithmic intervals (0.001, 0.01, 0.1...)
plt.title('PINN Generalization Capacity ($R^2$) vs. Physics Loss Weight ($\lambda$)', fontsize=13, fontweight='bold')
plt.xlabel('Physics Weight ($\lambda$) - Log Scale', fontsize=12)
plt.ylabel('Test Set $R^2$ Score', fontsize=12)

plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
