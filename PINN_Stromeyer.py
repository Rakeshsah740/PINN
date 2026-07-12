"""
Physics-Informed Neural Network for Stromeyer's Law

"""
import pickle
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
df = pd.read_excel("V3.xlsx")

feature_columns = [
    'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
    'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'sigma_a', 'Is that Runouts?'
]

X = df[feature_columns].values.astype(float)
y = np.log10(df['N'].values.astype(float)).reshape(-1, 1)

# Keep track of where sigma_a 
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
    def __call__(self, x):
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
        

        log10_sigma_f = self.param('log10_sigma_f', lambda key: jnp.array([3.0]))
        # Initialize b (fatigue exponent is usually negative, e.g., -0.1)
        b = self.param('b', lambda key: jnp.array([-0.1]))

        return nn_pred
    
class EnduranceNeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        nn_out = nn.Dense(features=2)(x)
        return nn_out
      

def compute_sigma_endurance(x, params_endurance, scaler_X_endurance):
    x_endurance = jnp.delete(x, jnp.array([15, 16]), axis=1)
    x_endurance_scaled = scaler_X_endurance.transform(x_endurance)
    x_endurance_pred = model_endurance.apply(params_endurance, jnp.array(x_endurance_scaled))
    x_endurance_pred_unscaled = scaler_y_endurance.inverse_transform(np.array(x_endurance_pred))
    return x_endurance_pred_unscaled[:, 1].reshape(-1, 1)

def mse_loss(params, x, y):
    predictions = model.apply(params, x)
    return jnp.mean((predictions - y) ** 2)

def physics_loss(params,x, sigma_endurance_pred, sigma_a_raw):
    nn_pred= model.apply(params, x)
    log10_sigma_f = params['params']['log10_sigma_f'][0]
    b = params['params']['b'][0]    

    # Avoid log of zero/negative
    delta_sigma = jnp.maximum(sigma_a_raw - sigma_endurance_pred, 1e-8)
   

    stromeyer_pred = ((jnp.log10(delta_sigma) - log10_sigma_f) / b - jnp.log10(2.0))

    return jnp.mean((nn_pred - stromeyer_pred) ** 2)

def total_loss(params, x, sigma_a_raw, y, lambda_phys,sigma_endurance_pred):
    mse = mse_loss(params, x, y)
    phys = physics_loss(params, x, sigma_endurance_pred, sigma_a_raw)
    return lambda_phys *mse +   phys

   
# Initialize model and parameters                                            
model = PhysicsInformedNN()
optimizer = optax.adam(learning_rate=0.001)


print("Loading model assets...")
with open("endurance_pinn_model.pkl", 'rb') as f:
    assets = pickle.load(f)

params_endurance = assets['model_params']
scaler_X_endurance = assets['scaler_X']
scaler_y_endurance = assets['scaler_y']

model_endurance = EnduranceNeuralNetwork()  

@jax.jit
def train_step(params, opt_state, x, sigma_a_raw, y, lambda_phys,sigma_endurance_pred):
    loss, grads = jax.value_and_grad(total_loss)(params, x, sigma_a_raw, y, lambda_phys, sigma_endurance_pred)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


lambda_values = [ 1.0, 10.0]  # Physics loss weight sweep
#lambda_values = [0.0001, 0.001, 0.01]
   
num_epochs = 1000 
r2_scores = []
mae_scores = []
nn_loss_history = []
phys_loss_history = []
total_loss_history = []
training_history = []



sigma_endurance_train = compute_sigma_endurance(X_train_np, params_endurance, scaler_X_endurance)
sigma_endurance_test = compute_sigma_endurance(X_test_np, params_endurance, scaler_X_endurance)

print("Starting Parameter Tuning Sweep...")
for lamb in lambda_values:
    print(f"Training with lambda_phys = {lamb}...")
    
    # CRUCIAL: Re-initialize parameters so each run starts fresh
    key = jax.random.PRNGKey(42)
    params = model.init(key, X_train[0:1])
    opt_state = optimizer.init(params)
    
    # Internal training loop for current lambda
    for epoch in range(1, num_epochs + 1):
        params, opt_state, _ = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train, lamb, sigma_endurance_train)
        phys_loss_value = physics_loss(params, X_train, sigma_endurance_train, sigma_a_train_raw)
        nn_loss_value = mse_loss(params, X_train, y_train)
        total_loss_value = total_loss(params, X_train, sigma_a_train_raw, y_train, lamb, sigma_endurance_train)
        nn_loss_history.append(nn_loss_value)
        phys_loss_history.append(phys_loss_value)
        total_loss_history.append(total_loss_value)

        log10_sigma_f = float(params['params']['log10_sigma_f'][0])

    b = float(params['params']['b'][0])

    # Delta sigma statistics
    raw_delta = sigma_a_train_raw - sigma_endurance_train

    """
    training_history.append({
        "Lambda": lamb,
        "Epoch": epoch,
        "NN_Loss": float(nn_loss_value),
        "Physics_Loss": float(phys_loss_value),
        "Total_Loss": float(total_loss_value),
        "Raw_Delta_Min": float(jnp.min(raw_delta)),
        "Raw_Delta_Max": float(jnp.max(raw_delta)),
        "Negative_Delta_Fraction": float(jnp.mean(raw_delta <= 0)),
        "Log10_Sigma_f": log10_sigma_f,
        "Sigma_f": float(10**log10_sigma_f),
        "b": b
    })
    """


    # Evaluate performance on test set after training completes
    y_pred= model.apply(params, X_test)
    y_pred = np.array(y_pred)

    current_r2 = r2_score(y_test_np, y_pred)
    current_mae = mean_absolute_error(y_test_np, y_pred)
    
    # Save the scores to our tracking lists
    r2_scores.append(current_r2)
    mae_scores.append(current_mae)
    print(f"-> Final R²: {current_r2:.4f} | MAE: {current_mae:.4f}\n")
    print(f" Final NN Loss: {nn_loss_value:.6f} | Physics Loss: {phys_loss_value:.6f} | Total Loss: {total_loss_value:.6f}\n")    
    log10_sigma_f = params['params']['log10_sigma_f'][0]
    b = params['params']['b'][0]
    
    print(f" Learned log10(sigma_f'): {log10_sigma_f:.4f} | Learned b: {b:.4f}\n")

    """ Save training history to Excel 
    history_df = pd.DataFrame(training_history)

    history_df.to_excel(
        "PINN_training_history.xlsx",
        index=False
    )

    print("Training history saved to PINN_training_history.xlsx")
    """

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
