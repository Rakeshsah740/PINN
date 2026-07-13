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

from tqdm import tqdm

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

# Keep track of where sigma_a 
sigma_a_idx = 15
sigma_a_train_raw = jnp.array(X_train_np[:, [sigma_a_idx]])
sigma_a_test_raw = jnp.array(X_test_np[:, [sigma_a_idx]])

# We also need the runout indicator for later analysis
runout_idx = 16
runout_train_raw = jnp.array(X_train_np[:, [runout_idx]])
runout_test_raw  = jnp.array(X_test_np[:, [runout_idx]])



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

def physics_loss(params,x, sigma_endurance_pred, sigma_a_raw,runout_flag):
    nn_pred= model.apply(params, x)
    log10_sigma_f = params['params']['log10_sigma_f'][0]
    b = params['params']['b'][0]    

    delta_sigma = jnp.maximum(sigma_a_raw - sigma_endurance_pred, 1e-8)
    stromeyer_pred = ((jnp.log10(delta_sigma) - log10_sigma_f) / b - jnp.log10(2.0))

    mask = 1.0 - runout_flag
    sq_err = (nn_pred - stromeyer_pred) ** 2
    masked_sq_err = sq_err * mask

    n_valid = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(masked_sq_err) / n_valid

def total_loss(params, x, sigma_a_raw, y, lambda_phys,sigma_endurance_pred, runout_flag):
    mse = mse_loss(params, x, y)
    phys = physics_loss(params, x, sigma_endurance_pred, sigma_a_raw, runout_flag)
    return   mse +  lambda_phys * phys

   
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
def train_step(params, opt_state, x, sigma_a_raw, y, lambda_phys,sigma_endurance_pred, runout_flag):
    loss, grads = jax.value_and_grad(total_loss)(params, x, sigma_a_raw, y, lambda_phys, sigma_endurance_pred, runout_flag)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


lambda_values = [0.0001,0.001,0.01, 1.0,10]  # Physics loss weight sweep
#lambda_values = [ 0.1, 1]

pbar = tqdm(lambda_values, desc="Lambda Tuning Sweep")
   
num_epochs = 400
r2_scores = []
mae_scores = []

training_history = []




sigma_endurance_train = compute_sigma_endurance(X_train_np, params_endurance, scaler_X_endurance)
sigma_endurance_test = compute_sigma_endurance(X_test_np, params_endurance, scaler_X_endurance)

print("Starting Parameter Tuning Sweep...")
for lamb in pbar:
    print(f"Training with lambda_phys = {lamb}...")
    
    # CRUCIAL: Re-initialize parameters so each run starts fresh
    key = jax.random.PRNGKey(42)
    params = model.init(key, X_train[0:1])
    opt_state = optimizer.init(params)

    # reset per-lambda tracking here
    b_history = []
    log10_sigma_f_history = []
    nn_loss_history = []
    phys_loss_history = []
    total_loss_history = []
    
    # Internal training loop for current lambda
    for epoch in range(1, num_epochs + 1):
        params, opt_state, _ = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train, lamb, sigma_endurance_train, runout_train_raw)
        phys_loss_value = physics_loss(params, X_train, sigma_endurance_train, sigma_a_train_raw, runout_train_raw)
        nn_loss_value = mse_loss(params, X_train, y_train)
        total_loss_value = total_loss(params, X_train, sigma_a_train_raw, y_train, lamb, sigma_endurance_train, runout_train_raw)
        nn_loss_history.append(nn_loss_value)
        phys_loss_history.append(phys_loss_value)
        total_loss_history.append(total_loss_value)

        log10_sigma_f_history.append(float(params['params']['log10_sigma_f'][0]))
        b_history.append(float(params['params']['b'][0]))
    
    
    print("log10_sigma_f range:", min(log10_sigma_f_history), max(log10_sigma_f_history))
    print("=> sigma_f' range (real units):", 10**min(log10_sigma_f_history), "to", 10**max(log10_sigma_f_history))

    l1 = total_loss(params, X_train, sigma_a_train_raw, y_train, 0.0, sigma_endurance_train, runout_train_raw)
    l2 = total_loss(params, X_train, sigma_a_train_raw, y_train, 100.0, sigma_endurance_train, runout_train_raw)
    print(l1, l2)
        

    

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
    pbar.set_postfix({ 
        "R²": f"{current_r2:.4f}",
        "MAE": f"{current_mae:.4f}",
        "NN_Loss": f"{nn_loss_value:.6f}",
        "Physics_Loss": f"{phys_loss_value:.6f}",
        "Total_Loss": f"{total_loss_value:.6f}"
    })
    

    """
    print(f"-> Final R²: {current_r2:.4f} | MAE: {current_mae:.4f}\n")
    print(f" Final NN Loss: {nn_loss_value:.6f} | Physics Loss: {phys_loss_value:.6f} | Total Loss: {total_loss_value:.6f}\n")    
    log10_sigma_f = params['params']['log10_sigma_f'][0]
    b = params['params']['b'][0]
    
    print(f" Learned log10(sigma_f'): {log10_sigma_f:.4f} | Learned b: {b:.4f}\n")
    
    Save training history to Excel 
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
       

 # 5. PREDICTING FOR A SPECIFIC ALLOY (SYNTHETIC S-N CURVE)
    # ============================================================
    
model_endurance = EnduranceNeuralNetwork()

""" Z=4
alloy = jnp.array([
        92.3155, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480, # Elements (Al to Mg) ; Z = 4,
        0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280, # Elements (Cr to Ti)
        0, 1, 0                                        # T5=0, T6=1, T7=0

    ])
    
alloy_scaled = scaler_X_endurance.transform(alloy.reshape(1, -1))  # Reshape to 2D for scaler
scaled_pred = model_endurance.apply(params_endurance, jnp.array(alloy_scaled))
predicted_endurance = scaler_y_endurance.inverse_transform(np.array(scaled_pred))  # Convert back to original scale
print(f"Predicted log10(N) for the alloy: {predicted_endurance[0][0]:.4f}")
print(f"Predicted sigma_endurance for the alloy: {predicted_endurance[0][1]:.4f}")
    

# Define your target stress range (e.g., from 40 MPa to 250 MPa)
stress_range = np.linspace(predicted_endurance[0][1], 250, 10)

alloy_data = []
for sigma in stress_range:
    row = [
        92.3155, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480, # Elements (Al to Mg) ; Z = 4,
        0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280, # Elements (Cr to Ti)
        0, 1, 0,                                        # T5=0, T6=1, T7=0
        sigma                                           # The changing stress level
    ]
    alloy_data.append(row)


# Actual test points for the alloy (for comparison)
# Data
sigma_a_stress = np.array([238.9, 144.7, 191, 
                    168.7, 156.1, 133.8, 98.4, 91.3, 114.7, 106.2, 
                    84.2, 72.2, 72.2, 78.2, 78.2, 123.9, 78.2, 78.2])


N_stress = np.array([
    14009, 14823, 27433, 66863, 91377, 230870, 
    355297, 372672, 422572, 427476, 719897, 2027306, 3126784, 1975940, 1818832, 3152047, 
    10000000, 10000000
])
     

"""
# Z = 8
alloy = jnp.array([
        88.0132, 10.80, 0.1850, 0.0131, 0.06140, 0.3080, # Elements (Al to Mg) ; Z = 4,
        0.0011, 0.0018, 0.0067, 0.0012, 0.0005, 0.0554, # Elements (Cr to Ti)
        1, 0, 0                                        # T5=0, T6=1, T7=0

    ])
    
alloy_scaled = scaler_X_endurance.transform(alloy.reshape(1, -1))  # Reshape to 2D for scaler
scaled_pred = model_endurance.apply(params_endurance, jnp.array(alloy_scaled))
predicted_endurance = scaler_y_endurance.inverse_transform(np.array(scaled_pred))  # Convert back to original scale
print(f"Predicted log10(N) for the alloy: {predicted_endurance[0][0]:.4f}")
print(f"Predicted sigma_endurance for the alloy: {predicted_endurance[0][1]:.4f}")
    

# Define your target stress range
stress_range = np.linspace(predicted_endurance[0][1], 130, 10)

alloy_data = []
for sigma in stress_range:
    row = [
        88.0132, 10.80, 0.1850, 0.0131, 0.06140, 0.3080, # Elements (Al to Mg) ; Z = 4,
        0.0011, 0.0018, 0.0067, 0.0012, 0.0005, 0.0554, # Elements (Cr to Ti)
        1, 0, 0,                                       # T5=0, T6=1, T7=0
        sigma , 0                                          # The changing stress level
    ]
    alloy_data.append(row)


# Actual test points for the alloy (for comparison)
# Data
sigma_a_stress = np.array([
    100.0, 100.0, 70.0, 100.0, 80.0,
    90.0, 90.0, 120.0, 90.0, 120.0,
    120.0, 80.0, 70.0, 80.0,
    100.0, 90.0
])


N_stress = np.array([
    121476, 50161, 2710250, 471938, 10000000,
    10000000, 130237, 61654, 171024, 706714,
    22026, 729292, 10000000, 10000000, 
    116454, 262829
])
    
# 
    

# Convert to a NumPy array for preprocessing
alloy_data_np = np.array(alloy_data)

# 1. Scale the data using the exact same scaler from training
alloy_data_scaled = scaler.transform(alloy_data_np)

# 2. Extract the unscaled raw stress values for the physics branch
sigma_a_alloy_raw = alloy_data_np[:, [sigma_a_idx]]

# 3. Convert to JAX arrays
X_alloy_jax = jnp.array(alloy_data_scaled)
sigma_a_alloy_jax = jnp.array(sigma_a_alloy_raw)

# 4. Generate predictions (Outputs will be in log10(N))
log10_N_pred = model.apply(params, X_alloy_jax)

# 5. Convert back to raw physical cycles: N = 10^(log10(N))
N_pred_physical = 10 ** np.array(log10_N_pred)

# ============================================================
# 6 PLOT THE GENERATED S-N CURVE
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(N_pred_physical, stress_range, color='crimson', linewidth=2.5, label='PINN Predicted S-N Curve')
plt.hlines(y=predicted_endurance[0][1], xmin=N_pred_physical.max(), xmax=3e7, color='blue', linestyle='--', label=f'Predicted Endurance Limit: {predicted_endurance[0][1]:.2f} MPa')
plt.scatter(N_stress, sigma_a_stress, color='teal', s=60, alpha=0.7, label='Experimental Data Points')
plt.xscale('log') # S-N curves are traditionally viewed on a log scale for cycles
plt.xlabel('Cycles to Failure (N)', fontsize=12)
plt.ylabel('Stress Amplitude (MPa)', fontsize=12)
plt.title('Predicted Fatigue Life for Custom Alloy Composition (T6)', fontsize=14, fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.show()