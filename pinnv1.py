"""
PINN Implementation for Fatigue Life Prediction with Trainable Basquin Parameters 
with both losses plotted against each other to visualize the trade-off during training."""

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
# 1. PHYSICS-INFORMED NEURAL NETWORK WITH TRAINABLE BASQUIN
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
    
# ============================================================
# 2. LOSS AND TRAINING STEP DEFINITIONS
# ============================================================
def mse_loss(params, x, sigma_a_raw, y):
    predictions = model.apply(params, x, sigma_a_raw)
    return jnp.mean((predictions - y) ** 2)

def physics_loss(params,x, sigma_a_raw):
    nn_pred = model.apply(params, x, sigma_a_raw)
    log10_sigma_f = params['params']['log10_sigma_f'][0]
    b = params['params']['b'][0]
    basquin_pred = (1.0 / (b + 1e-8)) * (jnp.log10(sigma_a_raw) - log10_sigma_f) - jnp.log10(2)
    return jnp.mean((nn_pred - basquin_pred) ** 2)

def total_loss(params, x, sigma_a_raw, y):
    mse = mse_loss(params, x, sigma_a_raw, y)
    phys = physics_loss(params, x, sigma_a_raw)
    return mse + 0.001* phys  



# ============================================================
# 3. TRAINING LOOP
# ============================================================
if __name__ == '__main__':
    # ============================================================
    # 1. DATA LOADING & PREPARATION
    # ============================================================
    df = pd.read_excel("V3.xlsx")

    feature_columns = [
        'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
        'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'sigma_a', 'Is that Runouts?'
    ]

    X_data = df[feature_columns].values.astype(float)
    y_data = np.log10(df['N'].values.astype(float)).reshape(-1, 1)

    mask = X_data[:, -1] == 0

    X = X_data[mask, :-1]  # Exclude the last column (Is that Runouts?) for features
    y = y_data[mask]

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


    # Initialize model and parameters
    key = jax.random.PRNGKey(0)                                                 # Always exact same initialization of weights for reproducibility. This is crucial for debugging and consistent results.
    model = PhysicsInformedNN()
    params = model.init(key, X_train[0:1], sigma_a_train_raw[0:1])              # Initialize parameters using a single sample to get the correct shapes. This is a common practice in JAX/Flax to ensure the model's parameters are properly initialized before training.

    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, sigma_a_raw, y):
        loss, grads = jax.value_and_grad(total_loss)(params, x, sigma_a_raw, y)   # Compute the loss and its gradients with respect to the parameters
        updates, opt_state = optimizer.update(grads, opt_state)                 # Get the parameter updates from the optimizer based on the computed gradients
        params = optax.apply_updates(params, updates)                           # Apply the updates to the parameters to get the new parameters for the next iteration
        return params, opt_state, loss

    num_epochs = 1000
    train_loss_history = []
    test_loss_history = []
    epoch_history = []
    nn_loss_history = []
    phys_loss_history = []
    total_loss_history = []

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        params, opt_state, train_loss = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train)
        
        if epoch % 10 == 0 or epoch == 1:
            test_loss = mse_loss(params, X_test, sigma_a_test_raw, y_test)
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            epoch_history.append(epoch)
            nn_loss_history.append(mse_loss(params, X_train, sigma_a_train_raw, y_train))
            phys_loss_history.append(physics_loss(params, X_train, sigma_a_train_raw))
            total_loss_history.append(total_loss(params, X_train, sigma_a_train_raw, y_train))


            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | NN Loss: {nn_loss_history[-1]:.4f} | Phys Loss: {phys_loss_history[-1]:.4f}")

    # Inspect the learned physical constants
    learned_sigma_f = 10 ** float(params['params']['log10_sigma_f'][0])
    learned_b = float(params['params']['b'][0])
    print(f"\nLearned Basquin Parameters -> sigma_f': {learned_sigma_f:.2f}, b: {learned_b:.4f}")

    # ============================================================
    # 4. MEASURE PERFORMANCE & PLOT PREDICTIONS
    # ============================================================
    # Generate predictions on test set
    y_pred = np.array(model.apply(params, X_test, sigma_a_test_raw))

    r2 = r2_score(y_test_np, y_pred)
    mae = mean_absolute_error(y_test_np, y_pred)

    print(f"\nPINN Model R² Score: {r2:.4f}")
    print(f"PINN Model MAE:      {mae:.4f}")


    # Scatter Plot
    """     
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_np, y_pred, alpha=0.7, color='teal', label='PINN Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')
    plt.xlabel('True log10(N)') 
    plt.ylabel('Predicted log10(N)')
    plt.title(f'PINN (Basquin Law): True vs Predicted (R² = {r2:.3f})')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()


    plt.figure(figsize=(10, 6))

    # Use epoch_history directly for the X-axis mapping
    plt.plot(epoch_history, train_loss_history, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epoch_history, test_loss_history, label='Test Loss', color='orange', linestyle='--', linewidth=2)
    plt.yscale('log')  # Log scale for better visibility of loss trends

    plt.title('PINN Loss Over Epochs', fontsize=14, fontweight='bold')      
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE) - log scale', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_history, nn_loss_history, label='NN Loss', color='green', linestyle='-.', linewidth=3)
    plt.plot(epoch_history, phys_loss_history, label='Physics Loss', color='red', linestyle=':', linewidth=2)
    plt.plot(epoch_history, total_loss_history, label='Total Loss', color='purple', linestyle='-', linewidth=2)
    plt.yscale('log')  # Log scale for better visibility of loss trends
    plt.title('PINN Loss Components Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE) - log scale', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)


    plt.figure(figsize=(10, 6))
    plt.plot(nn_loss_history,phys_loss_history, label='Physics Loss vs NN Loss',color='magenta', linestyle='-', linewidth=2)
    plt.title('Data Loss vs Physics Loss During Training', fontsize=14, fontweight='bold')
    plt.xlabel('NN Loss (MSE)', fontsize=12)
    plt.ylabel('Physics Loss (MSE)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)    
    """ 



    # ============================================================
    # 5. PREDICTING FOR A SPECIFIC ALLOY (SYNTHETIC S-N CURVE)
    # ============================================================
    
    print("Loading model assets...")
    with open("endurance_pinn_model.pkl", 'rb') as f:
        assets = pickle.load(f)

    params_endurance = assets['model_params']
    scaler_X_endurance = assets['scaler_X']
    scaler_y_endurance = assets['scaler_y']

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
            sigma                                           # The changing stress level
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
    log10_N_pred = model.apply(params, X_alloy_jax, sigma_a_alloy_jax)

    # 5. Convert back to raw physical cycles: N = 10^(log10(N))
    N_pred_physical = 10 ** np.array(log10_N_pred)

    # ============================================================
    # 6. PLOT THE GENERATED S-N CURVE
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

    # Print out a few sample predictions
    print("\n--- Sample Predictions ---")
    for i in [0, len(stress_range)//2, -1]:
        print(f"Stress: {stress_range[i]:.1f} MPa ──► Predicted Life: {N_pred_physical[i][0]:,.0f} cycles")



