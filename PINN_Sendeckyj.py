"""
Physics-Informed Neural Network USING Sendeckyj model.
Here the Neural network learns the KV parameters involved the model (asymptotic fatigue strength, horizontal shift parameter, 
fatigue strength coefficient, fatigue exponent) and also predict the logN.

Formlation of Physics loss using residual.
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
        nn_pred = nn.Dense(features=5)(nn_out)

        logN = nn_pred[:,0]

        # Adding constrained to the output
        sigma_infinity = 50.0 + 150.0 * nn.softplus(nn_pred[:,1])   # Floor of 50 MPa and growly upto infinity
        N0 = 0.1 + 99.9 * nn.softplus(nn_pred[:,2])                 # Floor of 0.1 and growly upto infinity
        K = 200.0 + 1000.0 * nn.softplus(nn_pred[:,3])              # Floor of 200 MPa and growly upto infinity
        m = 0.01 + 0.29 *nn.sigmoid(nn_pred[:,4])                  # Floor of 0.01, range up to 0.3

        return logN, sigma_infinity, N0, K, m

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
      

def compute_sigma_endurance(x, model_endurance, params_endurance, scaler_X_endurance,scaler_y_endurance):
    x_endurance = jnp.delete(x, jnp.array([15]), axis=1)
    x_endurance_scaled = scaler_X_endurance.transform(x_endurance)
    x_endurance_pred = model_endurance.apply(params_endurance, jnp.array(x_endurance_scaled))
    x_endurance_pred_unscaled = scaler_y_endurance.inverse_transform(np.array(x_endurance_pred))
    return x_endurance_pred_unscaled[:, 1].reshape(-1, 1)
    


def mse_loss(params, model, x, y):
    logN_pred, _, _, _, _ = model.apply(params, x)
    return jnp.mean((logN_pred - y[:,0]) ** 2)

  
def physics_loss(params, model, x, sigma_a_raw,sigma_a_std):
    logN, sigma_infinity, N0, K, m = model.apply(params, x)
    N = 10**logN
    log10_N_plus_N0 = jnp.log10(N + N0)
    ratio = K * (10.0 ** (-m * log10_N_plus_N0))
    phy_calc = sigma_infinity + ratio
    residual = (sigma_a_raw - phy_calc)/sigma_a_std
    return jnp.mean(residual**2)

    

def total_loss(params, model, x, sigma_a_raw, y, lambda_phys,sigma_a_std):
    mse = mse_loss(params, model, x, y)
    phys = physics_loss(params, model, x,sigma_a_raw,sigma_a_std)
    return   mse +  lambda_phys * phys


def train_pinn_sendeckyj(data_path, num_epochs=650, lr=0.001, lamb=1e-4, random_state=42):

    # ============================================================
    # 1. DATA LOADING & PREPARATION
    # ============================================================
    df = pd.read_excel(data_path)

    feature_columns = [
        'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
        'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'sigma_a'
    ]

    X = df[feature_columns].values.astype(float)
    y = np.log10(df['N'].values.astype(float)).reshape(-1, 1)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=random_state
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

    # Use later for the normalization
    sigma_a_std = float(np.std(X_train_np[:, sigma_a_idx]))





    @jax.jit
    def train_step(params, opt_state, x, sigma_a_raw, y):
        loss, grads = jax.value_and_grad(
            lambda p: total_loss(p, model, x, sigma_a_raw, y, lamb,sigma_a_std)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    r2_history = []
    mae_scores = []
    nn_loss_history = []
    phys_loss_history = []
    total_loss_history = []
    epoch_history = []
    train_loss_history = []
    test_loss_history = []
 
        
    # Initialize model and parameters                                            
    model = PhysicsInformedNN()
    optimizer = optax.adam(learning_rate=lr)


    # Initialize parameters 
    key = jax.random.PRNGKey(42)
    params = model.init(key, X_train[0:1])
    opt_state = optimizer.init(params)


    pbar = tqdm(range(num_epochs), desc="Training Epochs")

    for epoch in pbar:
        params, opt_state, train_loss = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train)
        if epoch % 20 == 0 or epoch == 1:
            test_loss = mse_loss(params, model, X_test, y_test)
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            phys_loss_value = physics_loss(params,model, X_train, sigma_a_train_raw,sigma_a_std)
            nn_loss_value = mse_loss(params,model, X_train, y_train)
            total_loss_value = total_loss(params,model, X_train, sigma_a_train_raw, y_train, lamb,sigma_a_std)
            nn_loss_history.append(nn_loss_value)
            phys_loss_history.append(phys_loss_value)
            total_loss_history.append(total_loss_value)
            epoch_history.append(epoch)

            # Evaluate on test set periodically
            y_pred_test, _, _, _, _  = model.apply(params, X_test)
            y_pred_test = np.array(y_pred_test)
            current_r2 = r2_score(y_test_np, y_pred_test)
            current_mae = mean_absolute_error(y_test_np, y_pred_test)
            r2_history.append(current_r2)
            mae_scores.append(current_mae)

            pbar.set_postfix({
                "R^2": f"{r2_history[-1]:.6f}",
                "NN_Loss": f"{nn_loss_history[-1]:.6f}",
                "Physics_Loss": f"{phys_loss_history[-1]:.6f}",
                "Total_Loss": f"{total_loss_history[-1]:.6f}"})

    # Evaluate performance on test set after training completes
    y_pred, sigma_infinity, N0, K, m = model.apply(params, X_test)
    y_pred = np.array(y_pred)
    final_r2 = r2_score(y_test_np, y_pred)
    final_mae = mean_absolute_error(y_test_np, y_pred)

    print(f"\n{'='*50}")
    print(f"Final Results (λ = {lamb})")
    print(f"{'='*50}")
    print(f"R² Score: {final_r2:.4f}")
    print(f"MAE: {final_mae:.4f}")

    print(f"Learned Parameters:")
    print(f"sigma_infinity min: {np.min(np.array(sigma_infinity)):.4f}, max: {np.max(np.array(sigma_infinity)):.4f}")
    print(f"N0 min: {np.min(np.array(N0)):.4f}, max: {np.max(np.array(N0)):.4f}")
    print(f"K min: {np.min(np.array(K)):.4f}, max: {np.max(np.array(K)):.4f}")
    print(f"m min: {np.min(np.array(m)):.4f}, max: {np.max(np.array(m)):.4f}")

    # Pack histories into a clean dictionary
    history = {
        "epoch": epoch_history,
        "train_loss": train_loss_history,
        "test_loss": test_loss_history,
        "nn_loss": nn_loss_history,
        "phys_loss": phys_loss_history,
        "total_loss": total_loss_history,
        "r2": r2_history,
        "mae": mae_scores
    }

    metrics = {"final_r2": final_r2, "final_mae": final_mae}

    return params, model, scaler, metrics, history

if __name__ == "__main__":
    # Call your pipeline function with custom parameters
    trained_params, model, scaler, metrics, history = train_pinn_sendeckyj(
        data_path="V4_including_synt.xlsx",
        num_epochs=550,
        lr=0.001,
        lamb=0.1
    )

    # PLOT PERFORMANCE
    # ============================================================
    plt.figure(figsize=(10, 5))

    # Plot R² Score profile
    plt.plot(history['epoch'], history['r2'], marker='o', color='dodgerblue', linewidth=2.5, markersize=8)
    plt.xscale('linear') 
    plt.title(r'($R^2$) Over Epochs ($\lambda$)', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Test $R^2$ Score', fontsize=12)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()


        
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['nn_loss'], label='NN Loss', color='green', linestyle='-.', linewidth=3)
    plt.plot(history['epoch'], history['phys_loss'], label='Physics Loss', color='red', linestyle=':', linewidth=2)
    plt.plot(history['epoch'], history['total_loss'], label='Total Loss', color='purple', linestyle='-', linewidth=2)
    plt.yscale('log')  # Log scale for better visibility of loss trends
    plt.title('PINN Loss Components Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE) - log scale', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['epoch'], history['test_loss'], label='Test Loss', color='orange', linestyle='--', linewidth=2)
    plt.yscale('log')  # Log scale for better visibility of loss trends
    plt.title('Training loss vs Test loss', fontsize=14, fontweight='bold')
    plt.legend()


 

    
    # 5. PREDICTING FOR A SPECIFIC ALLOY (SYNTHETIC S-N CURVE)
    # ============================================================
    with open("endurance_pinn_model.pkl", 'rb') as f:
        assets = pickle.load(f)

    params_endurance = assets['model_params']
    scaler_X_endurance = assets['scaler_X']
    scaler_y_endurance = assets['scaler_y']
    model_endurance = EnduranceNeuralNetwork()

    # Z=1
    alloy1 = jnp.array([[
            92.1128, 7.300, 0.1040, 0.0082, 0.0166, 0.3300, # Elements (Al to Mg) ; Z = 1,
            0.0016, 0.0025, 0.0053, 0.0005, 0.0005, 0.1180, # Elements (Cr to Ti)
            0, 1, 0                                        # T5=0, T6=1, T7=0
            ,0
        ]])
        
    
    predicted_endurance1 = compute_sigma_endurance(alloy1,model_endurance, params_endurance, scaler_X_endurance,scaler_y_endurance)
    print(f"Predicted sigma_endurance for the (z=1): {predicted_endurance1[0][0]:.4f}")
        

    # Define your target stress range 
    stress_range1 = np.linspace(predicted_endurance1[0][0], 290, 20)

    alloy_data1 = []
    for sigma in stress_range1:
        row = [
            92.1128, 7.300, 0.1040, 0.0082, 0.0166, 0.3300, # Elements (Al to Mg) ; Z = 1,
            0.0016, 0.0025, 0.0053, 0.0005, 0.0005, 0.1180, # Elements (Cr to Ti)
            0, 1, 0,                                        # T5=0, T6=1, T7=0
            sigma                                            # The changing stress level
        ]
        alloy_data1.append(row)


    # Actual test points for the alloy (for comparison)
    # Data
    sigma_a_stress1 = np.array([274.3, 275.9, 261.8,
    279.6, 214.5, 237.1, 270.6, 258.4,
    259.4, 226.6, 221.1, 192.0, 206.4, 98.9,
    115.3,  91.8, 72.9,
    72.9, 85.0, 67.6, 67.6, 72.9,
    62.6, 78.6, 67.6, 67.6, 58.0,
    62.6

    ])


    N_stress1 = np.array([
    15, 35, 179, 
    258, 278, 284, 519, 788, 
    1370, 1487, 6257, 8658, 15610,
    216183,
    384791,  896141, 1139222,
    1528914, 1925880, 2744484, 3509964, 3903503,
    4415340, 8384682, 10000084, 10000089, 10000103,
    10000106
    ])

    # Convert to a NumPy array for preprocessing
    alloy_data_np1 = np.array(alloy_data1)

    # 1. Scale the data using the exact same scaler from training
    alloy_data_scaled1 = scaler.transform(alloy_data_np1)


    # 3. Convert to JAX arrays
    X_alloy_jax1 = jnp.array(alloy_data_scaled1)

    # 4. Generate predictions (Outputs will be in log10(N))
    log10_N_pred1, _, _, _, _= model.apply(trained_params, X_alloy_jax1)

    # 5. Convert back to raw physical cycles: N = 10^(log10(N))
    N_pred_physical1 = 10 ** np.array(log10_N_pred1)

    # ============================================================
    # 6 PLOT THE GENERATED S-N CURVE
    # ============================================================
    plt.figure(figsize=(8, 6))
    plt.plot(N_pred_physical1, stress_range1, color='crimson', linewidth=2.5, label='PINN Predicted S-N Curve')
    plt.hlines(y=predicted_endurance1[0][0], xmin=N_pred_physical1.max(), xmax=3e7, color='blue', linestyle='--', label=f'Predicted Endurance Limit: {predicted_endurance1[0][0]:.2f} MPa')
    plt.scatter(N_stress1, sigma_a_stress1, color='teal', s=60, alpha=0.7, label='Experimental Data Points')
    plt.xscale('log') # S-N curves are traditionally viewed on a log scale for cycles
    plt.xlabel('Cycles to Failure (N)', fontsize=12)
    plt.ylabel('Stress Amplitude (MPa)', fontsize=12)
    plt.title('Interpolation - Predicted Fatigue Life for Z = 1', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    


    # Z=4
    alloy4 = jnp.array([[
            92.3155, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480, # Elements (Al to Mg) ; Z = 4,
            0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280, # Elements (Cr to Ti)
            0, 1, 0                                        # T5=0, T6=1, T7=0
            ,0
        ]])
        
    
    predicted_endurance4 = compute_sigma_endurance(alloy4,model_endurance, params_endurance, scaler_X_endurance,scaler_y_endurance)
    print(f"Predicted sigma_endurance for the (z=4): {predicted_endurance4[0][0]:.4f}")
        

    # Define your target stress range (e.g., from 40 MPa to 250 MPa)
    stress_range4 = np.linspace(predicted_endurance4[0][0], 306, 20)

    alloy_data4 = []
    for sigma in stress_range4:
        row = [
            92.3155, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480, # Elements (Al to Mg) ; Z = 4,
            0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280, # Elements (Cr to Ti)
            0, 1, 0,                                        # T5=0, T6=1, T7=0
            sigma                                            # The changing stress level
        ]
        alloy_data4.append(row)


    # Actual test points for the alloy (for comparison)
    # Data
    sigma_a_stress4 = np.array([311.3, 297.0, 296.3, 291.5, 273.7, 283.1, 
        300.3, 285.8, 264.7, 215.8,
        238.9, 144.7, 191, 
        168.7, 156.1, 133.8, 98.4, 91.3, 114.7, 106.2, 
        84.2, 72.2, 72.2, 78.2, 78.2, 123.9, 78.2, 78.2])


    N_stress4 = np.array([
        18, 83, 174, 339, 412, 421, 474, 1279, 4546, 
        21202,14009, 14823, 27433, 66863, 91377, 230870, 
        355297, 372672, 422572, 427476, 719897, 2027306, 3126784, 1975940, 1818832, 3152047, 
        10000000, 10000000
    ])

    # Convert to a NumPy array for preprocessing
    alloy_data_np4 = np.array(alloy_data4)

    # 1. Scale the data using the exact same scaler from training
    alloy_data_scaled4 = scaler.transform(alloy_data_np4)


    # 3. Convert to JAX arrays
    X_alloy_jax4 = jnp.array(alloy_data_scaled4)

    # 4. Generate predictions (Outputs will be in log10(N))
    log10_N_pred4, _, _, _, _ = model.apply(trained_params, X_alloy_jax4)

    # 5. Convert back to raw physical cycles: N = 10^(log10(N))
    N_pred_physical4 = 10 ** np.array(log10_N_pred4)

    # ============================================================
    # 6 PLOT THE GENERATED S-N CURVE
    # ============================================================
    plt.figure(figsize=(8, 6))
    plt.plot(N_pred_physical4, stress_range4, color='crimson', linewidth=2.5, label='PINN Predicted S-N Curve')
    plt.hlines(y=predicted_endurance4[0][0], xmin=N_pred_physical4.max(), xmax=3e7, color='blue', linestyle='--', label=f'Predicted Endurance Limit: {predicted_endurance4[0][0]:.2f} MPa')
    plt.scatter(N_stress4, sigma_a_stress4, color='teal', s=60, alpha=0.7, label='Experimental Data Points')
    plt.xscale('log') # S-N curves are traditionally viewed on a log scale for cycles
    plt.xlabel('Cycles to Failure (N)', fontsize=12)
    plt.ylabel('Stress Amplitude (MPa)', fontsize=12)
    plt.title('Predicted Fatigue Life for Z = 4', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)



    plt.show()
