"""
Physics-Informed Neural Network for Stromeyer's Law where the graphs starts from the endurance prediction

Formlation of Physics loss using comparision of Stromeyer's equation and nn predicition.
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
        nn_pred = nn.Dense(features=1)(nn_out)
        

        log10_sigma_f = self.param('log10_sigma_f', lambda key: jnp.array([3.0]))
        # Initialize b (fatigue exponent is usually negative, e.g., -0.1)
        raw_b = self.param('raw_b', lambda key: jnp.array([-0.1]))
        b = -jax.nn.softplus(raw_b) - 0.02   # always negative, never near 0

        return nn_pred, log10_sigma_f, b
    
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
    predictions,_,_ = model.apply(params, x)
    return jnp.mean((predictions.reshape(-1, 1) - y.reshape(-1, 1)) ** 2)

def physics_loss(params, model, x, sigma_endurance_pred, sigma_a_raw,y_std):
    nn_pred, log10_sigma_f, b= model.apply(params, x)
     

    delta_sigma = jnp.maximum(sigma_a_raw - sigma_endurance_pred, 1e-8)
    stromeyer_pred = ((jnp.log10(delta_sigma) - log10_sigma_f) / b - jnp.log10(2.0))

    residual = (nn_pred - stromeyer_pred)/y_std
   
    return jnp.mean(residual**2)



def total_loss(params, model, x, sigma_a_raw, y, lambda_phys,sigma_endurance_pred,y_std):
    mse = mse_loss(params,model, x, y)
    phys = physics_loss(params,model, x, sigma_endurance_pred, sigma_a_raw,y_std)
    return   mse +  lambda_phys * phys


def train_pinn_stromeyer(data_path, num_epochs=900, lr=0.001, lamb=1e-5, random_state=42):
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
    
    # Use later to normalize residual on physics loss
    y_std = float(np.std(y_train_np))
  

   
    # Initialize model and parameters                                            
    model = PhysicsInformedNN()
    optimizer = optax.adam(learning_rate=lr)


    print("Loading Endurance model assets...")
    with open("endurance_pinn_model.pkl", 'rb') as f:
        assets = pickle.load(f)

    params_endurance = assets['model_params']
    scaler_X_endurance = assets['scaler_X']
    scaler_y_endurance = assets['scaler_y']

    model_endurance = EnduranceNeuralNetwork()  

    @jax.jit
    def train_step(params, opt_state, x, sigma_a_raw, y, lamb,sigma_endurance_pred):
        loss, grads = jax.value_and_grad(
            lambda p: total_loss(p, model, x, sigma_a_raw, y, lamb, sigma_endurance_pred,y_std)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    

    r2_history = []
    mae_scores = []

    training_history = []
    b_history = []
    log10_sigma_f_history = []
    nn_loss_history = []
    phys_loss_history = []
    total_loss_history = []
    epoch_history = []
    train_loss_history = []
    test_loss_history = []
        

    sigma_endurance_train = compute_sigma_endurance(X_train_np,model_endurance, params_endurance, scaler_X_endurance,scaler_y_endurance)



        
    # Initialize 
    key = jax.random.PRNGKey(42)
    params = model.init(key, X_train[0:1])
    opt_state = optimizer.init(params)


    pbar = tqdm(range(num_epochs), desc="Training Epochs")
    for epoch in pbar:
        params, opt_state, train_loss = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train, lamb, sigma_endurance_train)
        #params, opt_state, train_loss = train_step(params, opt_state, X_train, sigma_a_train_raw, y_train)
        
        if epoch % 20 == 0 or epoch == 1:
            test_loss = mse_loss(params, model, X_test, y_test)
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            phys_loss_value = physics_loss(params,model, X_train, sigma_endurance_train, sigma_a_train_raw,y_std)
            nn_loss_value = mse_loss(params,model, X_train, y_train)
            total_loss_value = total_loss(params,model, X_train, sigma_a_train_raw, y_train, lamb, sigma_endurance_train,y_std)
            nn_loss_history.append(nn_loss_value)
            phys_loss_history.append(phys_loss_value)
            total_loss_history.append(total_loss_value)
            epoch_history.append(epoch)

            # Evaluate on test set periodically
            y_pred_test,_,_ = model.apply(params, X_test)
            y_pred_test = np.array(y_pred_test)
            current_r2 = r2_score(y_test_np, y_pred_test)
            current_mae = mean_absolute_error(y_test_np, y_pred_test)
            r2_history.append(current_r2)
            

            log10_sigma_f_history.append(float(params['params']['log10_sigma_f'][0]))
            b_val = float(-jax.nn.softplus(params['params']['raw_b'][0]) - 0.02)
            b_history.append(b_val)
            pbar.set_postfix({
                "R^2": f"{r2_history[-1]:.6f}",
                "NN_Loss": f"{nn_loss_history[-1]:.6f}",
                "Physics_Loss": f"{phys_loss_history[-1]:.6f}",
                "Total_Loss": f"{total_loss_history[-1]:.6f}"})



    # Evaluate performance on test set after training completes
    y_pred ,_,_ = model.apply(params, X_test)
    y_pred = np.array(y_pred)
    final_r2 = r2_score(y_test_np, y_pred)
    final_mae = mean_absolute_error(y_test_np, y_pred)

    print(f"\n{'='*50}")
    print(f"Final Results (λ = {lamb})")
    print(f"{'='*50}")
    print(f"R² Score: {final_r2:.4f}")
    print(f"MAE: {final_mae:.4f}")

    # Pack histories into a clean dictionary
    history = {
        "epoch": epoch_history,
        "train_loss": train_loss_history,
        "test_loss": test_loss_history,
        "nn_loss": nn_loss_history,
        "phys_loss": phys_loss_history,
        "total_loss": total_loss_history,
        "r2": r2_history,
        "mae": mae_scores,
        "b": b_history,
        "log10_sigma_f_history": log10_sigma_f_history
    }

    metrics = {"final_r2": final_r2, "final_mae": final_mae}

    return params, model, scaler, metrics, history


       
if __name__ == "__main__":
    # Call your pipeline function with custom parameters
    trained_params, model, scaler, metrics, history = train_pinn_stromeyer(
        data_path="V4_without_z1.xlsx",
        num_epochs=900,
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

    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['b'], color='orange', linestyle='--', linewidth=2)
    plt.title('b', fontsize=14, fontweight='bold')
    plt.legend()

    
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['log10_sigma_f_history'], label='sigma_f', color='orange', linestyle='--', linewidth=2)
    plt.title('Training log10_sigma_f', fontsize=14, fontweight='bold')
    plt.legend()

    """ 

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
    log10_N_pred1,_,_ = model.apply(trained_params, X_alloy_jax1)

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
    log10_N_pred4,_,_ = model.apply(trained_params, X_alloy_jax4)

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


    
    # Z=6
    alloy6 = jnp.array([[
	    87.9112, 10.8900, 0.1820, 0.0188, 0.6180, 0.3100,
   	    0.0016, 0.0024, 0.0098, 0.0010, 0.0005, 0.0547,
        0, 0, 1
        ,0
        ]])
        
    
    predicted_endurance6 = compute_sigma_endurance(alloy6,model_endurance, params_endurance, scaler_X_endurance,scaler_y_endurance)
    print(f"Predicted sigma_endurance for the (z=6): {predicted_endurance6[0][0]:.4f}")
        

    # Define your target stress range 
    stress_range6 = np.linspace(predicted_endurance6[0][0], 183, 20)

    alloy_data6 = []
    for sigma in stress_range4:
        row = [
	    87.9112, 10.8900, 0.1820, 0.0188, 0.6180, 0.3100,
   	    0.0016, 0.0024, 0.0098, 0.0010, 0.0005, 0.0547,
        0, 0, 1,
        sigma                                      
        ]
        alloy_data6.append(row)


    # Actual test points for the alloy (for comparison)
    # Data
    sigma_a_stress6 = np.array([
    157.9, 139.1, 129.5, 108.7, 175.1, 130.0, 87.2, 97.0,
    173.5, 155.1, 132.3, 93.2, 185.4, 76.1,
    90.0, 70.0, 90.0, 90.0, 80.0, 80.0, 75.0, 70.0, 90.0, 85.0,
    75.0, 90.0, 70.0, 85.0, 85.0, 85.0, 75.0, 90.0, 80.0

	])


    N_stress6 = np.array([
     215, 746, 3099, 28209, 82, 1596, 3184417, 94365,
    57, 215, 746, 122050, 11, 10000000, 
    41467, 10000000, 118785, 65953, 379510, 528162, 123278, 10000000, 
    77271, 40492, 4331637, 43224, 10000000, 143503, 252227, 69910, 
    10000000, 189875, 1310118
    ])

    # Convert to a NumPy array for preprocessing
    alloy_data_np6 = np.array(alloy_data6)

    # 1. Scale the data using the exact same scaler from training
    alloy_data_scaled6 = scaler.transform(alloy_data_np6)


    # 3. Convert to JAX arrays
    X_alloy_jax6 = jnp.array(alloy_data_scaled6)

    # 4. Generate predictions (Outputs will be in log10(N))
    log10_N_pred6,_,_ = model.apply(trained_params, X_alloy_jax6)

    # 5. Convert back to raw physical cycles: N = 10^(log10(N))
    N_pred_physical6 = 10 ** np.array(log10_N_pred6)

    # ============================================================
    # 6 PLOT THE GENERATED S-N CURVE
    # ============================================================
    plt.figure(figsize=(8, 6))
    plt.plot(N_pred_physical6, stress_range6, color='crimson', linewidth=2.5, label='PINN Predicted S-N Curve')
    plt.hlines(y=predicted_endurance6[0][0], xmin=N_pred_physical6.max(), xmax=3e7, color='blue', linestyle='--', label=f'Predicted Endurance Limit: {predicted_endurance6[0][0]:.2f} MPa')
    plt.scatter(N_stress6, sigma_a_stress6, color='teal', s=60, alpha=0.7, label='Experimental Data Points')
    plt.xscale('log') # S-N curves are traditionally viewed on a log scale for cycles
    plt.xlabel('Cycles to Failure (N)', fontsize=12)
    plt.ylabel('Stress Amplitude (MPa)', fontsize=12)
    plt.title('Interpolation - Predicted Fatigue Life for Z = 6', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)


    # Z = 8
    alloy = jnp.array([[
            88.0132, 10.80, 0.1850, 0.0131, 0.6140, 0.3080, # Elements (Al to Mg) ; Z = 4,
            0.0011, 0.0018, 0.0067, 0.0012, 0.0005, 0.0554, # Elements (Cr to Ti)
            1, 0, 0 ,                                       # T5=1, T6=0, T7=0
            0                                            # Placeholders for sigma and runout flag

        ]])
        
    predicted_endurance = compute_sigma_endurance(alloy,model_endurance, params_endurance, scaler_X_endurance,scaler_y_endurance)
    
    print(f"Predicted sigma_endurance for the alloy (z=8): {predicted_endurance[0][0]:.4f}")


    # Define your target stress range
    stress_range = np.linspace(predicted_endurance[0][0], 285, 10)

    alloy_data = []
    for sigma in stress_range:
        row = [
            88.0132, 10.80, 0.1850, 0.0131, 0.6140, 0.3080, # Elements (Al to Mg) ; Z = 4,
            0.0011, 0.0018, 0.0067, 0.0012, 0.0005, 0.0554, # Elements (Cr to Ti)
            1, 0, 0,                                       # T5=0, T6=1, T7=0
            sigma                                          # The changing stress level
        ]
        alloy_data.append(row)


    # Actual test points for the alloy (for comparison)
    # Data
    sigma_a_stress = np.array([
        209.6, 184.3, 135.9, 135.4, 111.2,
        250.9, 179.5, 210.3, 221.5, 87.1,
        89.6, 234.1, 114.3,
        100.0, 100.0, 70.0, 100.0, 80.0,
        90.0, 90.0, 120.0, 90.0, 120.0,
        120.0, 80.0, 70.0, 80.0,
        100.0, 90.0
    ])


    N_stress = np.array([
        65, 162, 11212, 24259, 115780,
        114, 2405, 3949, 712, 168729,
        675778, 11, 210658,
        121476, 50161, 2710250, 471938, 10000000,
        10000000, 130237, 61654, 171024, 706714,
        22026, 729292, 10000000, 10000000, 
        116454, 262829
    ])
        
        

    # Convert to a NumPy array for preprocessing
    alloy_data_np = np.array(alloy_data)

    # 1. Scale the data using the exact same scaler from training
    alloy_data_scaled = scaler.transform(alloy_data_np)


    # 3. Convert to JAX arrays
    X_alloy_jax = jnp.array(alloy_data_scaled)
   

    # 4. Generate predictions (Outputs will be in log10(N))
    log10_N_pred,_,_ = model.apply(trained_params, X_alloy_jax)

    # 5. Convert back to raw physical cycles: N = 10^(log10(N))
    N_pred_physical = 10 ** np.array(log10_N_pred)

    # ============================================================
    # 6 PLOT THE GENERATED S-N CURVE
    # ============================================================
    plt.figure(figsize=(8, 6))
    plt.plot(N_pred_physical, stress_range, color='crimson', linewidth=2.5, label='PINN Predicted S-N Curve')
    plt.hlines(y=predicted_endurance[0][0], xmin=N_pred_physical.max(), xmax=3e7, color='blue', linestyle='--', label=f'Predicted Endurance Limit: {predicted_endurance[0][0]:.2f} MPa')
    plt.scatter(N_stress, sigma_a_stress, color='teal', s=60, alpha=0.7, label='Experimental Data Points')
    plt.xscale('log') # S-N curves are traditionally viewed on a log scale for cycles
    plt.xlabel('Cycles to Failure (N)', fontsize=12)
    plt.ylabel('Stress Amplitude (MPa)', fontsize=12)
    plt.title('Predicted Fatigue Life for Z = 8', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)

    plt.show()