"""

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
import pickle


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

def mse_loss(params, x, y):
    predictions = model.apply(params, x)
    return jnp.mean((predictions - y) ** 2)

df = pd.read_excel("V3.xlsx")

feature_columns = [
        'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
        'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'Is that Runouts?'
    ]

X_data = df[feature_columns].values.astype(float)
y_raw_N = df['N'].values.astype(float).reshape(-1, 1)
y_raw_sigma = df['sigma_a'].values.astype(float).reshape(-1, 1)

mask = X_data[:, -1] == 1

X = X_data[mask, :-1]  # Exclude the last column (Is that Runouts?) for features
y = np.hstack([np.log10(y_raw_N[mask]), y_raw_sigma[mask]])


X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


# Crucial: Scale features so the neural network branches stabilize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# Create a scaler for y just like X
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_np)
y_test_scaled = y_scaler.transform(y_test_np)


# Convert to JAX arrays
X_train = jnp.array(X_train_scaled)
X_test = jnp.array(X_test_scaled)
y_train = jnp.array(y_train_scaled)
y_test = jnp.array(y_test_scaled)




# Initialize model and parameters
key = jax.random.PRNGKey(0)                                                 # Always exact same initialization of weights for reproducibility. This is crucial for debugging and consistent results.
model = EnduranceNeuralNetwork()
params = model.init(key, X_train[0:1])              # Initialize parameters using a single sample to get the correct shapes. This is a common practice in JAX/Flax to ensure the model's parameters are properly initialized before training.

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y)   # Compute the loss and its gradients with respect to the parameters
    updates, opt_state = optimizer.update(grads, opt_state)                 # Get the parameter updates from the optimizer based on the computed gradients
    params = optax.apply_updates(params, updates)                           # Apply the updates to the parameters to get the new parameters for the next iteration
    return params, opt_state, loss

num_epochs = 1000
train_loss_history = []
test_loss_history = []
epoch_history = []

print("Starting Endurance training...")
for epoch in range(num_epochs):
    params, opt_state, train_loss = train_step(params, opt_state, X_train, y_train)

    if epoch % 10 == 0:
        y_pred_test = model.apply(params, X_test)
        test_loss = jnp.mean((y_pred_test - y_test) ** 2)
        
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        epoch_history.append(epoch + 1)

        print(f"Epoch {epoch + 1}/{num_epochs} ── Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

# Create a dictionary holding everything needed for future deployment
saved_assets = {
    'model_params': params,
    'scaler_X': scaler,
    'scaler_y': y_scaler
}

# Save everything into a single binary file
model_filename = "endurance_pinn_model.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(saved_assets, f)
       

y_pred_scaled = model.apply(params, X_test)
y_pred = y_scaler.inverse_transform(np.array(y_pred_scaled))  # Convert back to original scale


r2 = r2_score(y_test_np, y_pred, multioutput='uniform_average')  # R² score for the first output (log10(N))

mae = mean_absolute_error(y_test_np, y_pred, multioutput='raw_values')  # MAE for the first output (log10(N))

print(f"Average R² Score: {r2:.4f}")
print(f"MAE for log10(N): {mae[0]:.4f}")
print(f"MAE for sigma_e: {mae[1]:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_np[:, 0], y_pred[:, 0], alpha=0.7, color='teal', label='NN Predictions')
plt.plot([y_test_np[:, 0].min(), y_test_np[:, 0].max()], [y_test_np[:, 0].min(), y_test_np[:, 0].max()], 'r--', label='Perfect Fit')
plt.xlabel('True log10(N)', fontsize=12)        
plt.ylabel('Predicted log10(N)', fontsize=12)
plt.legend()


plt.figure(figsize=(8, 6))
plt.scatter(y_test_np[:, 1], y_pred[:, 1], alpha=0.7, color='orange', label='NN Predictions')
plt.plot([y_test_np[:, 1].min(), y_test_np[:, 1].max()], [y_test_np[:, 1].min(), y_test_np[:, 1].max()], 'r--', label='Perfect Fit')
plt.xlabel('True sigma_endurance', fontsize=12) 
plt.ylabel('Predicted sigma_endurance', fontsize=12)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(epoch_history, train_loss_history, label='Train Loss', color='blue')
plt.plot(epoch_history, test_loss_history, label='Test Loss', color='orange')       
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Training and Test Loss over Epochs', fontsize=14)    
plt.yscale('log')  # Log scale for better visibility of loss trends
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=12) 
plt.show()


"""
#Z=4
row = [
     92.3155, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480,    # Elements (Al to Mg) ; Z = 4,
     0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280,     # Elements (Cr to Ti)
     0, 1, 0                                             # T5=0, T6=1, T7=0
    ]
"""
    
# Z=8
row = [
        88.0132, 10.80, 0.1850, 0.0131, 0.06140, 0.3080, # Elements (Al to Mg) ; 
        0.0011, 0.0018, 0.0067, 0.0012, 0.0005, 0.0554, # Elements (Cr to Ti)
        1, 0, 0                                            # T5=0, T6=1, T7=0
                                             
        ]
"""
#z=5
row = [
    87.0651, 8.7100, 0.2780, 2.6400, 0.3420, 0.2660,  # Elements (Al to Mg) 
    0.0072, 0.0035, 0.6190, 0.0033, 0.0005, 0.0654,    # Elements (Cr to Ti)
    0, 1, 0                                             # T5=0, T6=1, T7=0
   
]



# z =3
row = [
    84.4233, 9.8200, 0.9660, 3.1700, 0.5220, 0.2830,  # Elements (Al to Mg) 
    0.0390, 0.0533, 0.6110, 0.0560, 0.0070, 0.0494,    # Elements (Cr to Ti)
    1, 0, 0                                             # T5=0, T6=1, T7=0
]
"""
        
X_new = np.array(row).reshape(1, -1)
X_new_scaled = scaler.transform(X_new)  
pred_scaled = model.apply(params, jnp.array(X_new_scaled))
pred = y_scaler.inverse_transform(np.array(pred_scaled))  # Convert back to original scale  
print(f"Predicted log10(N) for the new sample: {pred[0, 0]:.4f}")
print(f"Predicted sigma_endurance for the new sample: {pred[0, 1]:.4f}")
