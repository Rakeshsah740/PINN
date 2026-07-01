import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'sigma_a'
]

X = df[feature_columns].values.astype(float)
y = np.log10(df['N'].values.astype(float)).reshape(-1, 1)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to JAX arrays
X_train = jnp.array(X_train_np)
X_test = jnp.array(X_test_np)
y_train = jnp.array(y_train_np)
y_test = jnp.array(y_test_np)

# ============================================================
# 2. MODEL DEFINITION
# ============================================================
class CustomNeuralNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=20)(x)
        x = nn.relu(x)
        x = nn.Dense(features=20)(x)
        x = nn.relu(x)
        x = nn.Dense(features=20)(x)
        x = nn.relu(x)
        x = nn.Dense(features=20)(x)
        x = nn.relu(x)
        x = nn.Dense(features=20)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# Initialize model and parameters
key = jax.random.PRNGKey(0)
model = CustomNeuralNet()
params = model.init(key, X_train[0:1])

# ============================================================
# 3. LOSS AND TRAINING STEP DEFINITIONS
# ============================================================
# Define Mean Squared Error Loss
def mse_loss(params, x, y):
    predictions = model.apply(params, x)
    return jnp.mean((predictions - y) ** 2)

# Set up the Adam Optimizer
optimizer = optax.adam(learning_rate=0.0075)
opt_state = optimizer.init(params)

# JIT-compiled training step function for speed
@jax.jit
def train_step(params, opt_state, x, y):
    # Calculates both the loss value and the gradients
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ============================================================
# 4. TRAINING LOOP
# ============================================================
num_epochs = 100
train_loss_history = []
test_loss_history = []

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    # Perform update step and get training loss
    params, opt_state, train_loss = train_step(params, opt_state, X_train, y_train)
    
    # Calculate test loss (no gradients/updates needed)
    test_loss = mse_loss(params, X_test, y_test)
    
    # Store history (converting JAX device arrays to standard floats)
    train_loss_history.append(float(train_loss))
    test_loss_history.append(float(test_loss))
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

# ============================================================
# 5. PLOTTING THE LOSS
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss', color='blue', linewidth=2)
plt.plot(range(1, num_epochs + 1), test_loss_history, label='Test Loss', color='orange', linestyle='--', linewidth=2)

plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=12)


# ============================================================
# 6. MEASURE PERFORMANCE & PLOT PREDICTIONS
# ============================================================

# 1. Generate predictions using the trained parameters
# We use jax.device_get() or np.array() to bring the data back to standard NumPy
y_pred = np.array(model.apply(params, X_test))

# 2. Calculate Scikit-Learn Metrics
from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y_test_np, y_pred)
mae = mean_absolute_error(y_test_np, y_pred)

print(f"Flax Model R² Score: {r2:.4f}")
print(f"Flax Model MAE:      {mae:.4f}")

# 3. True vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, y_pred, alpha=0.7, color='purple', label='Predictions')

# Perfect prediction reference line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')

plt.xlabel('True log10(N)', fontsize=12) 
plt.ylabel('Predicted log10(N)', fontsize=12)
plt.title(f'Flax NN: True vs Predicted log10(N) (R² = {r2:.3f})', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()