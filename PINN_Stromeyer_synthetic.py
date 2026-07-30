"""
Prediction of the cycle for the synthesis stresses.
"""
from csv import excel

from flax import linen as nn
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pickle


from PINN_Stromeyer import train_pinn_stromeyer, compute_sigma_endurance


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
    
      
with open("endurance_pinn_model.pkl", 'rb') as f:
    assets = pickle.load(f)

params_endurance = assets['model_params']
scaler_X_endurance = assets['scaler_X']
scaler_y_endurance = assets['scaler_y']
model_endurance = EnduranceNeuralNetwork()

def compute_sigma_endurance(x, model_endurance, params_endurance, scaler_X_endurance,scaler_y_endurance):
    x_endurance = jnp.delete(x, jnp.array([15]), axis=1)
    x_endurance_scaled = scaler_X_endurance.transform(x_endurance)
    x_endurance_pred = model_endurance.apply(params_endurance, jnp.array(x_endurance_scaled))
    x_endurance_pred_unscaled = scaler_y_endurance.inverse_transform(np.array(x_endurance_pred))
    return x_endurance_pred_unscaled[:, 1].reshape(-1, 1)

if __name__ == "__main__":

    print("Stromeyer Model")
    trained_params_sm, model_sm, scaler_sm, metrics_sm, history_sm = train_pinn_stromeyer(
        data_path="V4.xlsx",
        num_epochs=1200,
        lr=0.001,
        lamb=0.01
    )

    df = pd.read_excel("excel/Synthetic_Prediction_SM_IN.xlsx")

    feature_columns = [
            'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
            'Pb 82', 'Sn 50', 'Ti 22', 'T5 ?', 'T6 ?', 'T7 ?', 'sigma_a'
        ]

    X = df[feature_columns].values.astype(float)

    # Store prediction
    predicted_N = []

    for i in range(0, X.shape[0]):
        x_sample = X[i,:].reshape(1,-1)

        # Scale the data using the exact same scaler from training
        alloy_data_scaled4 = scaler_sm.transform(x_sample)

        # Convert to JAX arrays
        X_alloy_jax = jnp.array(alloy_data_scaled4)

        #Generate predictions (Outputs will be in log10(N))
        log10_N_pred,_,_ = model_sm.apply(trained_params_sm, X_alloy_jax)
    
        #Convert back to raw physical cycles: N = 10^(log10(N))
        N_pred_physical = 10 ** np.array(log10_N_pred)[0,0]

        predicted_N.append([x_sample[0,-1], N_pred_physical])

    # Convert to DataFrame
    synthetic_samples = pd.DataFrame(
        predicted_N ,
        columns=[ "sigma_a", "N"]
    )

    # Save to Excel
    synthetic_samples.to_excel(
        "excel/Synthetic_Prediction_SM_out.xlsx",
        index=False
    )

    print("Synthetic samples saved successfully")