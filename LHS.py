from scipy.stats import qmc
import pandas as pd
import jax.numpy as jnp

df = pd.read_excel("V5.xlsx")

feature_columns = [
    'Z', 'sigma_a'
]

X = df[feature_columns].values.astype(float)

# Store generated data
synthetic_data = []

for i in range (1,12):
    x_z_i = X[X[:,0] == i]

    # Skip if alloy does not exist
    if x_z_i.shape[0] == 0:                             # rows
        print("Z =", i, "not found, skipping")
        continue

    sigma_min = x_z_i[:,1].min()
    sigma_max = x_z_i[:,1].max()

    n_samples = 100

    # 1-dimensional LHS
    sampler = qmc.LatinHypercube(d=1, seed=42)

    # Generate samples in [0,1]
    lhs = sampler.random(n_samples)

    # Scale to stress range
    sigma_lhs = qmc.scale(lhs,
                        l_bounds=[sigma_min],
                        u_bounds=[sigma_max])

    sigma_lhs = sigma_lhs.flatten()

    # Save Z and generated sigma_a
    for sigma in sigma_lhs:
        synthetic_data.append([i, sigma])

# Convert to DataFrame
synthetic_df = pd.DataFrame(
    synthetic_data,
    columns=["Z", "sigma_a"]
)

# Save to Excel
synthetic_df.to_excel(
    "excel/Synthetic_sigma_LHS.xlsx",
    index=False
)

print("Synthetic data saved successfully")



