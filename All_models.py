

from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pickle
import torch

from pinnv1 import train_pinn_basquin
from PINN_Stromeyer import train_pinn_stromeyer, compute_sigma_endurance
from PINN_KV import train_pinn_kv
from PINN_Sendeckyj import train_pinn_sendeckyj


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_path="V4_including_synt_without_z1.xlsx"


    print("Basquin Model")
    trained_params_bq, model_bq, scaler_bq, metrics_bq, history_bq = train_pinn_basquin(
        data_path=data_path,
        num_epochs=1000,
        lr=0.001,
        lamb=1
    )

    
    print("Stromeyer Model")
    trained_params_sm, model_sm, scaler_sm, metrics_sm, history_sm = train_pinn_stromeyer(
        data_path=data_path,
        num_epochs=2000,
        lr=0.001,
        lamb=1
    )

    
    print("Kohout-Vechet Model")
    trained_params_kv, model_kv, scaler_kv, metrics_kv, history_kv = train_pinn_kv(
        data_path=data_path,
        num_epochs=2000,
        lr=0.001,
        lamb=1
    )

    print("Sendeckyj Model")
    trained_params_sd, model_sd, scaler_sd, metrics_sd, history_sd = train_pinn_sendeckyj(
        data_path=data_path,
        num_epochs=1500,
        lr=0.001,
        lamb=1
    )


# 1. Define your alloy configurations in a list of dictionaries
alloys_data = [
        # --- ALLOY 1 (Z = 1) ---
    {
        "z_id": 1,
        "features": [
            92.1128, 7.300, 0.1040, 0.0082, 0.0166, 0.3300, # Elements (Al to Mg) ; Z = 1,
            0.0016, 0.0025, 0.0053, 0.0005, 0.0005, 0.1180, # Elements (Cr to Ti)
            0, 1, 0                                          # T5=0, T6=1, T7=0
    
        ],
        "endurance_base": [
            92.1128, 7.300, 0.1040, 0.0082, 0.0166, 0.3300, # Elements (Al to Mg) ; Z = 1,
            0.0016, 0.0025, 0.0053, 0.0005, 0.0005, 0.1180, # Elements (Cr to Ti)
            0, 1, 0                                        # T5=0, T6=1, T7=0
            ,0
        ],
        "max_stress": 290,  # Custom upper limit for stress range
        "Rm": 283,
        "Rp0.1": 205,
        "k": 5.5,
        "N_endurance": 2e6,
        "endurance": 65,
        "sigma_a_stress": np.array([
                98.9,
                115.3,  91.8, 72.9,
                72.9, 85.0, 67.6, 67.6, 72.9,
                62.6, 78.6, 67.6, 67.6, 58.0,
                62.6
        ]),
        "N_stress": np.array([
                216183,
                384791,  896141, 1139222,
                1528914, 1925880, 2744484, 3509964, 3903503,
                4415340, 8384682, 10000084, 10000089, 10000103,
                10000106
        ]),
            "sigma_a_lcf" : np.array([
                274.3, 275.9, 261.8,
                279.6, 214.5, 237.1, 270.6, 258.4,
                259.4, 226.6, 221.1, 192.0, 206.4
        ]),

        "N_lcf" : np.array([
                15, 35, 179, 
                258, 278, 284, 519, 788, 
                1370, 1487, 6257, 8658, 15610
        ])
    },


    # --- ALLOY 2 (Z = 4) ---
    {
        "z_id": 4,
        "features": [
            92.3155, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480, # Elements (Al to Mg)
            0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280, # Elements (Cr to Ti)
            0, 1, 0                                         # T5=0, T6=1, T7=0
        ],
        "endurance_base": [
            92.3155, 7.0300, 0.1200, 0.0031, 0.0432, 0.3480,
            0.0009, 0.0024, 0.0082, 0.0007, 0.0005, 0.1280,
            0, 1, 0, 100
        ],
        "max_stress": 310,  # Custom upper limit for stress range
        "Rm": 305.6,
        "Rp0.1": 243,
        "k": 4.93,
        "N_endurance": 2e6,
        "endurance": 73,
        "sigma_a_stress": np.array([
            238.9, 144.7, 191, 168.7, 156.1, 133.8, 98.4, 91.3, 114.7, 106.2, 
            84.2, 72.2, 72.2, 78.2, 78.2, 123.9, 78.2, 78.2
        ]),
        "N_stress": np.array([
            14009, 14823, 27433, 66863, 91377, 230870, 355297, 372672, 422572, 
            427476, 719897, 2027306, 3126784, 1975940, 1818832, 3152047, 
            10000000, 10000000
        ]),
            "sigma_a_lcf" : np.array([
            311.3, 297.0, 296.3, 291.5, 273.7, 283.1, 
            300.3, 285.8, 264.7, 215.8
        ]),

        "N_lcf" : np.array([
            18, 83, 174, 339, 412, 421, 474, 1279, 4546, 
            21202
        ])
    },

    # --- ALLOY 3 (Z = 6) ---
        {
            "z_id": 6,
            "features": [
                87.9112, 10.8900, 0.1820, 0.0188, 0.6180, 0.3100,
                0.0016, 0.0024, 0.0098, 0.0010, 0.0005, 0.0547,
                0, 0, 1
            ],
            "endurance_base": [
                87.9112, 10.8900, 0.1820, 0.0188, 0.6180, 0.3100,
                0.0016, 0.0024, 0.0098, 0.0010, 0.0005, 0.0547,
                0, 0, 1,100
            ],
            "max_stress": 190,  # Custom upper limit for stress range
            "Rm": 182,
            "Rp0.1": 101,
            "k": 12.4,
            "N_endurance": 1e6,
            "endurance": 74,
            "sigma_a_stress": np.array([
                90.0, 70.0, 90.0, 90.0, 80.0, 80.0, 75.0, 70.0, 90.0, 85.0,
                75.0, 90.0, 70.0, 85.0, 85.0, 85.0, 75.0, 90.0, 80.0
            ]),
            "N_stress": np.array([
                41467, 10000000, 118785, 65953, 379510, 528162, 123278, 10000000, 
                77271, 40492, 4331637, 43224, 10000000, 143503, 252227, 69910, 
                10000000, 189875, 1310118
            ]),
                "sigma_a_lcf" : np.array([
                157.9, 139.1, 129.5, 108.7, 175.1, 130.0, 87.2, 97.0,
                173.5, 155.1, 132.3, 93.2, 185.4, 76.1
            ]),
    
            "N_lcf" : np.array([
                215, 746, 3099, 28209, 82, 1596, 3184417, 94365,
                57, 215, 746, 122050, 11, 10000000
            ])
        },
    
    # --- ALLOY 4 ---
    {
        "z_id": 8,
        "features": [
            88.0132, 10.80, 0.1850, 0.0131, 0.6140, 0.3080, # Elements (Al to Mg)
            0.0011, 0.0018, 0.0067, 0.0012, 0.0005, 0.0554, # Elements (Cr to Ti)
            1, 0, 0                                         # T5=1, T6=0, T7=0
        ],
        "endurance_base": [
            88.0132, 10.80, 0.1850, 0.0131, 0.6140, 0.3080,
            0.0011, 0.0018, 0.0067, 0.0012, 0.0005, 0.0554,
            1, 0, 0, 100
        ],
        "Rm": 284,
        "Rp0.1": 188,
        "k": 9.7,
        "N_endurance": 1e6,
        "endurance": 78,
        "sigma_a_stress": np.array([
            100.0, 100.0, 70.0, 100.0, 80.0, 90.0, 90.0, 120.0, 
            90.0, 120.0, 120.0, 80.0, 70.0, 80.0, 100.0, 90.0
        ]),
        "N_stress": np.array([
            121476, 50161, 2710250, 471938, 10000000, 10000000, 130237, 61654, 
            171024, 706714, 22026, 729292, 10000000, 10000000, 116454, 262829
        ]),

        "sigma_a_lcf" : np.array([
            209.6, 184.3, 135.9, 135.4, 111.2,
            250.9, 179.5, 210.3, 221.5, 87.1,
            89.6, 234.1, 114.3
        ]),

        "N_lcf" : np.array([
            65, 162, 11212, 24259, 115780,
            114, 2405, 3949, 712, 168729,
            675778, 11, 210658
        ])

    }
]
# Model definitions for fast iteration over predictions & plots
models_info = [
    {
        "name": "Basquin",
        "model": model_bq,
        "params": trained_params_bq,
        "color": "skyblue",
        "history": history_bq,
        "type": "standard",
    },
    {
        "name": "Stromeyer",
        "model": model_sm,
        "params": trained_params_sm,
        "color": "lightcoral",
        "history": history_sm,
        "type": "tuple_3",
    },
    {
        "name": "KV",
        "model": model_kv,
        "params": trained_params_kv,
        "color": "mediumseagreen",
        "history": history_kv,
        "type": "tuple_6",
    },
    {
        "name": "Sendeckyj",
        "model": model_sd,
        "params": trained_params_sd,
        "color": "sandybrown",
        "history": history_sd,
        "type": "tuple_5",
    },
]

# ============================================================
# MAIN LOOP OVER ALL ALLOYS
# ============================================================
for alloy in alloys_data:
    z_id = alloy["z_id"]
    alloy_base_jax = jnp.array([alloy["endurance_base"]])

    # 1. Compute predicted endurance limit
    predicted_endurance = compute_sigma_endurance(
        alloy_base_jax,
        model_endurance,
        params_endurance,
        scaler_X_endurance,
        scaler_y_endurance,
    )[0][0]

    # 2. Generate stress range
    stress_range = np.linspace(predicted_endurance, alloy["Rm"], 10)

    # 3. Construct input feature rows for each stress level
    alloy_data = [alloy["features"] + [sigma] for sigma in stress_range]
    alloy_data_np = np.array(alloy_data)

    # 4. Scale inputs and convert to JAX
    alloy_data_scaled = scaler_bq.transform(alloy_data_np)
    X_alloy_jax = jnp.array(alloy_data_scaled)

    # 5. Figure 2: R² Score over Epochs
    plt.figure(figsize=(10, 5))
    for m in models_info:
        plt.plot(
            m["history"]["epoch"],
            m["history"]["r2"],
            marker="o",
            label=m["name"],
            color=m["color"],
            linewidth=2.5,
            markersize=8,
        )

    plt.xscale("linear")
    plt.title(
        rf"($R^2$) Over Epochs ($\lambda$) — Z = {z_id}",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Test $R^2$ Score", fontsize=12)
    plt.ylim(-0.5, 1)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    
    # 6. Figure 1: S-N Curve for the current alloy
    plt.figure(figsize=(8, 6))

    for m in models_info:
        # Generate predictions dynamically based on return signature
        res = m["model"].apply(m["params"], X_alloy_jax)
        log10_N_pred = res[0] if isinstance(res, tuple) else res
        N_pred_physical = (10 ** np.array(log10_N_pred)).flatten()

        """
        # Factor of 2 scatter band (N/2 to 2N)
        N_scatter_low = N_pred_physical / 2.0
        N_scatter_high = N_pred_physical * 2.0

        # Factor of 3 scatter band (N/3 to 3N)
        N_scatter_2_low = N_pred_physical / 3.0
        N_scatter_2_high = N_pred_physical * 3.0

        # Fill factor 3 band (fainter)
        plt.fill_betweenx(
            stress_range,
            N_scatter_2_low,
            N_scatter_2_high,
            color=m["color"],
            alpha=0.08
        )

        # Fill factor 2 band (clearer)
        plt.fill_betweenx(
            stress_range,
            N_scatter_low,
            N_scatter_high,
            color=m["color"],
            alpha=0.18
        )
        """

        # Plot predicted S-N curve
        plt.plot(
            N_pred_physical,
            stress_range,
            label=m["name"],
            color=m["color"],
            linewidth=2.5,
        )
    
        # Plot horizontal endurance limit line
        plt.hlines(
            y=predicted_endurance,
            xmin=N_pred_physical.max(),
            xmax=3e7,
            color=m["color"],
            linestyle="--",
        )
    

    # Scatter experimental points
    plt.scatter(
        alloy["N_stress"],
        alloy["sigma_a_stress"],
        color="blue",
        s=60,
        alpha=0.7,
        label="Experimental Data Points",
    )

    plt.scatter(
            alloy["N_lcf"],
            alloy["sigma_a_lcf"],
            color="green",
            s=60,
            alpha=0.7,
            label="Experimental Data Points",
        )

    # Calculate Basquin constant C using the endurance point
    C = alloy["endurance"] * (alloy["N_endurance"] ** (1 / alloy["k"]))
    N_Rp01 = (C / alloy["Rp0.1"]) ** alloy["k"]
    N_basquin = np.logspace(np.log10(alloy["N_endurance"]), np.log10(N_Rp01), 200)
    sigma_basquin = C * (N_basquin ** (-1/alloy["k"]))
    plt.plot(N_basquin, sigma_basquin, color='black', linewidth=2,label='Experimental Curve')
    plt.plot([10, N_Rp01], [alloy["Rm"], alloy["Rp0.1"]], color='black', linewidth=2)  
    plt.hlines(y=alloy["endurance"], xmin=alloy["N_endurance"], xmax= 3e7, color='black', linewidth=2)


    plt.xscale("log")
    plt.xlabel("Cycles to Failure (N)", fontsize=12)
    plt.ylabel("Stress Amplitude (MPa)", fontsize=12)
    plt.title(
        f"Predicted Fatigue Life for Z = {z_id}", fontsize=14, fontweight="bold"
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.show()
