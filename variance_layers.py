import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


nqubits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]
layers = [1, 5, 10, 15, 20, 25, 40, 60, 80, 100]
trials = 50

color = [
    "royalblue",
    "firebrick",
    "olive",
    "black",
    "magenta",
    "darkgoldenrod",
    "gray",
    "salmon",
    "seagreen",
    "rosybrown",
    "darkkhaki",
    "teal",
    "hotpink",
    "peru",
    "palevioletred",
    "limegreen",
]

variance_qubit = []
for k in range(len(nqubits)):
    variance_layer = []
    print(f"Number of qubits {nqubits[k]}")
    for j in range(len(layers)):
        print(f"Number of layers {layers[j]}")
        df_combined = pd.DataFrame()
        for i in range(trials):
            print(f"Trial {i+1}")
            file_name = (
                f"Q{nqubits[k]}/gradients/gradients_Q{nqubits[k]}_L{layers[j]}_M{i}.npz"
            )
            loaded_data = np.load(file_name, allow_pickle=True)
            grads = loaded_data["grads"]
            gradients_df = pd.DataFrame([grads])
            df_combined = pd.concat([df_combined, gradients_df], ignore_index=True)

        variances = df_combined.var()
        variances_array = variances.values
        mean_variance_1_layer = tf.reduce_mean(variances_array)
        print(f"Gradiente medio: {mean_variance_1_layer}")
        print("=" * 60)
        variance_layer.append(mean_variance_1_layer)
    variance_qubit.append(variance_layer)

print(f"valori sulle y {variance_qubit}")
print(f"valori sulle x {layers}")

for i in range(len(nqubits)):
    plt.plot(
        layers,
        variance_qubit[i],
        marker="o",
        linestyle="-",
        color=color[i],
        label=f"Qubits {nqubits[i]}",
    )

plt.yscale("log")
plt.xlabel("Layers")
plt.ylabel("Variance")
plt.title("Variance vs Layers")

plt.grid(True)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("variance_layers_outside.pdf")
plt.close()

print("Variance vs Layers")
