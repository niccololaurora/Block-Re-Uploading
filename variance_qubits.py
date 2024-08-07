import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


nqubits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
layers = [1, 5, 10, 15, 20, 25, 40, 60, 80, 100]
trials = 50


# Carica i gradienti per i modelli M0, M1, ..., M9
variance_layer = []
for k in range(len(layers)):
    variance_qubit = []
    for j in range(len(nqubits)):
        df_combined = pd.DataFrame()
        for i in range(trials):
            file_name = (
                f"Q{nqubits[j]}/gradients/gradients_Q{nqubits[j]}_L{layers[k]}_M{i}.npz"
            )
            loaded_data = np.load(file_name, allow_pickle=True)
            grads = loaded_data["grads"]
            gradients_df = pd.DataFrame([grads])
            df_combined = pd.concat([df_combined, gradients_df], ignore_index=True)

        variances = df_combined.var()
        variances_array = variances.values
        mean = tf.reduce_mean(variances_array)
        variance_qubit.append(mean)
    variance_layer.append(variance_qubit)


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
    "stateblue",
    "hotpink",
    "peru",
    "palevioletred",
    "limegreen",
]
for i in range(len(layers)):
    plt.plot(
        nqubits,
        variance_layer[i],
        marker="o",
        linestyle="-",
        color=color[i],
        label=f"Layers {layers[i]}",
    )

plt.yscale("log")
plt.xlabel("Qubits")
plt.ylabel("Variance")
plt.title("Variance vs Number of Qubits")

plt.grid(True)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("variance_qubits.pdf")
plt.close()

print("Variance vs Qubits")
