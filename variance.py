import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


df_combined = pd.DataFrame()
params_list = []
nqubits = [1, 2, 4, 6, 8, 16]
layers = [40, 60, 80, 100]


# Carica i gradienti per i modelli M0, M1, ..., M9
variance_layer = []
for k in range(len(layers)):
    variance_qubit = []
    for j in range(len(nqubits)):
        for i in range(10):
            file_name = f"trainability/gradients_Q{nqubits[j]}_L{layers[k]}_M{i}.npz"
            loaded_data = np.load(file_name, allow_pickle=True)
            grads = loaded_data["grads"]
            gradients_df = pd.DataFrame([grads[15:25]])
            df_combined = pd.concat([df_combined, gradients_df], ignore_index=True)

        variances = df_combined.var()
        variances_array = variances.values
        mean = tf.reduce_mean(variances_array)
        variance_qubit.append(mean)
    variance_layer.append(variance_qubit)

color = ["b", "r", "g", "black", "magenta", "purple"]
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
plt.legend()
plt.show()
