import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


df_combined = pd.DataFrame()
params_list = []
nqubits = [1, 2, 4, 6, 8, 16]


# Carica i gradienti per i modelli M0, M1, ..., M9
variance_qubit = []
for j in range(len(nqubits)):
    for i in range(2):
        file_name = f"trainability/gradients_Q{nqubits[j]}_M{i}.npz"
        loaded_data = np.load(file_name, allow_pickle=True)
        grads = loaded_data["grads"]
        gradients_df = pd.DataFrame([grads[15:25]])
        print(grads[15:25])
        # print(gradients_df)
        df_combined = pd.concat([df_combined, gradients_df], ignore_index=True)

    print(df_combined)
    print(df_combined.var())
    variances = df_combined.var()
    variances_array = variances.values
    mean = tf.reduce_mean(variances_array)
    variance_qubit.append(mean)


plt.plot(
    nqubits, variance_qubit, marker="o", linestyle="-", color="b", label="Variance"
)

# Add labels and title
plt.xlabel("Number of Qubits (nqubits)")
plt.ylabel("Variance (variance_qubit)")
plt.title("Variance vs Number of Qubits")

# Display the plot
plt.grid(True)
plt.legend()
plt.show()
