import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


options = ["accuracy"]
qubit = int(input("How many qubits: "))
layer = int(input("How many layers: "))

qubits = list(range(1, qubit + 1))
layers = list(range(1, layer + 1))

# Initialize arrays to hold all qubit accuracy data for training and validation
q_train = np.full((len(qubits), len(layers)), np.nan)
q_val = np.full((len(qubits), len(layers)), np.nan)

for qubit in qubits:
    print(f"Qubit {qubit}")
    for layer in layers:
        print(f"Layer {layer}")
        for option in options:
            print(f"Option {option}")
            file_train = (
                f"Q{qubit}/statistics/history_train_{option}_q{qubit}-l{layer}.npy"
            )
            file_val = f"Q{qubit}/statistics/history_val_{option}_q{qubit}-l{layer}.npy"

            if os.path.exists(file_train):
                data_train = np.load(file_train)
                q_train[qubit - 1, layer - 1] = data_train[-1]
            else:
                print(
                    f"There is no file named statistics/history_train_{option}_q{qubit}-l{layer}.npy"
                )

            if os.path.exists(file_val):
                data_val = np.load(file_val)
                q_val[qubit - 1, layer - 1] = data_val[-1]
            else:
                print(
                    f"There is no file named statistics/history_val_{option}_q{qubit}-l{layer}.npy"
                )


# Determine the common color scale range
vmin = min(np.nanmin(q_train), np.nanmin(q_val))
vmax = max(np.nanmax(q_train), np.nanmax(q_val))

# Plotting the heatmap for training and validation side by side with a common color scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

dim_ticks = 12
dim_label = 12
dim_title = 26

sns.heatmap(
    q_train,
    xticklabels=layers,
    yticklabels=qubits,
    cmap="viridis",
    annot=True,
    fmt=".3f",
    mask=np.isnan(q_train),  # Masking NaN values
    vmin=vmin,
    vmax=vmax,
    ax=ax1,
    annot_kws={"size": dim_label},
    cbar=False,
)
ax1.set_title(f"Training Accuracy", fontsize=dim_title)
ax1.set_xlabel("Layers", fontsize=dim_label)
ax1.set_ylabel("Qubits", fontsize=dim_label)
ax1.tick_params(axis="both", which="major", labelsize=dim_ticks)


sns.heatmap(
    q_val,
    xticklabels=layers,
    yticklabels=qubits,
    cmap="viridis",
    annot=True,
    fmt=".3f",
    mask=np.isnan(q_val),  # Masking NaN values
    vmin=vmin,  # Common minimum value
    vmax=vmax,  # Common maximum value
    cbar=True,  # Enable the color bar for the second heatmap
    ax=ax2,
    annot_kws={"size": dim_label},
    cbar_ax=fig.add_axes([0.92, 0.3, 0.02, 0.4]),
)
ax2.set_title(f"Validation Accuracy", fontsize=dim_title)
ax2.set_yticks([])
ax2.set_xlabel("Layers", fontsize=dim_label)
ax2.tick_params(axis="both", which="major", labelsize=dim_ticks)

# Adjust layout to make room for the color bar

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.subplots_adjust(hspace=0.2, wspace=0.3)

plt.savefig(f"Heatmap-.pdf")
