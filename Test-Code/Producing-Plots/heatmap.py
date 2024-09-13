import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


options = ["accuracy"]
architecture = str(input("Embedding or Full-Architecture: "))
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
            file_train = f"{architecture}/statistics/history_train_{option}_q{qubit}-l{layer}.npy"
            file_val = (
                f"{architecture}/statistics/history_val_{option}_q{qubit}-l{layer}.npy"
            )

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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
    cbar=False,
)
ax1.set_title(f"Training Accuracy")
ax1.set_xlabel("Layers")
ax1.set_ylabel("Qubits")


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
    cbar_ax=fig.add_axes([0.92, 0.3, 0.02, 0.4]),
)
ax2.set_title(f"Validation Accuracy")
ax2.set_yticks([])
ax2.set_xlabel("Layers")

# Adjust layout to make room for the color bar
plt.subplots_adjust(right=0.9)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(f"Heatmap-4x4-{architecture}-G.pdf")
