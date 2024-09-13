import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


qubits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
layers = [1, 2, 3, 4]
option = ["loss", "accuracy"]
locality = str(input("L or G: "))
architecture = str(input("Embedding or Full-Architecture: "))

# Initialize arrays to hold all qubit accuracy data for training and validation
q_train = 0
q_val = 0
square_diff_acc = np.zeros((len(qubits), len(layers)))
square_diff_loss = np.zeros((len(qubits), len(layers)))

for qubit in qubits:
    print(f"Qubit {qubit}")
    for layer in layers:
        print(f"Layer {layer}")

        file_train_acc = (
            f"{architecture}/statistics/history_train_accuracy_q{qubit}-l{layer}.npy"
        )
        file_val_acc = (
            f"{architecture}/statistics/history_val_accuracy_q{qubit}-l{layer}.npy"
        )

        file_train_loss = (
            f"{architecture}/statistics/history_train_loss_q{qubit}-l{layer}.npy"
        )
        file_val_loss = (
            f"{architecture}/statistics/history_val_loss_q{qubit}-l{layer}.npy"
        )

        if os.path.exists(file_train_loss):
            data = np.load(file_train_loss)
            q_train_loss = data
        else:
            print(
                f"There is no file named {architecture}/statistics/history_train_loss_q{qubit}-l{layer}.npy"
            )

        if os.path.exists(file_val_loss):
            data = np.load(file_val_loss)
            q_val_loss = data
        else:
            print(
                f"There is no file named {architecture}/statistics/history_val_loss_q{qubit}-l{layer}.npy"
            )

        if os.path.exists(file_train_acc):
            data = np.load(file_train_acc)
            q_train_acc = data
        else:
            print(
                f"There is no file named {architecture}/statistics/history_train_accuracy_q{qubit}-l{layer}.npy"
            )

        if os.path.exists(file_val_acc):
            data = np.load(file_val_acc)
            q_val_acc = data
        else:
            print(
                f"There is no file named {architecture}/statistics/history_val_accuracy_q{qubit}-l{layer}.npy"
            )

        square_diff_loss[qubit - 1, layer - 1] = np.sum(
            (q_train_loss - q_val_loss) ** 2
        ) / len(q_train_loss)
        square_diff_acc[qubit - 1, layer - 1] = np.sum(
            (q_train_acc - q_val_acc) ** 2
        ) / len(q_train_acc)


# Plotting the heatmap for training
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.heatmap(
    square_diff_loss,
    xticklabels=layers,
    yticklabels=qubits,
    cmap="viridis",
    annot=True,
    fmt=".6f",
    vmin=0.00001,
    vmax=0.02,
    cbar=False,
)

plt.title(f"MSE Loss")
plt.xlabel("Layers")
plt.ylabel("Qubits")
plt.tight_layout()


plt.subplot(1, 2, 2)
sns.heatmap(
    square_diff_acc,
    xticklabels=layers,
    yticklabels=qubits,
    cmap="viridis",
    annot=True,
    fmt=".6f",
    vmin=0.00001,
    vmax=0.02,
)

plt.title(f"MSE Accuracy")
plt.xlabel("Layers")
plt.yticks([])
plt.tight_layout()
plt.savefig(f"MSE-{architecture}-4x4-{locality}.pdf")
