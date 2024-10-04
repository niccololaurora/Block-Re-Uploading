import numpy as np
import matplotlib.pyplot as plt

# Lista di file che contengono le accuracies di training per ogni seed
seeds = np.arange(0, 10, 1)  # Da seed 0 a 9
qubits = [1, 2]
layers = [1, 2, 3, 4]

# Dizionario per contenere i file per ciascun qubit
file_lists_loss_train = {q: [] for q in qubits}
file_lists_acc_train = {q: [] for q in qubits}
file_lists_loss_val = {q: [] for q in qubits}
file_lists_acc_val = {q: [] for q in qubits}

for l in layers:
    for s in seeds:
        for q in qubits:
            file_lists_loss_train[q] += [
                f"history_train_loss_q{q}-l{l}_lr0.001_S{s}.npy"
            ]
            file_lists_loss_val[q] += [f"history_val_loss_q{q}-l{l}_lr0.001_S{s}.npy"]
            file_lists_acc_train[q] += [
                f"history_train_accuracy_q{q}-l{l}_lr0.001_S{s}.npy"
            ]
            file_lists_acc_val[q] += [
                f"history_val_accuracy_q{q}-l{l}_lr0.001_S{s}.npy"
            ]


for q in qubits:
    # Loss
    all_losses_train = []
    all_losses_val = []
    for file in file_lists_loss_train[q]:
        loss = np.load(file)
        all_losses_train.append(loss)

    for file in file_lists_loss_val[q]:
        loss = np.load(file)
        all_losses_val.append(loss)

    # Accuracy
    all_acc_train = []
    all_acc_val = []
    for file in file_lists_acc_train[q]:
        loss = np.load(file)
        all_acc_train.append(loss)

    for file in file_lists_acc_val[q]:
        loss = np.load(file)
        all_acc_val.append(loss)

    # Converti in array numpy per calcolare statistiche
    all_losses_train = np.array(all_losses_train)
    all_losses_val = np.array(all_losses_val)
    all_acc_train = np.array(all_acc_train)
    all_acc_val = np.array(all_acc_val)

    # Calcola la media e la deviazione standard per ogni epoca
    mean_loss_train = np.mean(all_losses_train, axis=0)
    std_loss_train = np.std(all_losses_train, axis=0)
    mean_loss_val = np.mean(all_losses_val, axis=0)
    std_loss_val = np.std(all_losses_val, axis=0)
    mean_acc_train = np.mean(all_acc_train, axis=0)
    std_acc_train = np.std(all_acc_train, axis=0)
    mean_acc_val = np.mean(all_acc_val, axis=0)
    std_acc_val = np.std(all_acc_val, axis=0)

    print(f"Accuracy Training {mean_acc_train[199]} con {std_acc_train[199]}")
    print(f"Accuracy Validation {mean_acc_val[199]} con {std_acc_val[199]}")
    print(f"Loss Training {mean_loss_train[199]} con {std_loss_train[199]}")
    print(f"Loss Validation {mean_loss_val[199]} con {std_loss_val[199]}")

    # Numero di epoche (uguale alla lunghezza del loss di ciascun seed)
    epochs = range(1, len(mean_loss_train) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 6), sharey=True, sharex=True
    )

    # Primo plot su ax1
    ax1.plot(
        epochs,
        mean_loss_train,
        label=f"Mean Training Loss (Qubit 1)",
        color="Royalblue",
    )
    ax1.fill_between(
        epochs,
        mean_loss_train - std_loss_train,
        mean_loss_train + std_loss_train,
        color="Royalblue",
        alpha=0.2,
        label=f"Std. Deviation",
    )
    ax1.set_title(f"Training Loss")
    ax1.set_xlim(1, len(mean_loss_train))
    ax1.set_ylabel("Loss")

    # Secondo plot su ax2 (per un altro qubit, ad esempio)
    ax2.plot(
        epochs, mean_loss_val, label=f"Mean Training Loss (Qubit 2)", color="Royalblue"
    )
    ax2.fill_between(
        epochs,
        mean_loss_val - std_loss_val,
        mean_loss_val + std_loss_val,
        color="Royalblue",
        alpha=0.2,
        label=f"Std. Deviation",
    )
    ax2.set_title(f"Validation Loss")
    ax2.set_xlim(1, len(mean_loss_train))

    ax3.plot(epochs, mean_acc_train, label=f"Mean Training Loss", color="Salmon")
    ax3.fill_between(
        epochs,
        mean_acc_train - std_acc_train,
        mean_acc_train + std_acc_train,
        color="Salmon",
        alpha=0.2,
        label=f"Std. Deviation",
    )
    ax3.set_title(f"Training Accuracy")
    ax3.set_xlim(1, len(mean_loss_train))
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")

    ax4.plot(epochs, mean_acc_val, label=f"Mean Validation Loss", color="Salmon")
    ax4.fill_between(
        epochs,
        mean_acc_val - std_acc_val,
        mean_acc_val + std_acc_val,
        color="Salmon",
        alpha=0.2,
        label=f"Std. Deviation",
    )
    ax4.set_title(f"Validation Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_xlim(1, len(mean_loss_train))

    plt.tight_layout()

    # Mostra il plot
    plt.savefig(f"Q{q}-loss-accuracy-error-bars.pdf")
    plt.show()
