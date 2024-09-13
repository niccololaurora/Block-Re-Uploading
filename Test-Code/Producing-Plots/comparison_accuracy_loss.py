import numpy as np
import matplotlib.pyplot as plt


def accuracy_compare(qubit, layer, option, size, path):
    file_names = []
    for i in range(layer):
        file_names += [path + f"/statistics/history_{option}_loss_q{qubit}-l{i+1}.npy"]

    WIDTH = 0.5

    # Inizializza la figura e l'asse
    plt.figure(figsize=(10 * WIDTH * 3 / 2, 10 * WIDTH * 7 / 8))

    # Carica i dati e crea il plot
    for i, file_name in enumerate(file_names):
        data = np.load(file_name)  # Carica i dati dal file
        plt.plot(
            data, label=f"Layer {i+1}", alpha=0.8, lw=1.5, ls="-"
        )  # Traccia i dati e usa il nome del file come etichetta

    title = 0
    if option == "val":
        title = "Validation"
    if option == "train":
        title = "Training"

    # Configura le etichette e il titolo
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title + " Loss")
    plt.xlim(0, 200)  # Imposta il limite degli assi x
    plt.legend()

    # Salva il grafico come file immagine
    plt.savefig(
        path + f"/loss-{option}-q{qubit}-l{layer}-{size}x{size}.pdf", format="pdf"
    )


def loss_comparison(qubit, layer, option, size, path):
    # Genera i nomi dei file dinamicamente
    file_names = []
    for i in range(layer):
        file_names += [path + f"/statistics/history_{option}_loss_q{qubit}-l{i+1}.npy"]

    WIDTH = 0.5

    # Inizializza la figura e l'asse
    plt.figure(figsize=(10 * WIDTH * 3 / 2, 10 * WIDTH * 7 / 8))

    # Carica i dati e crea il plot
    for i, file_name in enumerate(file_names):
        data = np.load(file_name)  # Carica i dati dal file
        plt.plot(
            data, label=f"Layer {i+1}", alpha=0.8, lw=1.5, ls="-"
        )  # Traccia i dati e usa il nome del file come etichetta

    title = 0
    if option == "val":
        title = "Validation"
    if option == "train":
        title = "Training"

    # Configura le etichette e il titolo
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title + " Loss")
    plt.xlim(0, 200)  # Imposta il limite degli assi x
    plt.legend()

    # Salva il grafico come file immagine
    plt.savefig(path + f"/loss-{option}-q{qubit}-{size}x{size}.pdf", format="pdf")


def accuracy_comparison(qubit, layer, option, size, path):
    print(f"Qubit {qubit}")
    # Genera i nomi dei file dinamicamente
    file_names = []
    for i in range(layer):
        file_names += [
            path + f"/statistics/history_{option}_accuracy_q{qubit}-l{i+1}.npy"
        ]
    WIDTH = 0.5

    # Inizializza la figura e l'asse
    plt.figure(figsize=(10 * WIDTH * 3 / 2, 10 * WIDTH * 7 / 8))

    # Carica i dati e crea il plot
    for i, file_name in enumerate(file_names):
        data = np.load(file_name)  # Carica i dati dal file
        plt.plot(
            data, label=f"Layer {i+1}", alpha=0.8, lw=1.5, ls="-"
        )  # Traccia i dati e usa il nome del file come etichetta

    title = 0
    if option == "val":
        title = "Validation"
    if option == "train":
        title = "Training"

    # Configura le etichette e il titolo
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title + " Accuracy")
    plt.xlim(0, 200)  # Imposta il limite degli assi x
    plt.legend()

    # Salva il grafico come file immagine
    plt.savefig(path + f"/accuracy-{option}-q{qubit}-{size}x{size}.pdf", format="pdf")


def gradient_comparison(qubit, layer, size, path):
    # Genera i nomi dei file dinamicamente
    file_names = []
    for i in range(layer):
        file_names += [path + f"/statistics/abs_grads_q{qubit}-l{i+1}.npy"]

    WIDTH = 0.5

    # Inizializza la figura e l'asse
    plt.figure(figsize=(10 * WIDTH * 3 / 2, 10 * WIDTH * 7 / 8))

    # Carica i dati e crea il plot
    for i, file_name in enumerate(file_names):
        data = np.load(file_name)  # Carica i dati dal file
        plt.plot(
            data, label=f"Layer {i+1}", alpha=0.8, lw=1.5, ls="-"
        )  # Traccia i dati e usa il nome del file come etichetta

    # Configura le etichette e il titolo
    plt.xlabel("Epochs")
    plt.ylabel("Gradient")
    plt.title("Gradient")
    plt.xlim(0, 200)  # Imposta il limite degli assi x
    plt.legend()

    # Salva il grafico come file immagine
    plt.savefig(path + f"/gradient-q{qubit}-{size}x{size}.pdf", format="pdf")
