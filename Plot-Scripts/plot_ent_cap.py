import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carica la matrice di entangling capability dal file .npy
ent_cap_matrix = np.load("ent_cap_matrix.npy")

# Definisci nqubits e layers se non sono già definiti
nqubits = [1, 2, 3, 4]  # Esempio, sostituisci con i tuoi valori
layers = [1, 2, 3, 4]  # Esempio, sostituisci con i tuoi valori

plt.figure(figsize=(8, 6))

# Usa sns.heatmap per plottare la matrice
ax = sns.heatmap(
    ent_cap_matrix,
    annot=True,  # Mostra i valori nella heatmap
    fmt=".2f",  # Formattazione dei numeri
    cmap="viridis",  # Colormap per la heatmap
    xticklabels=layers,  # Etichette per l'asse X (layers)
    yticklabels=nqubits,  # Etichette per l'asse Y (qubits)
    annot_kws={"size": 16},  # Dimensione del testo nei valori annotati
)

# Titolo e etichette degli assi
plt.title("Entangling Capability", fontsize=16)
ax.set_xlabel("Layers", fontsize=12)
ax.set_ylabel("Qubits", fontsize=12)

# Mostra la heatmap
plt.tight_layout()
plt.savefig("Ent-Capability.pdf")  # Salva il grafico come PDF
plt.show()
