from qibo import Circuit, gates
from qibo.quantum_info import entangling_capability
from qclassifier import Qclassifier
from help_functions import block_sizes, blocks_details

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def embedding_circuit(resize, nqubits):
    """Method for building the classifier's embedding block.

    Returns:
        Qibo circuit.
    """
    c = Circuit(nqubits)
    block_width, block_height, positions = blocks_details(resize, nqubits)

    for j in range(nqubits):
        sizes = block_sizes(resize, block_width, block_height)
        for i in range(sizes[j]):
            if i % 3 == 1:
                c.add(gates.RZ(j, theta=0))
            else:
                c.add(gates.RY(j, theta=0))

    return c


def pooling_circuit(nqubits):
    c = Circuit(nqubits)
    for q in range(nqubits):
        c.add(gates.RX(q, theta=0))
    return c


def entanglement_circuit(resize, nqubits):

    c = Circuit(nqubits)
    if resize == 4:
        if nqubits == 2:
            c.add(gates.CZ(0, 1))

        elif nqubits == 3:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(1, 2))

        elif nqubits == 4:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(2, 3))

        elif nqubits == 5:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(2, 4))
            c.add(gates.CZ(3, 4))

        elif nqubits == 6:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(2, 5))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(4, 5))

        elif nqubits == 7:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 6))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(5, 6))

        elif nqubits == 8:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))

        elif nqubits == 9:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 2))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(7, 8))

        elif nqubits == 10:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 2))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 9))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 9))

        elif nqubits == 11:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 10))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 10))
            c.add(gates.CZ(9, 10))

        elif nqubits == 12:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 11))
            c.add(gates.CZ(9, 11))
            c.add(gates.CZ(10, 11))

        elif nqubits == 13:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(2, 5))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(9, 12))
            c.add(gates.CZ(10, 12))
            c.add(gates.CZ(11, 12))

        elif nqubits == 14:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(2, 5))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(9, 13))
            c.add(gates.CZ(10, 13))
            c.add(gates.CZ(11, 12))
            c.add(gates.CZ(12, 13))

        elif nqubits == 15:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(9, 13))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(10, 14))
            c.add(gates.CZ(11, 14))
            c.add(gates.CZ(12, 13))
            c.add(gates.CZ(13, 14))

        elif nqubits == 16:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(9, 13))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(10, 14))
            c.add(gates.CZ(11, 15))
            c.add(gates.CZ(12, 13))
            c.add(gates.CZ(13, 14))
            c.add(gates.CZ(14, 15))

    if resize == 8:
        if nqubits == 2:
            c.add(gates.CZ(0, 1))

        elif nqubits == 3:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(1, 2))

        elif nqubits == 4:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(2, 3))

        elif nqubits == 5:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(2, 4))
            c.add(gates.CZ(3, 4))

        elif nqubits == 6:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(2, 5))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(4, 5))

        elif nqubits == 7:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 5))
            c.add(gates.CZ(3, 6))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(5, 6))

        elif nqubits == 8:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))

        elif nqubits == 9:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(7, 8))

        elif nqubits == 10:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(8, 9))

        elif nqubits == 11:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 6))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 7))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 10))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(9, 10))

        elif nqubits == 12:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(10, 11))

        elif nqubits == 13:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(11, 12))

        elif nqubits == 14:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 10))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 13))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(11, 12))
            c.add(gates.CZ(12, 13))

        elif nqubits == 15:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 10))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 11))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 12))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 13))
            c.add(gates.CZ(9, 14))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(11, 12))
            c.add(gates.CZ(12, 13))
            c.add(gates.CZ(13, 14))

        elif nqubits == 16:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(9, 13))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(10, 14))
            c.add(gates.CZ(11, 15))
            c.add(gates.CZ(12, 13))
            c.add(gates.CZ(13, 14))
            c.add(gates.CZ(14, 15))

        else:
            raise_error(
                ValueError, "Number of qubits not supported by this architecture"
            )

    return c


def circuit(
    resize, entanglement, entanglement_between_layers, pooling, nqubits, nlayers
):
    circuit = Circuit(nqubits)

    for k in range(nlayers):
        # Embedding
        circuit += embedding_circuit(resize, nqubits)

        # Entanglement
        if entanglement == True:
            if nqubits != 1:
                circuit += entanglement_circuit(resize, nqubits)

        # Pooling
        if pooling != "no":
            circuit += pooling_circuit(nqubits)

        # If last layer break the loop
        if k == nlayers - 1:
            break

        # Entanglement between layers
        if entanglement_between_layers == True:
            if nqubits != 1:
                circuit += entanglement_circuit(resize, nqubits)

    return circuit


resize = 8
nqubits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
layers = [1, 2, 3, 4, 5, 6]

entanglement_between_layers = True
entanglement = True
pooling = "max"

"""
print("=" * 50)

c = embedding_circuit(resize, nqubits)
print(c.draw())
ent_cap = entangling_capability(c, samples=10)
print(f"Ent-Capability: {ent_cap}")

print("=" * 50)

c = pooling_circuit(nqubits)
print(c.draw())
ent_cap = entangling_capability(c, samples=10)
print(f"Ent-Capability: {ent_cap}")

print("=" * 50)

c = entanglement_circuit(resize, nqubits)
print(c.draw())
ent_cap = entangling_capability(c, samples=10)
print(f"Ent-Capability: {ent_cap}")

print("=" * 50)
print("=" * 50)

"""

ent_cap_matrix = np.zeros((len(nqubits), len(layers)))

for i, q in enumerate(nqubits):
    for j, l in enumerate(layers):

        # Crea il circuito con i parametri specificati
        c = circuit(resize, entanglement, entanglement_between_layers, pooling, q, l)

        # Calcola l'entangling capability
        ent_cap = entangling_capability(c, samples=10)

        # Assegna il valore calcolato alla matrice
        ent_cap_matrix[i, j] = ent_cap

        # Stampa il risultato per monitorare i progressi
        print("=" * 20)
        print(f"Ent-Capability for Qubits {q} and Layers {l}: {ent_cap_matrix[i, j]}")

np.save("ent_cap_matrix.npy", ent_cap_matrix)
plt.figure(figsize=(8, 6))

# Usa sns.heatmap per plottare la matrice
ax = sns.heatmap(
    ent_cap_matrix,
    annot=True,  # Mostra i valori nella heatmap
    fmt=".2f",  # Formattazione dei numeri
    cmap="viridis",  # Colormap per la heatmap
    xticklabels=layers,  # Etichette per l'asse X (layers)
    yticklabels=nqubits,
    annot_kws={"size": 16},
)

# Titolo e etichette degli assi
plt.title("Entangling Capability", fontsize=16)
ax.set_xlabel("Layers", fontsize=12)
ax.set_ylabel("Qubits", fontsize=12)

# Mostra la heatmap
plt.tight_layout()
plt.savefig("Ent-Capability-FinoQ10.pdf")
plt.show()
