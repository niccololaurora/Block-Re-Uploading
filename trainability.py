import os
import tensorflow as tf
import numpy as np

from pathlib import Path

from qclassifier import Qclassifier


# Fix number of gates
layers = 1
resize = 8
block_sizes = [[resize, resize]]
training_size = 10
digits = [0, 1]


# Fix number of qubits
nqubits = 1

# Sample image and parameters
n_models = 10
for i in range(n_models):
    print("=" * 60)
    print(f"Model {i}")

    # Gradients Folder
    LOCAL_FOLDER = Path(__file__).parent
    file_path = LOCAL_FOLDER / "trainability"
    if not os.path.exists("trainability"):
        os.makedirs("trainability")

    seed_value = i
    qclass = Qclassifier(
        training_size=training_size,
        validation_size=10,
        test_size=10,
        nepochs=1,
        batch_size=training_size,
        nlayers=layers,
        seed_value=seed_value,
        nqubits=nqubits,
        resize=resize,
        nclasses=len(digits),
        pooling="max",
        block_width=block_sizes[0][0],
        block_height=block_sizes[0][1],
        loss_2_classes="crossentropy-gradients",
        learning_rate=0.01,
        digits=digits,
    )

    training = qclass.get_train_set()
    grads, loss = qclass.trainability(training[0][0], training[1][0])

    print(f"Loss {loss}")
    print(f"Grads {grads[:10]}")

    name_file = f"gradients_M{i}" + ".npy"
    np.save(
        file_path / name_file,
        grads,
    )
