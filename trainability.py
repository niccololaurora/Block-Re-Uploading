import os
import tensorflow as tf
import numpy as np

from pathlib import Path

from help_functions import BinaryCrossentropy
from qclassifier import Qclassifier


# Images
digits = [0, 1]
training_size = 2

# Fix number of gates
layers = 1
resize = 8
block_sizes = [[resize, resize]]

# Fix number of qubits
nqubits = 1

# Sample image and parameters
n_models = 10
for i in range(n_models):
    print("=" * 60)
    print(f"Model {i}")

    seed_value = i
    qclass = Qclassifier(
        training_size=training_size,
        validation_size=10,
        test_size=10,
        nepochs=1,
        batch_size=2,
        nlayers=layers,
        seed_value=seed_value,
        nqubits=nqubits,
        resize=resize,
        nclasses=len(digits),
        pooling="max",
        block_width=block_sizes[0][0],
        block_height=block_sizes[0][1],
        loss_2_classes="crossentropy",
        learning_rate=0.01,
        digits=digits,
    )

    training = qclass.get_train_set()

    """
    random_index = tf.random.uniform(
        shape=[], minval=0, maxval=training_size, dtype=tf.int32
    )
    random_data = training[0][random_index]
    random_label = training[1][random_index]
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    grads = qclass.train_step(training[0], training[1], optimizer)

    LOCAL_FOLDER = Path(__file__).parent
    file_path = LOCAL_FOLDER / "trainability"
    if not os.path.exists("trainability"):
        os.makedirs("trainability")

    name_file = f"gradients_M{i}" + ".npy"
    np.save(
        file_path / name_file,
        grads,
    )
