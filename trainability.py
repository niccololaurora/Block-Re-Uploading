import os
import tensorflow as tf
import numpy as np
import argparse

from pathlib import Path

from qclassifier import Qclassifier


parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True, help="Number of layers")
args = parser.parse_args()

# Fix number of gates
# layers = [1, 5, 10, 15, 20, 25, 40, 60, 80, 100]
layers = args.layer
resize = 8

# Image details
block_width = [3, 3, 2, 4, 4]
block_height = [4, 4, 4, 4, 4]
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

training_size = 1
digits = [0, 1]

# Fix number of qubits
nqubits = 12

# Sample image and parameters
n_models = 50
for j in range(1):
    print("=" * 60)
    print(f"Number of qubits: {nqubits}")
    for k in range(1):
        print("*" * 30)
        print(f"Layers {layers}")
        for i in range(n_models):
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
                block_width=block_sizes[j][0],
                block_height=block_sizes[j][1],
                loss_2_classes="crossentropy-gradients",
                learning_rate=0.01,
                digits=digits,
            )

            qclass.print_circuit()

            training = qclass.get_train_set()
            vparams = qclass.get_vparams()
            grads, loss = qclass.trainability(training[0][0], training[1][0])

            name_file = f"gradients_Q{nqubits}_L{layers}_M{i}" + ".npz"
            np.savez(
                file_path / name_file,
                grads=grads,
                vparams=vparams,
            )
