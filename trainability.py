import os
import tensorflow as tf
import numpy as np
import argparse

from pathlib import Path

from help_functions import blocks_details
from qclassifier import Qclassifier


parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True, help="Number of layers")
args = parser.parse_args()

# Fix number of gates
# layers = [1, 5, 10, 15, 20, 25, 40, 60, 80, 100]
layers = args.layer
resize = 8


# Image details
# Fix number of qubits
nqubits = [15]
block_width, block_height, positions = blocks_details(nqubits[0])
digits = [0, 1]
local = False

# Sample image and parameters
n_models = 50
for j in range(len(nqubits)):
    print("=" * 60)
    print(f"Number of qubits: {nqubits[0]}")
    for k in range(1):
        print("*" * 30)
        print(f"Layers {layers}")
        for i in range(n_models):
            print(f"Model {i}")

            # Gradients Folder
            LOCAL_FOLDER = Path(__file__).parent
            file_path = LOCAL_FOLDER / "gradients"
            if not os.path.exists("gradients"):
                os.makedirs("gradients")

            seed_value = i
            qclass = Qclassifier(
                training_size=10,
                validation_size=10,
                test_size=10,
                nepochs=1,
                batch_size=1,
                nlayers=layers,
                seed_value=seed_value,
                nqubits=nqubits[0],
                resize=resize,
                nclasses=len(digits),
                pooling="max",
                block_width=block_width,
                block_height=block_height,
                loss_2_classes="crossentropy",
                learning_rate=0.01,
                digits=digits,
                positions=positions,
                local=local,
            )

            qclass.print_circuit()

            training = qclass.get_train_set()
            vparams = qclass.get_vparams()
            grads, loss = qclass.trainability(training[0][0], training[1][0])

            name_file = f"gradients_Q{nqubits[0]}_L{layers}_M{i}" + ".npz"
            np.savez(
                file_path / name_file,
                grads=grads,
                vparams=vparams,
            )
