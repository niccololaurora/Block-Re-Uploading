import os
import tensorflow as tf
import numpy as np

from pathlib import Path

from qclassifier import Qclassifier


# Fix number of gates
layers = [1, 5, 10, 20, 30, 40]
resize = 8
block_sizes = [[8, 8], [4, 8], [4, 4], [3, 4], [2, 4], [2, 2]]
training_size = 10
digits = [0, 1]

# Fix number of qubits
nqubits = [1, 2, 4, 6, 8, 16]

# Sample image and parameters
n_models = 10
for j in range(len(nqubits)):
    print("=" * 60)
    print(f"Number of qubits: {nqubits[j]}")
    for k in range(len(layers)):
        print("*" * 30)
        print(f"Layers {layers[k]}")
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
                nlayers=layers[k],
                seed_value=seed_value,
                nqubits=nqubits[j],
                resize=resize,
                nclasses=len(digits),
                pooling="max",
                block_width=block_sizes[j][0],
                block_height=block_sizes[j][1],
                loss_2_classes="crossentropy-gradients",
                learning_rate=0.01,
                digits=digits,
            )

            training = qclass.get_train_set()
            vparams = qclass.get_vparams()
            grads, loss = qclass.trainability(training[0][0], training[1][0])

            print(f"Seed {seed_value}")
            print(f"Params {vparams[:4]}")
            print(f"Loss {loss}")
            print(f"Grads {grads[:4]}")

            name_file = f"gradients_Q{nqubits[j]}_L{layers[k]}_M{i}" + ".npz"
            np.savez(
                file_path / name_file,
                grads=grads,
                vparams=vparams,
            )
