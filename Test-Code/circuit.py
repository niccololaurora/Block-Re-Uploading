import sys
import os

# Get the parent directory of the current file (test.py)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from qclassifier import Qclassifier
from help_functions import blocks_details

# Circuit
nqubits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
layers = [1, 2, 3]
entanglement = [True, False]
pooling = ["max", "no"]

# Data
dataset = "digits"
nclasses = 2
local = False
resize = 4
training_size = 250 * nclasses
validation_size = 250 * nclasses
test_size = 100 * nclasses

# Training
epochs = 50
batch_size = 50
learning_rate = 0.001
loss = "crossentropy"
seed = 42
trained_params = None

for q in nqubits:
    print(f"Qubit: {q}")
    for l in layers:
        print(f"Layers: {l}")
        for e in entanglement:
            print(f"Entanglement: {e}")
            for p in pooling:
                print(f"Pooling: {p}")
                block_width, block_height, positions = blocks_details(resize, q)
                my_class = Qclassifier(
                    training_size=training_size,
                    validation_size=validation_size,
                    test_size=test_size,
                    nepochs=epochs,
                    batch_size=batch_size,
                    nlayers=l,
                    seed_value=seed,
                    nqubits=q,
                    resize=resize,
                    nclasses=nclasses,
                    pooling=p,
                    block_width=block_width,
                    block_height=block_height,
                    loss_2_classes=loss,
                    learning_rate=learning_rate,
                    positions=positions,
                    local=local,
                    parameters_from_outside=trained_params,
                    dataset=dataset,
                    entanglement=e,
                )

                my_class.print_circuit()
