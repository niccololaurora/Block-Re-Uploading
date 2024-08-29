import math
import time
import os
import cProfile
import pstats
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from qclassifier import Qclassifier
from plot_functions import (
    Bloch,
    plot_predictions,
    plot_loss_accuracy,
    plot_sphere,
    plot_absolute_gradients,
)
from help_functions import create_target, blocks_details

LOCAL_FOLDER = Path(__file__).parent


def main():
    # ==============
    # Configuration
    # ==============
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="Number of layers")
    args = parser.parse_args()

    # Parameters to adjust
    dataset = "digits"
    nqubits = 1
    local = True
    resize = 4
    epochs = 2
    entanglement = True
    pretraining = False
    iterazione = str(1)

    # Standard parameters
    nclasses = 2
    learning_rate = 0.001
    loss = "crossentropy"
    training_size = 10 * nclasses
    validation_size = 10 * nclasses
    test_size = 10 * nclasses
    batch_size = 5
    layers = 1
    seed = 42
    # Options for pooling: ["max", "average", "no"]
    pooling = "max"
    block_width, block_height, positions = blocks_details(resize, nqubits)

    # =============
    # Pretraining
    trained_params = None
    pretrain_name = "NO-PRE"
    if pretraining == True:
        parameters_from_outside = np.load(
            f"statistics_{iterazione}/trained_params_q{nqubits}-l{layers}.npy"
        )
        trained_params = tf.Variable(parameters_from_outside, dtype=tf.float32)
        pretrain_name = f"PRE-{iterazione}"

    # =============
    # Folders
    file_path = LOCAL_FOLDER / "statistics"
    if not os.path.exists("statistics"):
        os.makedirs("statistics", exist_ok=True)
    if not os.path.exists("plots"):
        os.makedirs("plots", exist_ok=True)

    # =============
    # Summary
    entanglement_name = 0
    if entanglement == True:
        entanglement_name = "E"
    else:
        entanglement_name = "NO-E"

    with open(
        f"Summary-Q{nqubits}-L{layers}-{resize}x{resize}-{pooling.capitalize()}-{entanglement_name}-{pretrain_name}.txt",
        "w",
    ) as file:
        print(f"Dataset: {dataset}", file=file)
        print(f"Locality: {str(local)}", file=file)
        print(f"Qubits: {nqubits}", file=file)
        print(f"Layers: {layers}", file=file)
        print(f"Epochs: {epochs}", file=file)
        print(f"Classes: {nclasses}", file=file)
        print(f"Loss: {loss}", file=file)
        print(f"Learning Rate: {learning_rate}", file=file)
        print(f"Sizes: (T, V) = ({training_size}, {validation_size})", file=file)
        print(f"Dimension: {resize}", file=file)
        print(f"Pretraining: {str(pretraining)}", file=file)

    # =============
    # Training
    # for su i qubits
    for k in range(1):
        accuracy_qubits = []

        # for su i layers
        for j in range(1):

            my_class = Qclassifier(
                training_size=training_size,
                validation_size=validation_size,
                test_size=test_size,
                nepochs=epochs,
                batch_size=batch_size,
                nlayers=layers,
                seed_value=seed,
                nqubits=nqubits,
                resize=resize,
                nclasses=nclasses,
                pooling=pooling,
                block_width=block_width,
                block_height=block_height,
                loss_2_classes=loss,
                learning_rate=learning_rate,
                positions=positions,
                local=local,
                parameters_from_outside=trained_params,
                dataset=dataset,
                entanglement=entanglement,
            )

            # PREDICTIONS BEFORE TRAINING
            val_set = my_class.get_val_set()
            _, predictions, final_states_before = my_class.prediction_function(val_set)
            label_states = create_target(nclasses)

            plot_predictions(
                predictions,
                val_set[0],
                val_set[1],
                "pred_before_" + f"q{nqubits}" + f"l{layers}_" + ".pdf",
                4,
                4,
            )

            dict_b = {
                "labels": val_set[1],
                "final_states": final_states_before,
                "label_states": label_states,
            }

            name_file = "points_b_" + f"q{nqubits}-" + f"l{layers}" + ".npz"
            np.savez(
                file_path / name_file,
                dict_b,
            )

            (
                absolute_gradients,
                trained_params,
                history_train_loss,
                history_val_loss,
                history_train_accuracy,
                history_val_accuracy,
            ) = my_class.training_loop()

            _, predictions, final_states_after = my_class.prediction_function(val_set)

            plot_predictions(
                predictions,
                val_set[0],
                val_set[1],
                "pred_after_" + f"q{nqubits}-" + f"l{layers}_" + ".pdf",
                4,
                4,
            )

            dict_a = {
                "labels": val_set[1],
                "final_states": final_states_after,
                "label_states": label_states,
            }

            # SAVING
            name_file = "points_a_" + f"q{nqubits}-" + f"l{layers}" + ".npz"
            np.savez(
                file_path / name_file,
                dict_a,
            )

            name_file = "abs_grads_" + f"q{nqubits}-" + f"l{layers}" + ".npy"
            np.save(
                file_path / name_file,
                absolute_gradients,
            )

            name_file = "trained_params_" + f"q{nqubits}-" + f"l{layers}" + ".npy"
            np.save(
                file_path / name_file,
                trained_params,
            )

            name_file = "history_train_loss_" + f"q{nqubits}-" + f"l{layers}" + ".npy"
            np.save(
                file_path / name_file,
                history_train_loss,
            )

            name_file = "history_val_loss_" + f"q{nqubits}-" + f"l{layers}" + ".npy"
            np.save(
                file_path / name_file,
                history_val_loss,
            )

            name_file = (
                "history_train_accuracy_" + f"q{nqubits}-" + f"l{layers}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_train_accuracy,
            )

            name_file = "history_val_accuracy_" + f"q{nqubits}-" + f"l{layers}" + ".npy"
            np.save(
                file_path / name_file,
                history_val_accuracy,
            )

            # PLOTTING
            if nqubits == 1:
                plot_sphere(
                    nqubits,
                    layers,
                    val_set[1],
                    label_states,
                    final_states_before,
                    final_states_after,
                )
            plot_loss_accuracy(
                nqubits,
                layers,
                epochs,
                history_train_loss,
                history_val_loss,
                history_train_accuracy,
                history_val_accuracy,
            )

            plot_absolute_gradients(nqubits, layers, epochs, absolute_gradients)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    # Salva le statistiche su un file
    with open("profiling_stats.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
