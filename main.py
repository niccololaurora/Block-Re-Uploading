import math
import time
import os
import cProfile
import pstats
import numpy as np
import tensorflow as tf
from pathlib import Path
from qclassifier import Qclassifier
from plot_functions import Bloch, plot_predictions, plot_loss_accuracy, plot_sphere
from help_functions import create_target

LOCAL_FOLDER = Path(__file__).parent


def main():
    # ==============
    # Configuration
    # ==============
    epochs = 100
    learning_rate = 0.01
    loss = "crossentropy"
    digits = [5, 6]
    training_size = 200 * len(digits)
    validation_size = 200 * len(digits)
    test_size = 200 * len(digits)
    batch_size = 40
    resize = 8
    # layers = [1, 2, 3, 4, 5, 6]
    layers = [2]
    seed = 1
    # block_sizes = [[2, 4], [3, 4], [4, 4], [4, 8], [8, 8]]
    block_sizes = [[resize, resize]]
    # nqubits = [8, 6, 4, 2, 1]
    nqubits = [1]
    pooling = "max"

    file_path = LOCAL_FOLDER / "statistics"
    if not os.path.exists("statistics"):
        os.makedirs("statistics")
    if not os.path.exists("plots"):
        os.makedirs("plots")

    with open("summary.txt", "w") as file:
        print(f"Qubits: {nqubits[0]}", file=file)
        print(f"Layers: {layers[0]}", file=file)
        print(f"Epochs: {epochs}", file=file)
        print(f"Classes: {len(digits)}", file=file)
        print(f"Loss: {loss}", file=file)
        print(f"Learning Rate: {learning_rate}", file=file)
        print(f"Sizes: (T, V) = ({training_size}, {validation_size})", file=file)

    for k in range(len(nqubits)):
        accuracy_qubits = []
        for j in range(len(layers)):

            my_class = Qclassifier(
                training_size=training_size,
                validation_size=validation_size,
                test_size=test_size,
                nepochs=epochs,
                batch_size=batch_size,
                nlayers=layers[j],
                seed_value=seed,
                nqubits=nqubits[k],
                resize=resize,
                nclasses=len(digits),
                pooling=pooling,
                block_width=block_sizes[k][0],
                block_height=block_sizes[k][1],
                loss_2_classes=loss,
                learning_rate=learning_rate,
                digits=digits,
            )

            # PREDICTIONS BEFORE TRAINING
            val_set = my_class.get_val_set()
            _, predictions, final_states_before = my_class.prediction_function(val_set)
            label_states = create_target(len(digits))
            plot_predictions(predictions, val_set[0], val_set[1], "pred_b.pdf", 4, 4)

            dict_b = {
                "labels": val_set[1],
                "final_states": final_states_before,
                "label_states": label_states,
            }

            name_file = "points_b_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npz"
            np.savez(
                file_path / name_file,
                dict_b,
            )

            start_time = time.time()
            (
                trained_params,
                history_train_loss,
                history_val_loss,
                history_train_accuracy,
                history_val_accuracy,
            ) = my_class.training_loop()

            end_time = time.time()
            elapsed_time_minutes = (end_time - start_time) / 60

            _, predictions, final_states_after = my_class.prediction_function(val_set)
            plot_predictions(predictions, val_set[0], val_set[1], "pred_a.pdf", 4, 4)

            dict_a = {
                "labels": val_set[1],
                "final_states": final_states_after,
                "label_states": label_states,
            }

            # SAVING
            name_file = "points_a_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npz"
            np.savez(
                file_path / name_file,
                dict_a,
            )

            name_file = "trained_params_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            np.save(
                file_path / name_file,
                trained_params,
            )

            name_file = (
                "history_train_loss_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_train_loss,
            )

            name_file = (
                "history_val_loss_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_val_loss,
            )

            name_file = (
                "history_train_accuracy_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_train_accuracy,
            )

            name_file = (
                "history_val_accuracy_" + f"q{nqubits[k]}" + f"l{layers[j]}" + ".npy"
            )
            np.save(
                file_path / name_file,
                history_val_accuracy,
            )

            # PLOTTING
            plot_sphere(
                nqubits[k],
                layers[j],
                val_set[1],
                label_states,
                final_states_before,
                final_states_after,
            )
            plot_loss_accuracy(
                nqubits[k],
                layers[j],
                epochs,
                history_train_loss,
                history_val_loss,
                history_train_accuracy,
                history_val_accuracy,
            )


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
