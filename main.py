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
    parser.add_argument("--qubit", type=int, required=True, help="Number of qubits")
    parser.add_argument("--layer", type=int, required=True, help="Number of layers")
    parser.add_argument(
        "--learning_rate", type=float, required=True, help="Learning Rate"
    )
    # parser.add_argument("--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()

    # ====================
    # Parameters to adjust
    # ====================
    dataset = "digits"
    resize = 8
    epochs = 100

    # Circuit
    local = True
    entanglement = False
    pooling = "max"
    pretraining = False

    # 'target' --> Se voglio allenare finche non si raggiunge una loss fissata
    # 'No' --> Se voglio allenare per un numero fissato di epoche senza early stopping (1E-3)
    # 'fluctuation' --> Se voglio allenare finche la varianza è sotto una certa soglia (1E-4)
    criterion = "No"
    iterazione = str(1)

    # Outside parameters
    layers = args.layer
    nqubits = args.qubit
    learning_rate = args.learning_rate
    # seed_parameters = args.seed
    seed_parameters = 0

    # Loss and classes
    loss = "crossentropy"
    nclasses = 2

    # Training sizes
    training_size = 200 * nclasses
    validation_size = 200 * nclasses
    test_size = 10 * nclasses
    batch_size = 50

    # ====================
    # ====================

    # Options for pooling: ["max", "average", "no"]
    block_width, block_height, positions = blocks_details(resize, nqubits)

    # =============
    # Pretraining
    trained_params = None
    pretrain_name = "Nopre"
    if pretraining == True:
        parameters_from_outside = np.load(
            f"statistics_{iterazione}/trained_params_q{nqubits}-l{layers}.npy"
        )
        trained_params = tf.Variable(parameters_from_outside, dtype=tf.float32)
        pretrain_name = f"Pre{iterazione}"

    # =============
    # Folders
    file_path_stats = LOCAL_FOLDER / f"statistics"
    if not os.path.exists(f"statistics"):
        os.makedirs(f"statistics", exist_ok=True)
    file_path_plots = LOCAL_FOLDER / f"plots"
    if not os.path.exists(f"plots"):
        os.makedirs(f"plots", exist_ok=True)

    # =============
    # Summary
    entanglement_name = 0
    if entanglement == True:
        entanglement_name = "Ent"
    else:
        entanglement_name = "NEnt"

    if not os.path.exists(f"summary"):
        os.makedirs(f"summary", exist_ok=True)
    with open(
        f"summary/Summary-Q{nqubits}-L{layers}-{resize}x{resize}-{pooling.capitalize()}-{entanglement_name}-{pretrain_name}-lr{learning_rate}-S{seed_parameters}.txt",
        "a",
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
        print(f"Seed: {str(seed_parameters)}", file=file)

    # =============
    if not os.path.exists(f"epochs"):
        os.makedirs(f"epochs", exist_ok=True)
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
                seed_parameters=seed_parameters,
                seed_data=0,
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
                criterion=criterion,
            )

            my_class.print_circuit()

            # PREDICTIONS BEFORE TRAINING
            val_set = my_class.get_val_set()
            _, predictions, final_states_before = my_class.prediction_function(val_set)
            label_states = create_target(nclasses)

            name_file = (
                f"pred_before_q{nqubits}_l{layers}_lr{learning_rate}_S{seed_parameters}"
                + ".pdf"
            )
            plot_predictions(
                predictions,
                val_set[0],
                val_set[1],
                file_path_plots,
                name_file,
                4,
                4,
            )

            dict_b = {
                "labels": val_set[1],
                "final_states": final_states_before,
                "label_states": label_states,
            }

            name_file = f"points_b_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npz"
            np.savez(
                file_path_stats / name_file,
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

            name_file = f"pred_after_q{nqubits}_l{layers}_lr{learning_rate}_S{seed_parameters}.pdf"
            plot_predictions(
                predictions,
                val_set[0],
                val_set[1],
                file_path_plots,
                name_file,
                4,
                4,
            )

            dict_a = {
                "labels": val_set[1],
                "final_states": final_states_after,
                "label_states": label_states,
            }

            # SAVING
            name_file = f"points_a_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npz"
            np.savez(
                file_path_stats / name_file,
                dict_a,
            )

            name_file = f"abs_grads_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npy"
            np.save(
                file_path_stats / name_file,
                absolute_gradients,
            )

            name_file = f"trained_params_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npy"
            np.save(
                file_path_stats / name_file,
                trained_params,
            )

            name_file = f"history_train_loss_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npy"
            np.save(
                file_path_stats / name_file,
                history_train_loss,
            )

            name_file = f"history_val_loss_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npy"
            np.save(
                file_path_stats / name_file,
                history_val_loss,
            )

            name_file = f"history_train_accuracy_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npy"
            np.save(
                file_path_stats / name_file,
                history_train_accuracy,
            )

            name_file = f"history_val_accuracy_q{nqubits}-l{layers}_lr{learning_rate}_S{seed_parameters}.npy"
            np.save(
                file_path_stats / name_file,
                history_val_accuracy,
            )

            # PLOTTING
            if nqubits == 1:
                name = f"sphere_q{nqubits}_l{layers}_lr{learning_rate}_S{seed_parameters}-B.pdf"
                plot_sphere(
                    val_set[1], label_states, final_states_before, file_path_plots, name
                )
                name = f"sphere_q{nqubits}_l{layers}_lr{learning_rate}_S{seed_parameters}-A.pdf"
                plot_sphere(
                    val_set[1], label_states, final_states_after, file_path_plots, name
                )
            plot_loss_accuracy(
                nqubits,
                layers,
                epochs,
                history_train_loss,
                history_val_loss,
                history_train_accuracy,
                history_val_accuracy,
                file_path_plots,
            )

            plot_absolute_gradients(
                nqubits, layers, epochs, absolute_gradients, file_path_plots
            )


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()

    # Salva le statistiche su un file
    """
    with open("profiling_stats.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    """
