import math
import time
import os
import cProfile
import pstats
import numpy as np
import tensorflow as tf
from pathlib import Path
import optuna
from qclassifier import Qclassifier
from plot_functions import Bloch, plot_predictions, plot_loss_accuracy, plot_sphere
from help_functions import create_target

LOCAL_FOLDER = Path(__file__).parent

def objective(trial):
    # ==============
    # Configuration
    # ==============
    epochs = 2
    nclasses = 2
    training_size = 20 * nclasses
    validation_size = 20 * nclasses
    test_size = 100 * nclasses
    batch_size = 40
    resize = 8
    layers = [1]
    seed = 1
    block_sizes = [[resize, resize]]
    nqubits = [1]
    pooling = "max"

    # Hyperparameter to optimize
    learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.5, 0.01, 0.05, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5])

    file_path = LOCAL_FOLDER / "statistics"
    if not os.path.exists("statistics"):
        os.makedirs("statistics")
    if not os.path.exists("plots"):
        os.makedirs("plots")

    accuracy_qubits = []
    for k in range(len(nqubits)):
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
                nclasses=nclasses,
                pooling=pooling,
                block_width=block_sizes[k][0],
                block_height=block_sizes[k][1],
                loss_2_classes="fidelity",
                learning_rate=learning_rate  # Ensure Qclassifier can accept learning_rate
            )

            # PREDICTIONS BEFORE TRAINING

            val_set = my_class.get_val_set()
            _, predictions, final_states_before = my_class.prediction_function(val_set)
            label_states = create_target(nclasses)
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
            
            # Objective value to optimize
            accuracy_qubits.append(max(history_val_accuracy))

    return np.mean(accuracy_qubits)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=2)

    profiler.disable()

    # Salva le statistiche su un file
    with open("profiling_stats.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()

    with open("best_params.txt", "w") as f:
        f.write(f"Best trial value: {study.best_trial.value}\n")
        f.write("Best parameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
