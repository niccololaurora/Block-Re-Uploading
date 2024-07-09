import numpy as np
import tensorflow as tf
import math
import os

from qibo import Circuit, gates, hamiltonians, set_backend
from qibo.config import raise_error
from qibo.symbols import Z, I
from qibo.optimizers import optimize

from data import initialize_data, pooling_creator, create_blocks, shuffle
from help_functions import (
    fidelity,
    create_target,
    number_params,
    create_hamiltonian,
    block_sizes,
)


class Qclassifier:
    def __init__(
        self,
        training_size,
        validation_size,
        test_size,
        batch_size,
        nepochs,
        nlayers,
        pooling,
        seed_value,
        block_width,
        block_height,
        nqubits,
        nclasses,
        loss_2_classes,
        resize,
        learning_rate,
        digits,
        positions,
    ):

        np.random.seed(seed_value)
        set_backend("tensorflow")

        # TRAINING
        self.nclasses = nclasses
        self.training_size = training_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.targets = create_target(nclasses)
        self.loss = loss_2_classes
        self.learning_rate = learning_rate
        self.alpha = tf.Variable(tf.random.normal((nclasses,)), dtype=tf.float32)

        # IMAGE
        self.resize = resize
        self.train, self.test, self.validation = initialize_data(
            digits,
            self.training_size,
            self.test_size,
            self.validation_size,
            resize,
        )
        self.positions = positions
        self.block_width = block_width
        self.block_height = block_height

        # CIRCUIT
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.pooling = pooling
        self.n_embed_params = 2 * resize**2
        self.params_1layer = number_params(
            self.n_embed_params, self.nqubits, self.pooling
        )
        self.n_params = self.params_1layer * nlayers
        self.vparams = tf.Variable(
            tf.random.uniform((self.n_params,)),
            minval=-math.pi,
            maxval=math.pi,
            dtype=tf.float32,
        )
        self.hamiltonian = create_hamiltonian(self.nqubits, local=False)
        self.ansatz = self.circuit()

    def print_circuit(self):
        if not os.path.exists("ansatz_draw"):
            os.makedirs("ansatz_draw")
        filename = f"ansatz_draw/ansatz_Q{self.nqubits}_L{self.nlayers}.txt"
        with open(filename, "a") as file:
            print(self.ansatz.draw(), file=file)

    def get_vparams(self):
        return self.vparams

    def get_test_set(self):
        return self.test

    def get_val_set(self):
        return self.validation

    def get_train_set(self):
        return self.train

    def set_parameters(self, vparams):
        self.vparams = vparams

    def vparams_circuit(self, x):
        """Method which calculates the parameters necessary to embed an image in the circuit.

        Args:
            x: MNIST image.

        Returns:
            A list of parameters.
        """
        blocks = create_blocks(x, self.block_width, self.block_height, self.positions)
        pooling_values = pooling_creator(blocks, self.nqubits, self.pooling)

        # Dimensioni dei blocch: necessario quando ho blocchi di forme diverse
        sizes = block_sizes(self.resize, self.block_width, self.block_height)

        angles = []
        for nlayer in range(self.nlayers):

            # Encoding params
            for j in range(self.nqubits):
                for i in range(sizes[j]):
                    value = blocks[j][i]
                    angle = (
                        self.vparams[nlayer * self.params_1layer + i * 2] * value
                        + self.vparams[nlayer * self.params_1layer + (i * 2 + 1)]
                    )
                    angles.append(angle)

            # Pooling params
            if self.pooling != "no":
                for q in range(self.nqubits):
                    value = pooling_values[q]
                    angle = (
                        self.vparams[
                            nlayer * self.params_1layer + self.n_embed_params + 2 * q
                        ]
                        * value
                        + self.vparams[
                            nlayer * self.params_1layer
                            + self.n_embed_params
                            + (2 * q + 1)
                        ]
                    )
                    angles.append(angle)

        return angles

    def embedding_circuit(self):
        """Method for building the classifier's embedding block.

        Returns:
            Qibo circuit.
        """
        c = Circuit(self.nqubits)
        for j in range(self.nqubits):
            sizes = block_sizes(self.resize, self.block_width, self.block_height)
            for i in range(sizes[j]):
                if i % 3 == 1:
                    c.add(gates.RZ(j, theta=0))
                else:
                    c.add(gates.RY(j, theta=0))

        return c

    def pooling_circuit(self):
        """Method for building the classifier's pooling block.

        Returns:
            Qibo circuit.
        """
        c = Circuit(self.nqubits)
        for q in range(self.nqubits):
            c.add(gates.RX(q, theta=0))
        return c

    def entanglement_circuit(self):
        """Method for building the classifier's entanglement block.

        Returns:
            Qibo circuit.
        """

        c = Circuit(self.nqubits)
        if self.nqubits == 2:
            c.add(gates.CZ(0, 1))

        if self.nqubits == 3:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(1, 2))

        elif self.nqubits == 4:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(2, 3))

        elif self.nqubits == 5:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 3))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(2, 4))
            c.add(gates.CZ(3, 4))

        elif self.nqubits == 6:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 3))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(2, 5))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(4, 5))

        elif self.nqubits == 7:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 4))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 5))
            c.add(gates.CZ(3, 6))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(5, 6))

        elif self.nqubits == 8:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))

        elif self.nqubits == 9:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(7, 8))

        elif self.nqubits == 10:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(8, 9))

        elif self.nqubits == 11:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 6))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 7))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 10))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(9, 10))

        elif self.nqubits == 12:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(10, 11))

        elif self.nqubits == 13:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(11, 12))

        elif self.nqubits == 14:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 10))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 13))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(11, 12))
            c.add(gates.CZ(12, 13))

        elif self.nqubits == 15:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 5))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 6))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 7))
            c.add(gates.CZ(3, 4))
            c.add(gates.CZ(3, 8))
            c.add(gates.CZ(4, 9))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 10))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 11))
            c.add(gates.CZ(7, 8))
            c.add(gates.CZ(7, 12))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 13))
            c.add(gates.CZ(9, 14))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(11, 12))
            c.add(gates.CZ(12, 13))
            c.add(gates.CZ(13, 14))

        elif self.nqubits == 16:
            c.add(gates.CZ(0, 1))
            c.add(gates.CZ(0, 4))
            c.add(gates.CZ(1, 2))
            c.add(gates.CZ(1, 5))
            c.add(gates.CZ(2, 3))
            c.add(gates.CZ(2, 6))
            c.add(gates.CZ(3, 7))
            c.add(gates.CZ(4, 5))
            c.add(gates.CZ(4, 8))
            c.add(gates.CZ(5, 6))
            c.add(gates.CZ(5, 9))
            c.add(gates.CZ(6, 7))
            c.add(gates.CZ(6, 10))
            c.add(gates.CZ(7, 11))
            c.add(gates.CZ(8, 9))
            c.add(gates.CZ(8, 12))
            c.add(gates.CZ(9, 10))
            c.add(gates.CZ(9, 13))
            c.add(gates.CZ(10, 11))
            c.add(gates.CZ(10, 14))
            c.add(gates.CZ(11, 15))
            c.add(gates.CZ(12, 13))
            c.add(gates.CZ(13, 14))
            c.add(gates.CZ(14, 15))
        else:
            raise_error(
                ValueError, "Number of qubits not supported by this architecture"
            )
        return c

    def circuit(self):
        """Method which builds the architecture layer by layer by summing Qibo circuits.

        Returns:
            qibo.Circuit()
        """
        circuit = Circuit(self.nqubits)

        for k in range(self.nlayers):
            # Embedding
            circuit += self.embedding_circuit()

            # Entanglement
            if self.nqubits != 1:
                circuit += self.entanglement_circuit()

            # Pooling
            if self.pooling != "no":
                circuit += self.pooling_circuit()

            # If last layer break the loop
            if k == self.nlayers - 1:
                break

            # Entanglement between layers
            if self.nqubits != 1:
                circuit += self.entanglement_circuit()

        return circuit

    def circuit_output(self, image):

        parameters = self.vparams_circuit(image)
        self.ansatz.set_parameters(parameters)

        result = self.ansatz()
        predicted_state = result.state()

        expectation_value = self.hamiltonian.expectation(predicted_state)
        return expectation_value, predicted_state

    def accuracy(self, data):

        predictions_float, predicted_fids, _ = self.prediction_function(data)
        compare = np.zeros(len(data[0]))
        for i in range(len(data[0])):
            compare[i] = predicted_fids[i] == data[1][i]

        correct = tf.reduce_sum(compare)
        accuracy = correct / len(data[0])

        with open("comparison.txt", "a") as file:
            print("=" * 60)
            print(f"Predictions class {predicted_fids}", file=file)
            print(f"Labels {tf.cast(data[1], tf.int32)}", file=file)
            print(f"Corrected predictions {compare}", file=file)
            print(f"Accuracy {accuracy}", file=file)

        return accuracy

    def prediction_function(self, data):

        predictions_float = []
        predictions_fids = []
        predicted_states = []

        for x in data[0]:
            expectation, predicted_state = self.circuit_output(x)

            # Prediction float is a number between [0, nclasses-1]
            output = (self.nclasses - 1) * (expectation + 1) / 2

            # Prediction fid is the index corresponding to the highest fidelity
            # computed between the predicted state and the targets state
            fids = []
            for j in range(self.nclasses):
                fids.append(fidelity(predicted_state, self.targets[j]))
            label = tf.math.argmax(fids)

            # Append
            predictions_float.append(output)
            predictions_fids.append(label)
            predicted_states.append(predicted_state)

        return predictions_float, predictions_fids, predicted_states

    def loss_fidelity_weighted(self, data, labels):
        loss = 0.0
        for i in range(self.batch_size):
            _, pred_state = self.circuit_output(data[i])
            for j in range(self.nclasses):
                f_1 = fidelity(pred_state, tf.gather(self.targets, j)) ** 2
                f_2 = (
                    fidelity(
                        tf.gather(self.targets, tf.cast(labels[i], dtype=tf.int32)),
                        tf.gather(self.targets, j),
                    )
                    ** 2
                )
                loss += (tf.gather(self.alpha, j) * f_1 - f_2) ** 2
        return loss

    def loss_crossentropy(self, x_batch, y_batch):

        outputs = []
        for i in range(self.batch_size):
            if i > len(x_batch):
                raise_error(
                    ValueError,
                    f"Index {i} is out of bounds for array of length {len(x_batch)}",
                )

            expectation_value, _ = self.circuit_output(x_batch[i])
            outputs.append((expectation_value + 1) / 2)

        outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)
        loss = tf.keras.losses.BinaryCrossentropy()(y_batch, outputs)

        return loss

    def loss_fidelity(self, data, labels):
        cf = 0.0
        for i in range(self.batch_size):
            if i > len(data):
                raise_error(
                    ValueError,
                    f"Index {i} is out of bounds for array of length {len(data)}",
                )
            _, pred_state = self.circuit_output(data[i])
            label = tf.gather(labels, i)
            label = tf.cast(label, tf.int32)
            true_state = tf.gather(self.targets, label)
            cf += 1 - fidelity(pred_state, true_state) ** 2

        return cf

    def trainability(self, image, label):

        with tf.GradientTape() as tape:
            exp, _ = self.circuit_output(image)
            output = (exp + 1) / 2

            # Cast float64 --> float32
            output = tf.cast(output, dtype=tf.float32)

            # Reshape per fare reduction
            output = tf.reshape(output, (1, 1))
            label = tf.reshape(label, (1, 1))

            loss = tf.keras.losses.BinaryCrossentropy()(label, output)
        grads = tape.gradient(loss, self.vparams)
        grads = tf.math.abs(grads)

        return grads, loss

    # @tf.function
    def train_step(self, x_batch, y_batch, optimizer):
        """Evaluate loss function on one train batch.

        Args:
            batch_size (int): number of samples in one training batch.
            encoder (qibo.models.Circuit): variational quantum circuit.
            params (tf.Variable): parameters of the circuit.
            vector (tf.Tensor): train sample, in the form of 1d vector.

        Returns:
            loss (tf.Variable): average loss of the training batch.
        """

        # Cercare come tracciare dove il tracker di tensorflow perde traccia
        # e a causa di cio non calcola il gradiente
        if self.loss == "crossentropy":
            with tf.GradientTape() as tape:
                loss = self.loss_crossentropy(x_batch, y_batch)
                print(f"Loss: {loss}")
            grads = tape.gradient(loss, self.vparams)

            with open("grad_cross.txt", "a") as file:
                print("=" * 60, file=file)
                print(f"Gradients {grads}", file=file)
                print(f"Loss {loss}", file=file)

            grads = tf.math.real(grads)
            optimizer.apply_gradients(zip([grads], [self.vparams]))

            return loss

        if self.loss == "fidelity":
            with tf.GradientTape() as tape:
                loss = self.loss_fidelity(x_batch, y_batch)
            grads = tape.gradient(loss, self.vparams)
            grads = tf.math.real(grads)

            with open("grad_fid.txt", "a") as file:
                print("=" * 60, file=file)
                print(f"Gradients {grads}", file=file)
                print(f"Loss {loss}", file=file)

            optimizer.apply_gradients(zip([grads], [self.vparams]))
            return loss

        if self.loss == "weighted_fidelity":
            with tf.GradientTape() as tape:
                loss = self.loss_fidelity_weighted(x_batch, y_batch)
            trainable_variables = [self.vparams, self.alpha]
            grads = tape.gradient(loss, trainable_variables)
            grads = [tf.math.real(g) for g in grads]

            with open("grad_w_fid.txt", "a") as file:
                print("=" * 60, file=file)
                print(f"Gradients {grads}", file=file)
                print(f"Loss {loss}", file=file)

            optimizer.apply_gradients(zip(grads, trainable_variables))

            return loss

    def training_loop(self):
        """Method to train the classifier.

        Args:
            No

        Returns:
            No
        """
        trained_params = np.zeros((self.nepochs, self.n_params))
        history_train_loss = np.zeros(self.nepochs)
        history_val_loss = np.zeros(self.nepochs)
        history_train_accuracy = np.zeros(self.nepochs)
        history_val_accuracy = np.zeros(self.nepochs)

        number_of_batches = math.ceil(self.training_size / self.batch_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        loss = 0.0
        for epoch in range(self.nepochs):
            self.train = shuffle(self.train)
            print(f"Epoch {epoch}")
            for i in range(number_of_batches):
                print(f"Batch {i}")
                loss = self.train_step(
                    self.train[0][i * self.batch_size : (i + 1) * self.batch_size],
                    self.train[1][i * self.batch_size : (i + 1) * self.batch_size],
                    optimizer,
                )

                with open("history.txt", "a") as file:
                    print(f"Epoch {epoch}", file=file)
                    print(f"Batch {i}", file=file)
                    print(f"Loss: {loss}", file=file)
                    print(f"Parametri: {self.vparams[0:20]}", file=file)

            trained_params[epoch] = self.vparams

            history_train_loss[epoch] = loss
            history_val_loss[epoch] = self.loss_fidelity(
                self.validation[0], self.validation[1]
            )

            history_train_accuracy[epoch] = self.accuracy(self.train)
            history_val_accuracy[epoch] = self.accuracy(self.validation)

            with open("epochs.txt", "a") as file:
                print(f"Epoch {epoch}", file=file)
                print(f"Accuracy training {history_train_accuracy[epoch]}", file=file)
                print(f"Accuracy validation {history_val_accuracy[epoch]}", file=file)

        return (
            trained_params[-1],
            history_train_loss,
            history_val_loss,
            history_train_accuracy,
            history_val_accuracy,
        )
