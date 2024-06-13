import numpy as np
import tensorflow as tf
import math
from qibo import Circuit, gates, hamiltonians, set_backend
from data import initialize_data, pooling_creator, block_creator, shuffle
from help_functions import fidelity, create_target, number_params, create_hamiltonian


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

        # IMAGE
        self.train, self.test, self.validation = initialize_data(
            nclasses,
            self.training_size,
            self.test_size,
            self.validation_size,
            resize,
        )
        self.block_width = block_width
        self.block_height = block_height

        # CIRCUIT
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.pooling = pooling
        self.n_embed_params = 2 * self.block_width * self.block_height * self.nqubits
        self.params_1layer = number_params(
            self.n_embed_params, self.nqubits, self.pooling
        )
        self.n_params = self.params_1layer * nlayers
        self.vparams = tf.Variable(tf.random.normal((self.n_params,)), dtype=tf.float32)
        self.hamiltonian = create_hamiltonian(self.nqubits)
        self.ansatz = self.circuit()

    def get_test_set(self):
        return self.test

    def vparams_circuit(self, x):
        """Method which calculates the parameters necessary to embed an image in the circuit.

        Args:
            x: MNIST image.

        Returns:
            A list of parameters.
        """

        # Embedding angles
        blocks = block_creator(x, self.block_height, self.block_width)

        # Pooling angles
        pooling_values = pooling_creator(blocks, self.nqubits, self.pooling)

        angles = []
        for nlayer in range(self.nlayers):

            # Encoding params
            for j in range(self.nqubits):
                for i in range(self.block_width * self.block_height):
                    # print(f"i {i}")
                    # print(f"Before {blocks[j][i]}")
                    value = blocks[j][i]
                    # print(f"After {value}")
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
            for i in range(self.block_width * self.block_height):
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
        for q in range(0, self.nqubits - 1, 2):
            c.add(gates.CNOT(q, q + 1))
        for q in range(1, self.nqubits - 2, 2):
            c.add(gates.CNOT(q, q + 1))
        c.add(gates.CNOT(self.nqubits - 1, 0))
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
        if self.nclasses == 2:
            predicted_classes = tf.round(predictions_float)
            correct_predictions = tf.equal(tf.cast(predicted_classes, tf.int32), tf.cast(data[1], tf.int32))
            print(tf.cast(predicted_classes, tf.int32))
            print(tf.cast(data[1], tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            with open("comparison.txt", "a") as file:
                print("="*60)
                predictions_float_numpy = [tensor.numpy() for tensor in predictions_float]
                print(f"Predictions float {predictions_float_numpy}", file=file)
                print(f"Predictions class {predicted_classes}", file=file)
                print(f"Labels {data[1]}", file=file)
                print(f"Accuracy {accuracy}", file=file)

            return accuracy
        else:

            compare = np.zeros(len(data[0]))
            for i in range(len(data[0])):
                compare[i] = predicted_fids[i] == data[1][i]

            correct = tf.reduce_sum(compare)
            accuracy = correct / len(data[0])
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

    def loss_crossentropy(self, data, labels):
        expectation_values = []

        for i in range(self.batch_size):
            expectation_value, _ = self.circuit_output(data[i])
            expectation_values.append(expectation_value)
        
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        l = loss(expectation_values, labels)
        return l

    def loss_fidelity(self, data, labels):
        cf = 0.0
        for i in range(self.batch_size):
            _, pred_state = self.circuit_output(data[i])
            label = tf.gather(labels, i)
            label = tf.cast(label, tf.int32)
            true_state = tf.gather(self.targets, label)
            cf += (1 - fidelity(pred_state, true_state)) ** 2

        return cf

    @tf.function
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
        if (self.nclasses == 2) and (self.loss == "crossentropy"):
            with tf.GradientTape() as tape:
                loss = self.loss_crossentropy(x_batch, y_batch)
            grads = tape.gradient(loss, self.vparams)
            grads = tf.math.real(grads)
            optimizer.apply_gradients(zip([grads], [self.vparams]))
            return loss
        
        if (self.loss == "fidelity"):
            with open("history.txt", "a") as file:
                print(f"Elements in batch (train-step): ({len(x_batch), len(y_batch)})", file=file)
            with tf.GradientTape() as tape:
                loss = self.loss_fidelity(x_batch, y_batch)
            grads = tape.gradient(loss, self.vparams)
            grads = tf.math.real(grads)
            optimizer.apply_gradients(zip([grads], [self.vparams]))
            with open("history.txt", "a") as file:
                print(f"Loss (train-step): ({loss})", file=file)
                print(f"Grad (train-step): ({grads})", file=file)
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        loss = 0.0
        for epoch in range(self.nepochs):
            self.train = shuffle(self.train)
            print(f"Epoch {epoch}")
            for i in range(number_of_batches):
                print(f"Batch {i}")

                with open("history.txt", "a") as file:
                    print(f"Epoch {epoch}", file=file) 
                    print(f"Batch {i}", file=file) 
                    
                loss = self.train_step(
                    self.train[0][i * self.batch_size : (i + 1) * self.batch_size],
                    self.train[1][i * self.batch_size : (i + 1) * self.batch_size],
                    optimizer,
                )

                with open("history.txt", "a") as file:
                    print(f"Elements in batch (training-loop) ({len(self.train[0][i * self.batch_size : (i + 1) * self.batch_size])}, {len(self.train[1][i * self.batch_size : (i + 1) * self.batch_size])})", file=file)
                    print(f"Parametri: {self.vparams[0:20]}", file=file)
                

            trained_params[epoch] = self.vparams
            history_train_loss[epoch] = loss

            if (self.nclasses == 2) and (self.loss == "crossentropy"):
                history_val_loss[epoch] = self.loss_crossentropy(
                    self.validation[0], self.validation[1]
                )
            else:
                history_val_loss[epoch] = self.loss_fidelity(
                    self.validation[0], self.validation[1]
                )

            history_train_accuracy[epoch] = self.accuracy(self.train)
            history_val_accuracy[epoch] = self.accuracy(self.validation)
            

            with open("epochs.txt", "a") as file:
                print(f"Epoch {epoch}", file=file)

        return (
            trained_params[-1],
            history_train_loss,
            history_val_loss,
            history_train_accuracy,
            history_val_accuracy,
        )
