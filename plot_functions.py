import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from qibo import hamiltonians
from qibo.symbols import X, Y, Z

LOCAL_FOLDER = Path(__file__).parent


def plot_predictions(predictions, x_data, y_data, name, rows, columns):
    """Function to plot 25 classifier's predictions.
    It will plot a table 5x5 of MNIST images: each image will have above the classifier's prediction.

    Args:
        predictions (float list): Classifier's predictions.
        x_data: MNIST images.
        y_data (int list): True labels of the images.
        name (str): Name of the plot.

    Returns:
        Predictions table plot.
    """

    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 12))

    for i in range(rows):
        for j in range(columns):
            ax[i][j].imshow(x_data[i * columns + j], cmap="gray")

            is_correct = predictions[i * columns + j] == y_data[i * columns + j]
            title_color = "green" if is_correct else "red"
            ax[i][j].set_title(
                f"Prediction: {predictions[i * columns + j]}", color=title_color
            )
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    plt.savefig(name)
    plt.close()


def plot_sphere(qubit, layers, labels, label_states, final_states_b, final_states_a):
    file_path = LOCAL_FOLDER / "plots"
    color = ["r", "b", "g", "magenta", "yellow", "grayx"]

    # Sphere before training
    sfera = Bloch()
    for x, y in zip(final_states_b, labels):
        sfera.add_state(x, "point", color[y])

    for x in label_states:
        sfera.add_state(x, "vector", "black")

    name = f"q{qubit}-l{layers}-sphere-before.pdf"
    sfera.plot(file_path / name, save=True)

    # Sphere after training
    sfera = Bloch()
    for x, y in zip(final_states_a, labels):
        sfera.add_state(x, "point", color[y])

    for x in label_states:
        sfera.add_state(x, "vector", "black")

    name = file_path / f"q{qubit}-l{layers}-sphere-after.pdf"
    sfera.plot(file_path / name, save=True)


def plot_loss_accuracy(qubit, layers, epochs, train_loss, val_loss, train_acc, val_acc):
    file_path = LOCAL_FOLDER / "plots"
    WIDTH = 0.5

    epochs = np.arange(0, epochs, 1)

    fig, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=(10 * WIDTH * 3 / 2, 10 * WIDTH * 6 / 8)
    )

    # Plot sul primo asse
    ax1.plot(
        epochs,
        train_loss,
        label="Training",
        alpha=0.8,
        lw=1.5,
        ls="-",
        color="royalblue",
    )
    ax1.plot(
        epochs,
        val_loss,
        label="Validation",
        alpha=0.8,
        lw=1.5,
        ls="-",
        color="coral",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_title("Loss")
    ax1.legend()

    # Plot sul secondo asse
    ax2.plot(epochs, train_acc, label="Training", alpha=0.8,
        lw=1.5,
        ls="-", color="royalblue")
    ax2.plot(epochs, val_acc, label="Validation", alpha=0.8,
        lw=1.5,
        ls="-", color="coral")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Accuracy")
    ax2.legend()

    # fig.suptitle("4 qubits: 3 layers", fontsize=12)
    plt.tight_layout()
    name = f"q{qubit}-" + f"l{layers}" + "-loss-accuracy.pdf"
    plt.savefig(file_path / name, bbox_inches="tight")


class Bloch:
    def __init__(self, state=None, mode=None, color=None):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d", elev=30)
        self.empty_sphere()

    def add_state(self, state, mode, color):
        x, y, z = 0, 0, 0
        if state[0] == 1 and state[0] == 0:
            z = 1
        elif state[0] == 0 and state[0] == 1:
            z = -1
        else:
            x, y, z = self.cartesian_coordinates(state)

        if mode == "vector":
            vector = np.array([x, y, z])
            self.ax.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=0.1)

        if mode == "point":
            tpoint = [1, np.pi / 2 + np.pi / 4, np.pi + np.pi / 4]
            hpoint = [1, np.pi / 4, np.pi / 4]
            tx, ty, tz = self.spherical_to_cartesian(tpoint[0], tpoint[1], tpoint[2])
            hx, hy, hz = self.spherical_to_cartesian(hpoint[0], hpoint[1], hpoint[2])

            distance = np.sqrt((x - tx) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)
            h_distance = np.sqrt((hx - tx) ** 2 + (hy - ty) ** 2 + (hz - tz) ** 2)

            """
            scale_factor = 1
            alpha = np.exp(-scale_factor * (h_distance - distance))
            alpha = max(0, min(alpha, 1))
            """

            self.ax.scatter(x, y, z, color=color, s=10)

    def empty_sphere(self):
        "Function to create an empty sphere."

        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        self.ax.plot_surface(x, y, z, color="lavenderblush", alpha=0.2)

        # plot circular curves over the surface
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)

        self.ax.plot(x, y, z, color="black", alpha=0.25)
        self.ax.plot(z, x, y, color="black", alpha=0.25)
        self.ax.plot(y, z, x, color="black", alpha=0.1)

        # add axis lines
        line = np.linspace(-1, 1, 100)
        zeros = np.zeros_like(line)

        self.ax.plot(line, zeros, zeros, color="black", alpha=0.3)
        self.ax.plot(zeros, line, zeros, color="black", alpha=0.3)
        self.ax.plot(zeros, zeros, line, color="black", alpha=0.3)

        self.ax.text(1.2, 0, 0, "y", color="black", fontsize=15, ha="center")
        self.ax.text(0, -1.2, 0, "x", color="black", fontsize=15, ha="center")
        self.ax.text(0, 0, 1.2, r"$|0\rangle$", color="black", fontsize=15, ha="center")
        self.ax.text(
            0, 0, -1.2, r"$|1\rangle$", color="black", fontsize=15, ha="center"
        )

        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([-0.8, 0.8])

    def add_label_states(self, nclasses, solid=False):
        "Function which returns the label states for classification purposes."
        if nclasses == 2:
            targets = tf.constant(
                [np.array([1, 0], dtype="complex"), np.array([0, 1], dtype="complex")]
            )
            for target in targets:
                self.__call__(target, mode="vector", color="Black")

        if nclasses == 3:

            vertices_cartesian = [
                (np.sin(2 * np.pi / 3), 0, np.cos(2 * np.pi / 3)),
                (np.sin(4 * np.pi / 3), np.sin(np.pi / 3), np.sin(4 * np.pi / 3)),
            ]
            theta = np.zeros(2)
            phi = np.zeros(2)
            for i, vertex in enumerate(vertices_cartesian):
                t, p = self.cartesian_to_spherical(vertex[0], vertex[1], vertex[2])
                theta[i] = t
                phi[i] = p

            targets = tf.constant(
                [
                    np.array([1, 0], dtype="complex"),
                    np.array(
                        [
                            np.cos(theta[0] / 2),
                            np.sin(theta[0] / 2) * np.exp(1j * phi[0]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[1] / 2),
                            np.sin(theta[1] / 2) * np.exp(1j * phi[1]),
                        ],
                        dtype="complex",
                    ),
                ],
                dtype=tf.complex64,
            )
            for target in targets:
                self.add_state(target, mode="vector", color="Black")

        if nclasses == 4:
            if solid == True:
                self.solids(nclasses)

            vertices_cartesian = [
                (np.sqrt(8 / 9), 0, -1 / 3),
                (-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3),
                (-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3),
            ]

            angle = 5 * np.pi / 4
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            vertices_rotated = np.dot(vertices_cartesian, rotation_matrix)

            theta = np.zeros(3)
            phi = np.zeros(3)
            for i, vertex in enumerate(vertices_rotated):
                t, p = self.cartesian_to_spherical(vertex[0], vertex[1], vertex[2])
                theta[i] = t
                phi[i] = p

            targets = tf.constant(
                [
                    np.array(
                        [
                            np.cos(theta[0] / 2),
                            np.sin(theta[0] / 2) * np.exp(1j * phi[0]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[1] / 2),
                            np.sin(theta[1] / 2) * np.exp(1j * phi[1]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[2] / 2),
                            np.sin(theta[2] / 2) * np.exp(1j * phi[2]),
                        ],
                        dtype="complex",
                    ),
                    np.array([1, 0], dtype="complex"),
                ]
            )
            for target in targets:
                self.add_state(target, mode="vector", color="Black")

        if nclasses == 6:
            if solid == True:
                self.solids(nclasses)

            theta = [0, np.pi, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
            phi = [0, 0, 0, np.pi / 2, np.pi, np.pi / 2 * 3]

            targets = tf.constant(
                [
                    np.array(
                        [
                            np.cos(theta[0] / 2),
                            np.sin(theta[0] / 2) * np.exp(1j * phi[0]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[1] / 2),
                            np.sin(theta[1] / 2) * np.exp(1j * phi[1]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[2] / 2),
                            np.sin(theta[2] / 2) * np.exp(1j * phi[2]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[3] / 2),
                            np.sin(theta[3] / 2) * np.exp(1j * phi[3]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[4] / 2),
                            np.sin(theta[4] / 2) * np.exp(1j * phi[4]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[5] / 2),
                            np.sin(theta[5] / 2) * np.exp(1j * phi[5]),
                        ],
                        dtype="complex",
                    ),
                ],
            )
            for target in targets:
                self.add_state(target, mode="vector", color="Black")

        if nclasses == 8:
            if solid == True:
                self.solids(nclasses)

            theta = [
                np.pi / 4,
                np.pi / 4,
                3 * np.pi / 4,
                3 * np.pi / 4,
                -np.pi / 4,
                -np.pi / 4,
                -3 * np.pi / 4,
                -3 * np.pi / 4,
            ]
            phi = [
                np.pi / 4,
                3 * np.pi / 4,
                3 * np.pi / 4,
                np.pi / 4,
                np.pi / 4,
                3 * np.pi / 4,
                3 * np.pi / 4,
                np.pi / 4,
            ]
            targets = tf.constant(
                [
                    np.array(
                        [
                            np.cos(theta[0] / 2),
                            np.sin(theta[0] / 2) * np.exp(1j * phi[0]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[1] / 2),
                            np.sin(theta[1] / 2) * np.exp(1j * phi[1]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[2] / 2),
                            np.sin(theta[2] / 2) * np.exp(1j * phi[2]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[3] / 2),
                            np.sin(theta[3] / 2) * np.exp(1j * phi[3]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[4] / 2),
                            np.sin(theta[4] / 2) * np.exp(1j * phi[4]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[5] / 2),
                            np.sin(theta[5] / 2) * np.exp(1j * phi[5]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[6] / 2),
                            np.sin(theta[6] / 2) * np.exp(1j * phi[6]),
                        ],
                        dtype="complex",
                    ),
                    np.array(
                        [
                            np.cos(theta[7] / 2),
                            np.sin(theta[7] / 2) * np.exp(1j * phi[7]),
                        ],
                        dtype="complex",
                    ),
                ],
            )
            for target in targets:
                self.add_state(target, mode="vector", color="Black")

        if nclasses == 10:
            targets = 0

        return targets

    def solids(self, nclasses):

        if nclasses == 4:
            vertices_cartesian = [
                (np.sqrt(8 / 9), 0, -1 / 3),
                (-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3),
                (-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3),
                (0, 0, 1),
            ]

            angle = 5 * np.pi / 4
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            vertices_rotated = np.dot(vertices_cartesian, rotation_matrix)

            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 1), (3, 2)]

            for i, (x, y, z) in enumerate(vertices_rotated):
                self.ax.scatter(x, y, z, color="blue")

            for edge in edges:
                x_values = [vertices_rotated[edge[0]][0], vertices_rotated[edge[1]][0]]
                y_values = [vertices_rotated[edge[0]][1], vertices_rotated[edge[1]][1]]
                z_values = [vertices_rotated[edge[0]][2], vertices_rotated[edge[1]][2]]
                self.ax.plot(x_values, y_values, z_values, color="blue")

        if nclasses == 6:
            vertices_cartesian = [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]

            edges = [
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 4),
                (2, 5),
                (3, 4),
                (3, 5),
            ]

            for i, (x, y, z) in enumerate(vertices_cartesian):
                self.ax.scatter(x, y, z, color="blue")

            for edge in edges:
                x_values = [
                    vertices_cartesian[edge[0]][0],
                    vertices_cartesian[edge[1]][0],
                ]
                y_values = [
                    vertices_cartesian[edge[0]][1],
                    vertices_cartesian[edge[1]][1],
                ]
                z_values = [
                    vertices_cartesian[edge[0]][2],
                    vertices_cartesian[edge[1]][2],
                ]
                self.ax.plot(x_values, y_values, z_values, color="blue")

        if nclasses == 8:
            vertices_spherical = [
                (1, np.pi / 4, np.pi / 4),
                (1, np.pi / 4, 3 * np.pi / 4),
                (1, 3 * np.pi / 4, 3 * np.pi / 4),
                (1, 3 * np.pi / 4, np.pi / 4),
                (1, -np.pi / 4, np.pi / 4),
                (1, -np.pi / 4, 3 * np.pi / 4),
                (1, -3 * np.pi / 4, 3 * np.pi / 4),
                (1, -3 * np.pi / 4, np.pi / 4),
            ]
            vertices_cartesian = [
                self.spherical_to_cartesian(*vertex) for vertex in vertices_spherical
            ]

            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]

            for vertex in vertices_cartesian:
                self.ax.scatter(*vertex, color="blue")

            for edge in edges:
                x_values = [
                    vertices_cartesian[edge[0]][0],
                    vertices_cartesian[edge[1]][0],
                ]
                y_values = [
                    vertices_cartesian[edge[0]][1],
                    vertices_cartesian[edge[1]][1],
                ]
                z_values = [
                    vertices_cartesian[edge[0]][2],
                    vertices_cartesian[edge[1]][2],
                ]
                self.ax.plot(x_values, y_values, z_values, color="blue")

    def plot(self, name=None, save=False):
        "Function to plot and save the sphere."

        self.ax.set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()

        if save:
            plt.savefig(name)
        else:
            plt.show()

    def spherical_to_cartesian(self, r, theta, phi):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z

    def cartesian_to_spherical(self, x, y, z):
        phi = np.arctan2(y, x)
        theta = np.arccos(z)
        return theta, phi

    def cartesian_coordinates(self, state):
        "Function to determine the coordinates of a qubit in the sphere."

        sigma_X = hamiltonians.SymbolicHamiltonian(X(0))
        sigma_Y = hamiltonians.SymbolicHamiltonian(Y(0))
        sigma_Z = hamiltonians.SymbolicHamiltonian(Z(0))

        x = sigma_X.expectation(state)
        y = sigma_Y.expectation(state)
        z = sigma_Z.expectation(state)
        return x, y, z
