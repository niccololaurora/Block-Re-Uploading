import numpy as np
import tensorflow as tf
from qibo import hamiltonians
from qibo.symbols import Z, I
from qibo.config import raise_error


def create_target(nclasses):
    "Function which returns the label states for classification purposes."
    if nclasses == 2:
        targets = tf.constant(
            [np.array([1, 0], dtype="complex"), np.array([0, 1], dtype="complex")]
        )

    if nclasses == 3:

        vertices_cartesian = [
            (np.sin(2 * np.pi / 3), 0, np.cos(2 * np.pi / 3)),
            (np.sin(4 * np.pi / 3), np.sin(np.pi / 3), np.sin(4 * np.pi / 3)),
        ]
        theta = np.zeros(2)
        phi = np.zeros(2)
        for i, vertex in enumerate(vertices_cartesian):
            t, p = cartesian_to_spherical(vertex[0], vertex[1], vertex[2])
            theta[i] = t
            phi[i] = p

        targets = tf.constant(
            [
                np.array([1, 0], dtype="complex"),
                np.array(
                    [np.cos(theta[0] / 2), np.sin(theta[0] / 2) * np.exp(1j * phi[0])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[1] / 2), np.sin(theta[1] / 2) * np.exp(1j * phi[1])],
                    dtype="complex",
                ),
            ],
            dtype=tf.complex64,
        )

    if nclasses == 4:

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
            t, p = cartesian_to_spherical(vertex[0], vertex[1], vertex[2])
            theta[i] = t
            phi[i] = p

        targets = tf.constant(
            [
                np.array(
                    [np.cos(theta[0] / 2), np.sin(theta[0] / 2) * np.exp(1j * phi[0])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[1] / 2), np.sin(theta[1] / 2) * np.exp(1j * phi[1])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[2] / 2), np.sin(theta[2] / 2) * np.exp(1j * phi[2])],
                    dtype="complex",
                ),
                np.array([1, 0], dtype="complex"),
            ]
        )

    if nclasses == 6:

        theta = [0, np.pi, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
        phi = [0, 0, 0, np.pi / 2, np.pi, np.pi / 2 * 3]

        targets = tf.constant(
            [
                np.array(
                    [np.cos(theta[0] / 2), np.sin(theta[0] / 2) * np.exp(1j * phi[0])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[1] / 2), np.sin(theta[1] / 2) * np.exp(1j * phi[1])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[2] / 2), np.sin(theta[2] / 2) * np.exp(1j * phi[2])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[3] / 2), np.sin(theta[3] / 2) * np.exp(1j * phi[3])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[4] / 2), np.sin(theta[4] / 2) * np.exp(1j * phi[4])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[5] / 2), np.sin(theta[5] / 2) * np.exp(1j * phi[5])],
                    dtype="complex",
                ),
            ],
        )

    if nclasses == 8:

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
                    [np.cos(theta[0] / 2), np.sin(theta[0] / 2) * np.exp(1j * phi[0])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[1] / 2), np.sin(theta[1] / 2) * np.exp(1j * phi[1])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[2] / 2), np.sin(theta[2] / 2) * np.exp(1j * phi[2])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[3] / 2), np.sin(theta[3] / 2) * np.exp(1j * phi[3])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[4] / 2), np.sin(theta[4] / 2) * np.exp(1j * phi[4])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[5] / 2), np.sin(theta[5] / 2) * np.exp(1j * phi[5])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[6] / 2), np.sin(theta[6] / 2) * np.exp(1j * phi[6])],
                    dtype="complex",
                ),
                np.array(
                    [np.cos(theta[7] / 2), np.sin(theta[7] / 2) * np.exp(1j * phi[7])],
                    dtype="complex",
                ),
            ],
        )

    if nclasses == 10:
        targets = 0

    return targets


def cartesian_to_spherical(x, y, z):
    phi = np.arctan2(y, x)
    theta = np.arccos(z)
    return theta, phi


def create_hamiltonian(nqubits, local=False):
    """Method for building the hamiltonian used to evaluate expectation values.

    Returns:
        qibo.hamiltonians.SymbolicHamiltonian()
    """

    # Global hamiltonian
    if local == False:
        ham = I(0)
        for k in range(nqubits):
            ham *= Z(k)
        hamiltonian = hamiltonians.SymbolicHamiltonian(ham)
        return hamiltonian

    # Local hamiltonian
    else:
        ham = Z(0)
        for k in range(nqubits):
            ham *= I(k)
        hamiltonian = hamiltonians.SymbolicHamiltonian(ham)
        return hamiltonian


def fidelity(state1, state2):
    """
    Args: two vectors
    Output: inner product of the two vectors **2
    """
    norm = tf.math.abs(tf.reduce_sum(tf.math.conj(state1) * state2))
    return norm


def number_params(n_embed_params, nqubits, pooling):

    if pooling != "no":
        return 2 * nqubits + n_embed_params
    else:
        return n_embed_params


def block_sizes(resize, width, height):
    sizes = []

    for w, h in zip(width, height):
        size = w * h
        sizes.append(size)

    return sizes


def blocks_details(size, nqubits):
    if size == 4:
        if nqubits == 1:
            block_width = [size]
            block_height = [size]
            positions = [
                (0, 0),
            ]
            return block_width, block_height, positions

        elif nqubits == 2:
            block_width = [2, 2]
            block_height = [4] * 2
            positions = [
                (0, 0),
                (0, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 3:
            block_width = [2, 1, 1]
            block_height = [4] * 3
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 4:
            block_width = [2] * 4
            block_height = [2] * 4
            positions = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 5:
            block_width = [1, 1, 2, 2, 2]
            block_height = [2] * 5
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 6:
            block_width = [1, 1, 2, 1, 1, 2]
            block_height = [2] * 6
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 7:
            block_width = [1] * 6 + [2]
            block_height = [2] * 7
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 8:
            block_width = [1] * 8
            block_height = [2] * 8
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 9:
            block_width = [4] + [1] * 8
            block_height = [1] * 5 + [2] * 4
            positions = [
                (0, 0),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 10:
            block_width = [4] + [1] * 8 + [4]
            block_height = [1] * 10
            positions = [
                (0, 0),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
            ]
            return block_width, block_height, positions

        elif nqubits == 11:
            block_width = [1, 3] + [1] * 8 + [4]
            block_height = [1] * 11
            positions = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
            ]
            return block_width, block_height, positions

        elif nqubits == 12:
            block_width = [1, 3] + [1] * 8 + [1, 3]
            block_height = [1] * 12
            positions = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 13:
            block_width = [1, 1, 2] + [1] * 8 + [1, 3]
            block_height = [1] * 13
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 14:
            block_width = [1, 1, 2] + [1] * 8 + [1, 1, 2]
            block_height = [1] * 14
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 15:
            block_width = [1] * 14 + [2]
            block_height = [1] * 15
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 16:
            block_width = [1] * 16
            block_height = [1] * 16
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
            ]
            return block_width, block_height, positions

        else:
            raise_error(
                ValueError, "Number of qubits not supported by this architecture"
            )

    if size == 8:
        if nqubits == 1:
            block_width = [8]
            block_height = [8]
            positions = [
                (0, 0),
            ]
            return block_width, block_height, positions
        elif nqubits == 2:
            block_width = [4] * 2
            block_height = [8] * 2
            positions = [
                (0, 0),
                (0, 1),
            ]
            return block_width, block_height, positions
        elif nqubits == 3:
            block_width = [3, 3, 2]
            block_height = [8] * 3
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
            ]
            return block_width, block_height, positions
        elif nqubits == 4:
            block_width = [4] * 4
            block_height = [4] * 4
            positions = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions
        elif nqubits == 5:
            block_width = [3, 3, 2, 4, 4]
            block_height = [4] * 5
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 6:
            block_width = [3, 3, 2, 3, 3, 2]
            block_height = [4] * 6
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 7:
            block_width = [2, 2, 2, 2, 3, 3, 2]
            block_height = [4] * 7
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 8:
            block_width = [2] * 8
            block_height = [4] * 8
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 9:
            block_width = [1, 2, 2, 2, 1, 2, 2, 2, 2]
            block_height = [4] * 9
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 10:
            block_width = [1, 2, 2, 2, 1, 1, 2, 2, 2, 1]
            block_height = [4] * 10
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
            ]
            return block_width, block_height, positions

        elif nqubits == 11:
            block_width = [1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1]
            block_height = [4] * 11
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
            ]
            return block_width, block_height, positions

        elif nqubits == 12:
            block_width = [2] * 12
            block_height = [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 13:
            block_width = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            block_height = [3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 14:
            block_width = [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2]
            block_height = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 15:
            block_width = [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2]
            block_height = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2]
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
            ]
            return block_width, block_height, positions

        elif nqubits == 16:
            block_width = [2] * 16
            block_height = [2] * 16
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
            ]
            return block_width, block_height, positions

        else:
            raise_error(
                ValueError, "Number of qubits not supported by this architecture"
            )

    if size == 12:
        if nqubits == 1:
            block_width = [12]
            block_height = [12]
            positions = [
                (0, 0),
            ]
            return block_width, block_height, positions

        elif nqubits == 2:
            block_width = [6] * 2
            block_height = [12] * 2
            positions = [
                (0, 0),
                (0, 1),
            ]
            return block_width, block_height, positions
        elif nqubits == 3:
            block_width = [4] * 3
            block_height = [12] * 3
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
            ]
            return block_width, block_height, positions
        elif nqubits == 4:
            block_width = [6] * 4
            block_height = [6] * 4
            positions = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions
        elif nqubits == 5:
            block_width = [4, 4, 4, 6, 6]
            block_height = [6] * 5
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions
        elif nqubits == 6:
            block_width = [4] * 6
            block_height = [6] * 6
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
            return block_width, block_height, positions
        elif nqubits == 7:
            block_width = [3, 3, 3, 3, 4, 4, 4]
            block_height = [6] * 7
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
            return block_width, block_height, positions

        elif nqubits == 8:
            block_width = [3] * 8
            block_height = [6] * 8
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
            ]
            return block_width, block_height, positions

        elif nqubits == 9:
            block_width = [4] * 9
            block_height = [4] * 9
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            ]
            return block_width, block_height, positions

        else:
            raise_error(
                ValueError, "Number of qubits not supported by this architecture"
            )

    if size == 14:
        if nqubits == 1:
            block_width = [14]
            block_height = [14]
            positions = [
                (0, 0),
            ]
            return block_width, block_height, positions

        elif nqubits == 2:
            block_width = [7] * 2
            block_height = [14] * 2
            positions = [
                (0, 0),
                (0, 1),
            ]
            return block_width, block_height, positions
        elif nqubits == 3:
            block_width = [5, 5, 4]
            block_height = [14] * 3
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
            ]
            return block_width, block_height, positions
        elif nqubits == 4:
            block_width = [7] * 4
            block_height = [7] * 4
            positions = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 5:
            block_width = [5, 5, 4, 7, 7]
            block_height = [7] * 5
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 6:
            block_width = [5, 5, 4, 5, 5, 4]
            block_height = [7] * 6
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
            ]
            return block_width, block_height, positions

        else:
            raise_error(
                ValueError, "Number of qubits not supported by this architecture"
            )

    if size == 16:
        if nqubits == 1:
            block_width = [16]
            block_height = [16]
            positions = [
                (0, 0),
            ]
            return block_width, block_height, positions

        elif nqubits == 2:
            block_width = [8] * 2
            block_height = [16] * 2
            positions = [
                (0, 0),
                (0, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 3:
            block_width = [5, 5, 6]
            block_height = [16] * 3
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
            ]
            return block_width, block_height, positions

    if size == 18:
        if nqubits == 1:
            block_width = [18]
            block_height = [18]
            positions = [
                (0, 0),
            ]
            return block_width, block_height, positions

        elif nqubits == 2:
            block_width = [9] * 2
            block_height = [18] * 2
            positions = [
                (0, 0),
                (0, 1),
            ]
            return block_width, block_height, positions

        elif nqubits == 3:
            block_width = [6] * 3
            block_height = [18] * 3
            positions = [
                (0, 0),
                (0, 1),
                (0, 2),
            ]
            return block_width, block_height, positions

        else:
            raise_error(
                ValueError, "Number of qubits not supported by this architecture"
            )


def initialize_parameters(n_params, seed, parameters=None):
    tf.random.set_seed(seed)

    if parameters is None:
        return tf.Variable(
            tf.random.normal((n_params,), mean=1.0, stddev=1.0, dtype=tf.float32)
        )
    if parameters is not None:
        return parameters
