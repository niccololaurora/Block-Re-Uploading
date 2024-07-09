import numpy as np
import tensorflow as tf
from qibo import hamiltonians
from qibo.symbols import Z, I


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

    if local == False:
        ham = I(0)
        for k in range(nqubits):
            ham *= Z(k)
        hamiltonian = hamiltonians.SymbolicHamiltonian(ham)
        return hamiltonian
    else:
        ham = Z(0)
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


def blocks_details(nqubits):
    if nqubits == 1:
        block_width = [8]
        block_height = [8]
        positions = [
            (0, 0),
        ]
        return block_width, block_height, positions
    if nqubits == 2:
        block_width = [4, 4]
        block_height = [8, 8]
        positions = [
            (0, 0),
            (0, 1),
        ]
        return block_width, block_height, positions
    if nqubits == 3:
        block_width = [3, 3, 2]
        block_height = [8, 8, 8]
        positions = [
            (0, 0),
            (0, 1),
            (0, 2),
        ]
        return block_width, block_height, positions
    if nqubits == 4:
        block_width = [4, 4, 4, 4]
        block_height = [4, 4, 4, 4]
        positions = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        return block_width, block_height, positions
    if nqubits == 9:
        block_width = [1, 2, 2, 2, 1, 2, 2, 2, 2]
        block_height = [3, 3, 3, 3, 3, 3, 3, 3, 3]
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
