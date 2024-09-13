import numpy as np
import tensorflow as tf
from datetime import datetime


def load_data(resize, selected_classes):
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_filter = np.isin(y_train, selected_classes)
    test_filter = np.isin(y_test, selected_classes)
    x_train = x_train[train_filter]
    y_train = y_train[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]

    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    x_train = tf.image.resize(x_train, [resize, resize])
    x_test = tf.image.resize(x_test, [resize, resize])

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    training_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return (
        training_data,
        test_data,
    )


def initialize_data_new(
    dataset, training_size, test_size, validation_size, resize, seed
):
    """Method which prepares the validation, training and test datasets."""

    # ==============
    # Choosing the seed is necessary to control the shuffling and
    # the digits that will be selected
    x_train, y_train, x_test, y_test = 0, 0, 0, 0
    digits = 0
    np.random.seed(seed)

    # ==============
    # Choosing dataset and digits/clothes
    if dataset == "digits":
        digits = [0, 1]
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if dataset == "fashion":
        # Fashion: boot and trousers
        digits = [1, 9]
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )

    # ==============
    # Select classes of interest
    mask_train = np.isin(y_train, digits)
    mask_test = np.isin(y_test, digits)
    x_train, y_train = x_train[mask_train], y_train[mask_train]
    x_test, y_test = x_test[mask_test], y_test[mask_test]

    # ==============
    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # ==============
    # Resizing
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    x_train = tf.image.resize(x_train, [resize, resize])
    x_test = tf.image.resize(x_test, [resize, resize])

    # ==============
    # Converting labels to tf.tensor
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Normalize labels to [0, 1]
    if len(digits) == 2:
        y_train = np.where(y_train == digits[0], 0, 1)
        y_test = np.where(y_test == digits[0], 0, 1)

    # ==============
    # Concatenate data
    images = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    # ==============
    # Check for duplicates
    images_vectors = images.reshape(len(images), -1)
    unique_images, unique_indices = np.unique(images_vectors, axis=0, return_index=True)

    # Crea un nuovo dataset senza duplicati
    images_no_duplicates = images[unique_indices]
    labels_no_duplicates = labels[unique_indices]

    # Verifica il numero di immagini dopo aver rimosso i duplicati
    print(f"Numero di immagini originali: {len(images)}")
    print(f"Numero di immagini senza duplicati: {len(images_no_duplicates)}")
    print(
        f"Numero di immagini con duplicato: {len(images) - len(images_no_duplicates)}"
    )

    # ==============
    # Perfect balance of the classes
    indices_dict = {}
    for class_label in digits:
        # Training
        indeces_class = np.where(labels_no_duplicates == class_label)[0]
        sampled_indices_train = np.random.choice(
            indeces_class,
            size=(training_size + validation_size + test_size) // len(digits),
            replace=False,
        )
        indices_dict[class_label] = sampled_indices_train

    # TRAINING, VALIDATION and TESTING INDECES
    train_indices_dict = {}
    validation_indices_dict = {}
    test_indices_dict = {}
    num_per_class = (training_size + validation_size + test_size) // len(digits)
    num_train_per_class = training_size // len(digits)
    num_validation_per_class = validation_size // len(digits)
    num_test_per_class = test_size // len(digits)

    for class_label, indices in indices_dict.items():
        # Shuffle indeces
        np.random.shuffle(indices)

        # Divide gli indici in training, validation e test set
        train_indices = indices[:num_train_per_class]
        validation_indices = indices[
            num_train_per_class : num_train_per_class + num_validation_per_class
        ]
        test_indices = indices[num_train_per_class + num_validation_per_class :]

        # Assegna agli dizionari
        train_indices_dict[class_label] = train_indices
        validation_indices_dict[class_label] = validation_indices
        test_indices_dict[class_label] = test_indices

    # Concatenate the training, validation and testing indeces of the different digits
    train_indices = np.concatenate(list(train_indices_dict.values()))
    validation_indices = np.concatenate(list(validation_indices_dict.values()))
    test_indices = np.concatenate(list(test_indices_dict.values()))

    # Shuffle the training, validation and testing indeces
    np.random.shuffle(train_indices)
    np.random.shuffle(validation_indices)
    np.random.shuffle(test_indices)

    # Define the training, validation and testing dataset
    x_val = images_no_duplicates[validation_indices]
    y_val = images_no_duplicates[validation_indices]
    x_train = images_no_duplicates[train_indices]
    y_train = images_no_duplicates[train_indices]
    x_test = images_no_duplicates[test_indices]
    y_test = images_no_duplicates[test_indices]

    # ==============
    # Converting labels to tf.tensor
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    training_data = (x_train, y_train)
    test_data = (x_test, y_test)
    validation_data = (x_val, y_val)

    return (
        training_data,
        test_data,
        validation_data,
    )


def initialize_data(dataset, training_size, test_size, validation_size, resize, seed):
    """Method which prepares the validation, training and test datasets."""

    # Choosing the seed is necessary to choose the shuffling and
    # the digits that will be selected
    x_train, y_train, x_test, y_test = 0, 0, 0, 0
    digits = 0
    np.random.seed(seed)

    # ==============
    # Choosing dataset and digits/clothes
    # Fashion: boot and trousers
    if dataset == "digits":
        digits = [0, 1]
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if dataset == "fashion":
        digits = [1, 9]
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )

    # ==============
    # Select classes of interest
    mask_train = np.isin(y_train, digits)
    mask_test = np.isin(y_test, digits)
    x_train, y_train = x_train[mask_train], y_train[mask_train]
    x_test, y_test = x_test[mask_test], y_test[mask_test]

    # ==============
    # Perfect balance of the classes
    train_val_indices = []
    test_indices = []
    for class_label in digits:
        # Training
        train_indices_class = np.where(y_train == class_label)[0]
        sampled_indices_train = np.random.choice(
            train_indices_class,
            size=(training_size + validation_size) // len(digits),
            replace=False,
        )
        train_val_indices.extend(sampled_indices_train)

        # Test
        test_indices_class = np.where(y_test == class_label)[0]
        sampled_indices_test = np.random.choice(
            test_indices_class, size=test_size // len(digits), replace=False
        )
        test_indices.extend(sampled_indices_test)

    # TEST
    # Shuffle the dataset
    np.random.shuffle(test_indices)
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    # TRAINING AND VALIDATION
    # Shuffle the indices, in order to obtain different
    # training and validation dataset
    np.random.shuffle(train_val_indices)
    val_indices = []
    train_indices = []
    for n in range(len(digits)):
        vindex_1 = (
            n * int(validation_size / len(digits))
            + int(training_size / len(digits)) * n
        )
        vindex_2 = (
            int(validation_size / len(digits)) * (n + 1)
            + int(training_size / len(digits)) * n
        )
        tindex_1 = n * int(training_size / len(digits)) + int(
            validation_size / len(digits)
        ) * (n + 1)
        tindex_2 = (n + 1) * int(training_size / len(digits)) + int(
            validation_size / len(digits)
        ) * (n + 1)
        validation_pick = train_val_indices[vindex_1:vindex_2]
        train_pick = train_val_indices[tindex_1:tindex_2]

        val_indices.extend(validation_pick)
        train_indices.extend(train_pick)

    # Shuffle the training and validation dataset
    # Definition of validation and training dataset
    np.random.shuffle(val_indices)
    np.random.shuffle(train_indices)
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    # ==============
    # Resizing
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)
    x_val = tf.expand_dims(x_val, axis=-1)

    x_train = tf.image.resize(x_train, [resize, resize])
    x_test = tf.image.resize(x_test, [resize, resize])
    x_val = tf.image.resize(x_val, [resize, resize])

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_val = x_val / 255.0

    # Normalize labels to [0, 1]
    if len(digits) == 2:
        y_train = np.where(y_train == digits[0], 0, 1)
        y_test = np.where(y_test == digits[0], 0, 1)
        y_val = np.where(y_val == digits[0], 0, 1)

    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    training_data = (x_train, y_train)
    test_data = (x_test, y_test)
    validation_data = (x_val, y_val)

    return (
        training_data,
        test_data,
        validation_data,
    )


def pooling_creator(blocks, nqubits, mode):
    """Method to calculate the max or the average value of each block of an image.

    Args:
        blocks (float list): List where each element corresponds to the blocks into which we have divided the image.

    Returns:
        List of max values.
    """

    if mode == "max":
        max_values = []
        for i in range(nqubits):
            block = tf.reshape(blocks[i], [-1])
            max_values.append(tf.reduce_max(block))
        return max_values

    if mode == "average":
        average_values = []
        for i in range(nqubits):
            block = tf.reshape(blocks[i], [-1])
            mean = tf.reduce_sum(block) / (width * height)
            average_values.append(mean)
        return average_values


def block_creator(image, block_height, block_width):
    """Method to partition an image into blocks.

    Args:
        image: MNIST image.

    Returns:
        List containing images' blocks.
    """

    blocks = []
    for i in range(0, image.shape[0], block_height):
        for j in range(0, image.shape[1], block_width):
            block = image[i : i + block_height, j : j + block_width]
            block = tf.reshape(block, [-1])
            blocks.append(block)
    return blocks


def create_blocks(image, block_width, block_height, positions):

    blocks = []

    w = 0
    h = 0
    r = 0
    c = 0
    row_point = 0

    for i, (row, column) in enumerate(positions):

        if r != row:
            h = block_height[i - 1]
            r = row
            row_point = 0

        if c != column:
            c = column
            w = block_width[i - 1]

        """
        print(f"Row {row}")
        print(f"Colomn {column}")
        print(f"Sizes {block_width[i]}, {block_height[i]}")
        print(
            f"Coordinate [{row * h} : {row * h + block_height[i]}][{row_point} : {row_point + block_width[i]}]"
        )
        """

        block = image[
            row * h : row * h + block_height[i],
            row_point : row_point + block_width[i],
        ]

        if r == row:
            row_point += block_width[i]

        block = tf.reshape(block, [-1])
        """
        print("=" * 60)
        print(block)
        print("=" * 60)
        """
        blocks.append(block)

    return blocks


def shuffle(data):
    current_time = datetime.now()
    seconds_str = current_time.strftime("%S")
    seconds = int(seconds_str)

    x = tf.random.shuffle(data[0], seed=seconds)
    y = tf.random.shuffle(data[1], seed=seconds)

    return x, y
