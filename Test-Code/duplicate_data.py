import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Seleziono solo i digits di interess
digits = [0, 1]
mask_train = np.isin(y_train, digits)
mask_test = np.isin(y_test, digits)
x_train, y_train = x_train[mask_train], y_train[mask_train]
x_test, y_test = x_test[mask_test], y_test[mask_test]

print(f"Numero di immagini originali appena scaricate: {len(x_train)+ len(x_test)}")


# Unisce train e test set
images = np.concatenate((x_train, x_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

# Ridimensiona le immagini a 4x4
resize = 7
images = tf.expand_dims(images, axis=-1)  # Aggiunge la dimensione del canale
images = tf.image.resize(images, [resize, resize])
images = images.numpy().astype(np.float32) / 255.0  # Converti a numpy e normalizza

# Converti le immagini in vettori
images_vectors = images.reshape(len(images), -1)  # Appiattisci le immagini

# Identifica i vettori unici e gli indici corrispondenti
unique_images, unique_indices = np.unique(images_vectors, axis=0, return_index=True)

# Crea un nuovo dataset senza duplicati
images_no_duplicates = images[unique_indices]
labels_no_duplicates = labels[unique_indices]

# Verifica il numero di immagini dopo aver rimosso i duplicati
print(f"Numero di immagini originali: {len(images)}")
print(f"Numero di immagini senza duplicati: {len(images_no_duplicates)}")
print(f"Numero di immagini con duplicato: {len(images) - len(images_no_duplicates)}")


image_map = {}
for idx, vec in enumerate(images_vectors):
    key = tuple(vec)
    if key not in image_map:
        image_map[key] = []
    image_map[key].append(idx)

# Trova due indici di immagini identiche
duplicates = [indices for indices in image_map.values() if len(indices) > 1]

if len(duplicates) < 1 or len(duplicates[0]) < 2:
    print("Non ci sono abbastanza duplicati nel dataset.")
else:
    # Seleziona due indici duplicati per la visualizzazione
    idx1, idx2 = duplicates[0][:2]  # Prende i primi due indici dalla lista di duplicati

    # Estrai le immagini
    image1 = images[idx1].reshape(resize, resize)
    image2 = images[idx2].reshape(resize, resize)

    # Mostra le immagini affiancate
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(image1, cmap="gray")
    ax[0].set_title(f"Image {idx1}")
    ax[0].axis("off")

    ax[1].imshow(image2, cmap="gray")
    ax[1].set_title(f"Image {idx2}")
    ax[1].axis("off")

    plt.show()
