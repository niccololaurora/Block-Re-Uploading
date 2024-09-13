import sys
import os
import matplotlib.pyplot as plt

# Get the parent directory of the current file (test.py)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from data import initialize_data_new

dataset = "digits"
training_size = 100
test_size = 200
validation_size = 1000
resize = [4, 8]
seed = 1


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 6))

lista_assi = [ax1, ax3]
lista_assi_2 = [ax2, ax4]

for r, a, b in zip(resize, lista_assi, lista_assi_2):
    train, test, val = initialize_data_new(
        dataset, training_size, test_size, validation_size, r, seed
    )

    print(f"Type {type(train)}")
    print(f"Type {type(train[0])}")
    print(f"Type {type(train[1])}")
    print("=" * 30)
    print(f"Training size {train[0].shape}")
    print(f"Validation size {val[0].shape}")
    print(f"Test size {test[0].shape}")

    a.imshow(train[0][0], cmap="gray")
    b.imshow(train[0][7], cmap="gray")

    b.set_yticks([])
    a.set_yticks([])
    b.set_xticks([])
    a.set_xticks([])

# Minimize the space between columns
plt.subplots_adjust(wspace=0.0)

# Optionally, you can use tight_layout if necessary, but sometimes it may add space.
# plt.tight_layout()

name = f"Digits-complete.pdf"
plt.savefig(name, bbox_inches="tight")  # Save the figure with minimal extra space
