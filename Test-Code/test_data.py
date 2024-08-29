from data import initialize_data
import matplotlib.pyplot as plt

dataset = "digits"
training_size = 10
test_size = 10
validation_size = 10
resize = 18
seed = 1

train, test, val = initialize_data(
    dataset, training_size, test_size, validation_size, resize, seed
)


print(f"Training {len(train[0])}")
print(f"Validation {len(val[0])}")
print(f"Test {len(test[0])}")
print("=" * 30)
print(f"Training size {train[0].shape}")
print(f"Validation size {val[0].shape}")
print(f"Test size {test[0].shape}")


WIDTH = 0.5
fig, (ax1, ax2) = plt.subplots(
    ncols=2, figsize=(10 * WIDTH * 3 / 2, 10 * WIDTH * 6 / 8)
)

ax1.imshow(train[0][0], cmap="gray")
ax2.imshow(train[0][7], cmap="gray")

ax2.set_yticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax1.set_xticks([])

# fig.suptitle("4 qubits: 3 layers", fontsize=12)
plt.tight_layout()
name = f"Prova-{resize}x{resize}-.pdf"
plt.savefig(name, bbox_inches="tight")
