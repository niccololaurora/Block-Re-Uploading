from data import initialize_data
from plot_functions import plot_predictions


digits = [7, 8]
training_size = 20
test_size = 20
validation_size = 20
resize = 10
predictions = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6]

training_data, test_data, validation_data = initialize_data(
    digits, training_size, test_size, validation_size, resize
)

plot_predictions(predictions, training_data[0], training_data[1], "file_7_8.pdf", 4, 5)
