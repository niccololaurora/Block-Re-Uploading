## Files

1. **qclassifier.py**:

   - Contains the Block Re-Uploading class.
   - Supports three different loss functions: fidelity, weighted fidelity, binary cross-entropy (with both local or global Pauli Z).

2. **help_functions.py**:

   - Contains helpful functions.

3. **data.py**:

   - Contains the functions to manage the dataset.

4. **plot_functions.py**:

   - Contains the "Bloch" class (needed to build and plot the Bloch sphere).
   - Contains "plot_predictions" (plots images and corresponding predictions).
   - Contains "plot_sphere" (uses the Bloch class to plot the Bloch sphere).
   - Contains "plot_loss_accuracy" (plots accuracy and loss on the same canvas).

5. **trainability.py**:

   - Runs the Qclassifier for one epoch to evaluate the trainability of the chosen ansatz.
   - Saves the gradients in a folder named "trainability".

6. **variance.py**:
   - Plots the gradient curve using the files stored in the "trainability" folder.
