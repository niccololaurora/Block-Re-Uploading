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

## How can I use this code?

To use this code you need to change some parameters directly from the "main.py" file:

1. **dataset**: "digits", "fashion"
   You can choose between two datasets.

2. **resize**: "4", "8", "12", "14", "16", "18"
   The "resize" flag indicates the size of the images. You cannot values different from the list above, because the size
   of the image has implications on the entanglement structure of the circuit, which currently supports only those sizes.

3. **epochs**

4. **local**: "True" for local decoding, "False" for global decoding.

5. **entanglement**: "True" for entanglement, "False" for no entanglement (in a layer).

6. **pooling**: "max" for max pooling, "no" for no pooling.

7. **pretraining**: "True", "False".
   This flag must be "True" if you wanna load a set of pretrained parameters.

8. **criterion**: "No", 'target', 'fluctuation'.
   This flag lets you choose the convergence/stopping criterium: "No" option means that the training will continue up to the last epoch;
   "target" option means that the training will stop when the target loss value is reached; 'fluctuation' option means that the training will stop when the
   loss' fluctuations will be within 1e-04.
9. **iterazione**: str(1), str(2), str(3), ...
   If you wanna conduct a train of 200 epochs and you want to split it in 4 training of 50 epochs each, you must indicate here
   the iteration step. This number will be useful to produce plots, folders, files with the correct iteration number.

10. **layers**

11. **nqubits**

12. **learning_rate**

13. **seed_parameters**.
    Seed to generate the parameters of the architecture.

14. **loss**: "crossentropy"
    Other losses are supported, but are useless for our goals.

15. **training_size**

16. **validation_size**

17. **test_size**
    Useless. You can put whatever number you want.

18. **batch_size**
