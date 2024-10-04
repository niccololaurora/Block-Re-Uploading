import numpy as np
import matplotlib.pyplot as plt
import argparse
from comparison_accuracy_loss import (
    loss_comparison,
    accuracy_comparison,
    gradient_comparison,
)


qubit = [1, 2, 3, 4]
layers = int(input("Layers: "))
path = ""
option = ["val", "train"]
size = 4

for q in qubit:
    for opt in option:
        loss_comparison(q, layers, opt, size, path)
        accuracy_comparison(q, layers, opt, size, path)

# gradient_comparison(qubit, layers, size, path)
