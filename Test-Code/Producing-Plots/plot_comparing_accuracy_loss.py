import numpy as np
import matplotlib.pyplot as plt
import argparse
from every_plot import loss_comparison, accuracy_comparison, gradient_comparison


qubit = [1, 2, 3, 4]
layers = int(input("Layers: "))
arch = str(input("Architecture: "))
path = [f"Digits-4x4-L/{arch}", f"Digits-4x4-G/{arch}"]
option = ["val", "train"]
size = 4

for q in qubit:
    for p in path:
        for opt in option:
            loss_comparison(q, layers, opt, size, p)
            accuracy_comparison(q, layers, opt, size, p)

# gradient_comparison(qubit, layers, size, path)
