import numpy as np
import pandas as pd

# Crea una lista vuota per memorizzare i gradienti
gradients_list = []

# Carica i gradienti per i modelli M0, M1, ..., M9
for i in range(10):
    file_name = f"trainability/gradients_M{i}.npy"
    grads = np.load(file_name, allow_pickle=True)
    print(grads[0:10])
    print("=" * 60)

    # Aggiungi i gradienti alla lista
    gradients_list.append(grads[0:10])

# Crea un DataFrame dai gradienti
gradients_df = pd.DataFrame(gradients_list, index=[f"M{i}" for i in range(10)])
column_names = [f"$\\theta_{{{i}}}$" for i in range(10)]
gradients_df.columns = column_names

# Stampa il DataFrame
print(gradients_df)

# Genera una tabella LaTeX
# latex_table = gradients_df.to_latex(index=True)
output_path = "trainability/output_table.tex"
gradients_df.to_latex(output_path, escape=False)
