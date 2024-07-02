import numpy as np
import pandas as pd

# Crea una lista vuota per memorizzare i gradienti
gradients_list = []
params_list = []

# Carica i gradienti per i modelli M0, M1, ..., M9
for i in range(10):
    file_name = f"trainability/gradients_M{i}.npz"
    loaded_data = np.load(file_name, allow_pickle=True)
    grads = loaded_data["grads"]

    print(f"Grads {type(grads)}")
    vparams = loaded_data["vparams"]

    print(grads[0:10])
    print("=" * 60)

    # Aggiungi i gradienti alla lista
    params_list.append(vparams[:10])
    gradients_list.append(grads[:10])

# Crea un DataFrame dai gradienti
gradients_df = pd.DataFrame(gradients_list, index=[f"M{i}" for i in range(10)])
vparams_df = pd.DataFrame(params_list, index=[f"M{i}" for i in range(10)])
column_names = [f"$\\theta_{{{i}}}$" for i in range(10)]
gradients_df.columns = column_names
vparams_df.columns = column_names

# Variance
variances = gradients_df.var(axis=1)
print(f"Var {variances}")
variances_df = pd.DataFrame(variances, columns=["Variance"]).T
gradients_df = pd.concat([gradients_df, variances_df], axis=0)

# Stampa il DataFrame
print(gradients_df)
print(vparams_df)

output_path = "trainability/gradient_table.tex"
gradients_df.to_latex(output_path, escape=False)

output_path = "trainability/param_table.tex"
vparams_df.to_latex(output_path, escape=False)
